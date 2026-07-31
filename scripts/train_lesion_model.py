"""Train the lesion classifier and export it as a real .onnx file.

Pure-Python SGD on a 9 -> 12 -> 1 MLP over named morphological features. No
tensor library, because at this size none is needed and the no-install rule
(ADR-002) is worth more than the speed.

Two choices worth naming, because both are places where benchmarks are usually
quietly inflated:

  * The operating point is selected on the TRAINING split, never on test.
    Choosing a threshold on the test set and then reporting test sensitivity is
    leakage, and it is extremely common.
  * Feature statistics for the out-of-distribution gate come from training data
    only, for the same reason.

The resulting metrics are a pipeline-integrity check on synthetic data. They are
not evidence of clinical performance. See ADR-009.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nullius import dicom, imaging, onnx  # noqa: E402
from nullius.lesions import generate_cohort  # noqa: E402
from nullius.vision import (  # noqa: E402
    apply_platt,
    auroc,
    expected_calibration_error,
    fit_platt,
    specificity_at_sensitivity,
)

SEED = 20260731
COHORT_SIZE = 80
# A missed melanoma costs more than an unnecessary referral, so the operating
# point is pinned to sensitivity and the specificity it costs is reported.
TARGET_SENSITIVITY = 0.90
HIDDEN = 12
EPOCHS = 400
LEARNING_RATE = 0.35
L2 = 1e-4


def featurise(studies: list) -> list[dict]:
    rows = []
    for study in studies:
        dataset = dicom.read_file(study.path)
        clean, _ = dicom.deidentify(dataset)
        pre = imaging.preprocess(clean)
        rows.append(
            {
                "study_id": study.study_id,
                "patient_id": study.patient_id,
                "path": study.path,
                "split": study.split,
                "label": study.label,
                "body_site": study.body_site,
                "truth_note": study.truth_note,
                "severity": study.severity,
                "atypical": study.atypical,
                "features": pre.features,
                "quality": pre.quality,
                "diameter_mm": pre.diameter_mm,
                "vector": pre.vector(),
            }
        )
    return rows


def train(rows: list[dict]) -> tuple[list[list[float]], list[float], list[list[float]], list[float]]:
    rng = random.Random(SEED)
    n_in = len(imaging.FEATURE_NAMES)
    # He-style initialisation keeps early ReLU activations alive.
    w1 = [[rng.gauss(0, math.sqrt(2.0 / n_in)) for _ in range(HIDDEN)] for _ in range(n_in)]
    b1 = [0.0] * HIDDEN
    w2 = [[rng.gauss(0, math.sqrt(2.0 / HIDDEN))] for _ in range(HIDDEN)]
    b2 = [0.0]

    order = list(range(len(rows)))
    for epoch in range(EPOCHS):
        rng.shuffle(order)
        for index in order:
            x = rows[index]["vector"]
            y = float(rows[index]["label"])
            hidden_raw = [sum(x[i] * w1[i][j] for i in range(n_in)) + b1[j] for j in range(HIDDEN)]
            hidden = [v if v > 0 else 0.0 for v in hidden_raw]
            logit = sum(hidden[j] * w2[j][0] for j in range(HIDDEN)) + b2[0]
            probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))

            d_logit = probability - y
            for j in range(HIDDEN):
                grad = hidden[j] * d_logit + L2 * w2[j][0]
                d_hidden = w2[j][0] * d_logit if hidden_raw[j] > 0 else 0.0
                w2[j][0] -= LEARNING_RATE * grad
                b1[j] -= LEARNING_RATE * d_hidden
                for i in range(n_in):
                    w1[i][j] -= LEARNING_RATE * (x[i] * d_hidden + L2 * w1[i][j])
            b2[0] -= LEARNING_RATE * d_logit
    return w1, b1, w2, b2


def main() -> int:
    started = time.time()
    data_dir = ROOT / "data" / "dicom"
    model_dir = ROOT / "models"

    print(f"generating {COHORT_SIZE} synthetic DICOM studies -> {data_dir}")
    studies = generate_cohort(data_dir, count=COHORT_SIZE, seed=SEED)
    print(f"extracting {len(imaging.FEATURE_NAMES)} features per study")
    rows = featurise(studies)

    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    print(f"train {len(train_rows)} / test {len(test_rows)}")

    w1, b1, w2, b2 = train(train_rows)

    model_path = onnx.export_mlp(model_dir / "lesion-mlp.onnx", w1, b1, w2, b2)
    session = onnx.load_session(model_path)
    print(f"exported {model_path.name} ({model_path.stat().st_size} bytes), backend={session.backend}")

    train_raw = [session.run(r["vector"])[0] for r in train_rows]
    train_labels = [r["label"] for r in train_rows]

    # Platt scaling fitted on TRAIN ONLY. Fitting a calibrator on the test split
    # and then reporting test ECE is leakage of exactly the same kind as picking
    # a threshold there.
    platt_a, platt_b = fit_platt(train_raw, train_labels)
    train_scores = [apply_platt(s, platt_a, platt_b) for s in train_raw]
    print(f"Platt scaling fitted on TRAIN only: a={platt_a:.4f} b={platt_b:.4f}")

    specificity, operating_point = specificity_at_sensitivity(
        train_scores, train_labels, TARGET_SENSITIVITY
    )
    print(
        f"operating point chosen on TRAIN only: {operating_point:.4f} "
        f"(spec {specificity:.3f} @ sens {TARGET_SENSITIVITY:.2f})"
    )

    columns = list(zip(*[r["vector"] for r in train_rows]))
    metadata = {
        "name": "nullius-lesion-mlp",
        "version": "1.0.0",
        "architecture": f"{len(imaging.FEATURE_NAMES)}-{HIDDEN}-1 MLP, ReLU, sigmoid output",
        "feature_names": list(imaging.FEATURE_NAMES),
        "feature_mean": [round(statistics.fmean(c), 6) for c in columns],
        "feature_std": [round(max(statistics.pstdev(c), 1e-3), 6) for c in columns],
        "operating_point": round(operating_point, 6),
        "operating_point_rule": (
            f"highest specificity at sensitivity >= {TARGET_SENSITIVITY:.2f}, chosen on train"
        ),
        "calibration": {"method": "platt", "a": round(platt_a, 6), "b": round(platt_b, 6)},
        "trained_on": "synthetic dermoscopy generator (nullius.lesions)",
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "seed": SEED,
        "epochs": EPOCHS,
        "caveat": (
            "Synthetic data. These metrics measure pipeline integrity, not clinical validity. "
            "Not a medical device. See ADR-009."
        ),
    }
    metadata_path = model_dir / "lesion-mlp.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    features_path = model_dir / "lesion-features.json"
    features_path.write_text(
        json.dumps([{k: v for k, v in r.items() if k != "vector"} for r in rows], indent=2) + "\n",
        encoding="utf-8",
    )

    test_raw = [session.run(r["vector"])[0] for r in test_rows]
    test_scores = [apply_platt(s, platt_a, platt_b) for s in test_raw]
    test_labels = [r["label"] for r in test_rows]
    held_out_sensitivity = sum(
        1 for s, l in zip(test_scores, test_labels) if l == 1 and s >= operating_point
    ) / max(1, sum(test_labels))
    held_out_specificity = sum(
        1 for s, l in zip(test_scores, test_labels) if l == 0 and s < operating_point
    ) / max(1, len(test_labels) - sum(test_labels))
    atypical = sum(1 for r in rows if r.get("atypical"))
    print("\nheld-out test (synthetic - pipeline integrity, not clinical evidence):")
    print(f"  AUROC                     {auroc(test_scores, test_labels):.3f}")
    print(f"  sensitivity @ train point {held_out_sensitivity:.3f}")
    print(f"  specificity @ train point {held_out_specificity:.3f}")
    print(f"  ECE before calibration    {expected_calibration_error(test_raw, test_labels):.3f}")
    print(f"  ECE after calibration     {expected_calibration_error(test_scores, test_labels):.3f}")
    print(f"  atypical cases in cohort  {atypical}/{len(rows)} (label contradicts morphology)")
    print(f"\ndone in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
