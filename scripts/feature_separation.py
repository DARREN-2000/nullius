"""Recompute per-feature class separation on the current generator.

The honest generator (ADR-013) changed the cohort, which invalidated the old
Cohen's d table in the README and ADR-009. Deleting an inconvenient null result
is exactly the failure mode ADR-009 warns about, so it gets recomputed instead.

Run: python3 scripts/feature_separation.py
"""

from __future__ import annotations

import math
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nullius.dicom import read_file  # noqa: E402
from nullius.imaging import FEATURE_NAMES, preprocess  # noqa: E402
from nullius.lesions import generate_cohort  # noqa: E402


def auroc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUROC with tie handling, so a flat feature scores 0.5."""
    pairs = sorted(zip(scores, labels), key=lambda p: p[0])
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = average
        i = j + 1
    rank_sum = sum(r for r, (_, label) in zip(ranks, pairs) if label == 1)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        studies = generate_cohort(Path(tmp), count=80, seed=20260731)
        rows = []
        for study in studies:
            dataset = read_file(study.path)
            rows.append((preprocess(dataset).features, study.label))

    labels = [y for _, y in rows]
    table = []
    for name in FEATURE_NAMES:
        positives = [f[name] for f, y in rows if y == 1]
        negatives = [f[name] for f, y in rows if y == 0]
        pooled = math.sqrt(
            (statistics.pstdev(positives) ** 2 + statistics.pstdev(negatives) ** 2) / 2.0
        ) or 1e-9
        d = abs(statistics.mean(positives) - statistics.mean(negatives)) / pooled
        a = auroc([f[name] for f, _ in rows], labels)
        table.append((name, d, max(a, 1.0 - a)))

    print(f"{'feature':<24}{'cohen_d':>10}{'auroc':>9}")
    print("-" * 43)
    for name, d, a in sorted(table, key=lambda r: r[1], reverse=True):
        print(f"{name:<24}{d:>10.3f}{a:>9.3f}")
    print("-" * 43)
    ds = [d for _, d, _ in table]
    best = max(table, key=lambda r: r[2])
    print(f"cohort                   : {len(rows)} studies, {sum(labels)} suspicious")
    print(f"cohen d range            : {min(ds):.2f} to {max(ds):.2f}")
    print(f"best single feature      : {best[0]} (AUROC {best[2]:.3f}, d {best[1]:.2f})")
    print("A best-single-feature AUROC well below 1.0 is the point: no lone")
    print("measurement separates the classes, so the model has to combine them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
