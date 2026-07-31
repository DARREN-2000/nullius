"""The imaging inference path, governed by the same rule as the text path.

NULLIUS treats a classifier exactly as it treats a language model: an untrusted
proposer whose output is only allowed through if it survives gates. For imaging
those gates are:

  1. de-identification - the model never sees a direct identifier, and the
     scrubber's own output is re-checked rather than trusted.
  2. acquisition quality - an out-of-focus, blown-out or empty frame is
     rejected as unusable instead of being confidently classified.
  3. distribution - features far outside the training envelope are refused,
     because a score computed there is meaningless however confident it looks.
  4. confidence - scores in the indeterminate band are handed to a human
     rather than rounded to a decision.

A refusal is a successful outcome. The metric that matters is not accuracy on
the cases it answers, it is whether anything unsafe got served.

Every served result carries an exact Shapley attribution over named features, so
"why" is answerable in the same breath as "what". Nine features means 512
coalitions, which is small enough to enumerate exhaustively - so the values are
the real Shapley values, not a sampled approximation of them.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

from . import imaging
from .dicom import Dataset, DicomError, decode, deidentify, read_file, residual_identifiers
from .imaging import FEATURE_LABELS, FEATURE_NAMES
from .observability import TRACER
from .onnx import load_session

DISCLAIMER = (
    "Decision support only. This is a triage signal over morphological features, "
    "not a diagnosis, and it does not replace dermoscopic assessment or biopsy."
)


@dataclass(frozen=True)
class VisionGates:
    """Which gates are armed. Ablation flips these one at a time."""

    deidentification: bool = True
    quality: bool = True
    distribution: bool = True
    confidence: bool = True

    def without(self, name: str) -> "VisionGates":
        if not hasattr(self, name):
            raise ValueError(f"unknown gate {name!r}")
        return replace(self, **{name: False})


VISION_GATE_NAMES = ("deidentification", "quality", "distribution", "confidence")


@dataclass(frozen=True)
class VisionThresholds:
    min_focus: float = 0.40
    max_clipped: float = 0.45
    min_area_fraction: float = 0.015
    max_area_fraction: float = 0.80
    max_feature_z: float = 4.0
    review_low: float = 0.35
    review_high: float = 0.65


@dataclass
class VisionResult:
    served: bool
    refusal_reason: str | None
    probability: float | None
    triage: str
    # These are unknown when the file cannot even be decoded, so a refusal must
    # be constructible without them.
    study_uid: str | None = None
    pseudonym: str | None = None
    patient_id: str | None = None
    features: dict[str, float] = field(default_factory=dict)
    quality: dict[str, float] = field(default_factory=dict)
    attributions: list[dict[str, Any]] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    diameter_mm: float | None = None
    max_feature_z: float | None = None
    baseline_probability: float | None = None
    attribution_residual: float | None = None
    backend: str = ""
    latency_ms: float = 0.0
    trace_id: str = ""
    detail: str = ""
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "served": self.served,
            "refusalReason": self.refusal_reason,
            "probability": self.probability,
            "triage": self.triage,
            "studyUid": self.study_uid,
            "pseudonym": self.pseudonym,
            "patientId": self.patient_id,
            "features": self.features,
            "quality": self.quality,
            "attributions": self.attributions,
            "steps": self.steps,
            "diameterMm": self.diameter_mm,
            "maxFeatureZ": self.max_feature_z,
            "baselineProbability": self.baseline_probability,
            "attributionResidual": self.attribution_residual,
            "backend": self.backend,
            "latencyMs": round(self.latency_ms, 3),
            "traceId": self.trace_id,
            "detail": self.detail,
            "disclaimer": self.disclaimer,
        }


class ModelBundle:
    """An .onnx graph plus the metadata needed to use it responsibly.

    A model file on its own is not deployable: without the training feature
    statistics you cannot tell in-distribution from out, and without the
    operating point you are just guessing at 0.5.
    """

    def __init__(self, model_path: str | Path, prefer_onnxruntime: bool = True) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = self.model_path.with_suffix(".json")
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"{self.model_path} not found - run `python3 scripts/train_lesion_model.py` first"
            )
        self.metadata: dict[str, Any] = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.feature_names: list[str] = self.metadata["feature_names"]
        if tuple(self.feature_names) != FEATURE_NAMES:
            raise ValueError("model was trained on a different feature contract than this build")
        self.mean: list[float] = self.metadata["feature_mean"]
        self.std: list[float] = self.metadata["feature_std"]
        self.operating_point: float = float(self.metadata["operating_point"])
        calibration = self.metadata.get("calibration") or {}
        self.calibration_a: float = float(calibration.get("a", 1.0))
        self.calibration_b: float = float(calibration.get("b", 0.0))
        self.calibrated: bool = bool(calibration)
        self.session = load_session(self.model_path, prefer_onnxruntime=prefer_onnxruntime)

    @property
    def backend(self) -> str:
        return self.session.backend

    def predict_raw(self, features: Sequence[float]) -> float:
        """The network's own output, before calibration."""
        return float(self.session.run(features)[0])

    def predict(self, features: Sequence[float]) -> float:
        """Calibrated probability.

        A raw sigmoid output is a score, not a probability: a network that says
        0.9 is not thereby right nine times in ten. Platt scaling (fitted on the
        training split only) maps the score onto something whose numbers mean
        what they say, which is the difference between a number a clinician can
        act on and a number that merely ranks.
        """
        raw = self.predict_raw(features)
        if not self.calibrated:
            return raw
        return apply_platt(raw, self.calibration_a, self.calibration_b)

    def feature_z(self, features: Sequence[float]) -> list[float]:
        return [
            abs(value - self.mean[i]) / max(self.std[i], 1e-6) for i, value in enumerate(features)
        ]

    def _coalition(self, features: Sequence[float], mask: int, cache: dict[int, float]) -> float:
        """Model output with features in `mask` present and the rest at baseline."""
        hit = cache.get(mask)
        if hit is not None:
            return hit
        vector = [
            features[i] if (mask >> i) & 1 else self.mean[i] for i in range(len(self.feature_names))
        ]
        value = self.predict(vector)
        cache[mask] = value
        return value

    def shapley(self, features: Sequence[float]) -> tuple[list[float], float, float]:
        """Exact Shapley values over the feature set, baseline = training mean.

        phi_i = sum over coalitions S not containing i of
                  |S|! (n-|S|-1)! / n!  *  [ v(S + i) - v(S) ]

        With n = 9 there are 2^9 = 512 coalitions, so every one is enumerated and
        cached: no Monte-Carlo sampling, no additivity assumption, no kernel
        regression. The values satisfy the efficiency axiom exactly, and the
        caller is handed the residual so it can verify that rather than trust it.
        """
        n = len(self.feature_names)
        cache: dict[int, float] = {}
        weights = [
            math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for size in range(n)
        ]
        phi = [0.0] * n
        for mask in range(1 << n):
            size = mask.bit_count()
            without = self._coalition(features, mask, cache)
            for i in range(n):
                if (mask >> i) & 1:
                    continue
                gain = self._coalition(features, mask | (1 << i), cache) - without
                phi[i] += weights[size] * gain
        return phi, cache[0], cache[(1 << n) - 1]

    def attribute(
        self, features: Sequence[float], probability: float
    ) -> tuple[list[dict[str, Any]], float, float]:
        """Exact Shapley attribution, returned with the evidence that it is exact.

        Returns (contributions, baseline, residual). `residual` is
        |f(x) - baseline - sum(phi)|, which the efficiency axiom requires to be
        zero up to floating point. It is reported rather than asserted, because
        an explanation that cannot be checked is decoration.
        """
        phi, baseline, full = self.shapley(features)
        residual = abs(full - baseline - sum(phi))
        contributions: list[dict[str, Any]] = []
        for index, name in enumerate(self.feature_names):
            delta = phi[index]
            contributions.append(
                {
                    "feature": name,
                    "label": FEATURE_LABELS[name],
                    "value": round(float(features[index]), 4),
                    "trainingMean": round(self.mean[index], 4),
                    "delta": round(delta, 4),
                    "direction": "raises" if delta > 0 else ("lowers" if delta < 0 else "neutral"),
                    "method": "exact-shapley",
                }
            )
        contributions.sort(key=lambda c: -abs(c["delta"]))
        return contributions, baseline, residual


class VisionPipeline:
    def __init__(
        self,
        bundle: ModelBundle,
        store: Any = None,
        gates: VisionGates | None = None,
        thresholds: VisionThresholds | None = None,
    ) -> None:
        self.bundle = bundle
        self.store = store
        self.gates = gates or VisionGates()
        self.thresholds = thresholds or VisionThresholds()

    # ------------------------------------------------------------------ helpers
    def _refuse(self, span: Any, reason: str, detail: str, **extra: Any) -> VisionResult:
        TRACER.metrics.inc("nullius_vision_refusals_total", reason=reason)
        span.set(**{"vision.served": False, "vision.refusal_reason": reason})
        return VisionResult(
            served=False,
            refusal_reason=reason,
            probability=None,
            triage="refused",
            trace_id=span.trace_id,
            detail=detail,
            backend=self.bundle.backend,
            **extra,
        )

    def _audit(self, actor: str, actor_role: str, patient_id: str | None, trace_id: str, detail: str) -> None:
        if self.store is not None:
            self.store.audit(
                actor=actor,
                actor_role=actor_role,
                action="vision.classify",
                patient_id=patient_id,
                trace_id=trace_id,
                detail=detail,
            )

    # ------------------------------------------------------------------- public
    def classify(
        self,
        source: str | Path | bytes,
        actor: str = "system",
        actor_role: str = "clinician",
        patient_id: str | None = None,
    ) -> VisionResult:
        started = time.perf_counter()
        with TRACER.span("vision.classify", **{"vision.backend": self.bundle.backend}) as span:
            # ---------------------------------------------------------- decode
            try:
                with TRACER.span("vision.dicom_decode"):
                    dataset = decode(source) if isinstance(source, (bytes, bytearray)) else read_file(source)
            except DicomError as exc:
                self._audit(actor, actor_role, patient_id, span.trace_id, f"refused: unreadable DICOM ({exc})")
                return self._refuse(span, "unreadable_dicom", str(exc))

            study_uid = dataset.get("StudyInstanceUID")
            source_patient = patient_id or str(dataset.get("PatientID", "")).lower() or None

            # ------------------------------------------------- de-identification
            with TRACER.span("vision.deidentify") as deid_span:
                clean, _mapping = deidentify(dataset)
                residual = residual_identifiers(clean)
                deid_span.set(**{"phi.residual_count": len(residual)})
            pseudonym = str(clean.get("PatientID", ""))
            if residual and self.gates.deidentification:
                TRACER.metrics.inc("nullius_phi_blocks_total")
                self._audit(actor, actor_role, source_patient, span.trace_id, "refused: residual identifiers")
                return self._refuse(
                    span,
                    "phi_not_removed",
                    f"identifiers still present after scrubbing: {', '.join(residual)}",
                    study_uid=study_uid,
                    patient_id=source_patient,
                )

            # ------------------------------------------------------ preprocess
            with TRACER.span("vision.preprocess") as pre_span:
                pre = imaging.preprocess(clean)
                pre_span.set(**{f"image.{k}": v for k, v in pre.quality.items()})
            common = {
                "study_uid": study_uid,
                "pseudonym": pseudonym,
                "patient_id": source_patient,
                "features": pre.features,
                "quality": pre.quality,
                "steps": pre.steps,
                "diameter_mm": pre.diameter_mm,
            }

            # --------------------------------------------------- quality gate
            quality_failures = self._quality_failures(pre.quality)
            if quality_failures and self.gates.quality:
                TRACER.metrics.inc("nullius_vision_quality_blocks_total")
                self._audit(actor, actor_role, source_patient, span.trace_id, "refused: image quality")
                return self._refuse(
                    span,
                    "image_quality_insufficient",
                    "; ".join(quality_failures),
                    **common,
                )

            # ---------------------------------------------- distribution gate
            vector = pre.vector()
            z_scores = self.bundle.feature_z(vector)
            max_z = max(z_scores) if z_scores else 0.0
            span.set(**{"vision.max_feature_z": round(max_z, 3)})
            TRACER.metrics.observe("nullius_vision_ood_distance", max_z)
            common["max_feature_z"] = round(max_z, 3)
            if max_z > self.thresholds.max_feature_z and self.gates.distribution:
                worst = FEATURE_NAMES[z_scores.index(max_z)]
                TRACER.metrics.inc("nullius_vision_ood_blocks_total")
                self._audit(actor, actor_role, source_patient, span.trace_id, "refused: out of distribution")
                return self._refuse(
                    span,
                    "out_of_distribution",
                    f"{worst} sits {max_z:.1f} SD from the training mean; the score would be extrapolation",
                    **common,
                )

            # ---------------------------------------------------- inference
            with TRACER.span("vision.inference", **{"model.path": str(self.bundle.model_path)}) as inf_span:
                inference_started = time.perf_counter()
                probability = self.bundle.predict(vector)
                inference_ms = (time.perf_counter() - inference_started) * 1000.0
                inf_span.set(**{"model.probability": round(probability, 4), "model.latency_ms": round(inference_ms, 3)})
            TRACER.metrics.inc("nullius_vision_inferences_total", backend=self.bundle.backend)
            TRACER.metrics.observe("nullius_vision_inference_ms", inference_ms)
            TRACER.metrics.observe("nullius_vision_probability", probability)

            # --------------------------------------------------- confidence
            low, high = self.thresholds.review_low, self.thresholds.review_high
            if low <= probability <= high and self.gates.confidence:
                TRACER.metrics.inc("nullius_vision_indeterminate_total")
                self._audit(actor, actor_role, source_patient, span.trace_id, "referred: indeterminate")
                result = self._refuse(
                    span,
                    "indeterminate_needs_review",
                    f"score {probability:.2f} falls in the indeterminate band [{low}, {high}]",
                    **common,
                )
                result.probability = round(probability, 4)
                result.triage = "human review"
                result.latency_ms = (time.perf_counter() - started) * 1000.0
                return result

            # ------------------------------------------------------- serve
            attributions, baseline_probability, attribution_residual = self.bundle.attribute(
                vector, probability
            )
            span.set(**{"vision.attribution_residual": attribution_residual})
            TRACER.metrics.observe("nullius_vision_attribution_residual", attribution_residual)
            triage = "refer for dermoscopic review" if probability >= self.bundle.operating_point else "routine surveillance"
            TRACER.metrics.inc("nullius_vision_served_total")
            span.set(
                **{
                    "vision.served": True,
                    "vision.probability": round(probability, 4),
                    "vision.triage": triage,
                }
            )
            self._audit(
                actor,
                actor_role,
                source_patient,
                span.trace_id,
                f"classified study {study_uid} p={probability:.3f} -> {triage}",
            )
            return VisionResult(
                served=True,
                refusal_reason=None,
                probability=round(probability, 4),
                triage=triage,
                attributions=attributions,
                baseline_probability=round(baseline_probability, 4),
                attribution_residual=attribution_residual,
                backend=self.bundle.backend,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                trace_id=span.trace_id,
                detail=f"operating point {self.bundle.operating_point:.2f}",
                **common,
            )

    def _quality_failures(self, quality: dict[str, float]) -> list[str]:
        t = self.thresholds
        failures: list[str] = []
        if quality["focus_score"] < t.min_focus:
            failures.append(f"focus score {quality['focus_score']:.2f} below {t.min_focus}")
        if quality["clipped_fraction"] > t.max_clipped:
            failures.append(f"{quality['clipped_fraction']:.0%} of pixels clipped")
        if quality["lesion_area_fraction"] < t.min_area_fraction:
            failures.append("no lesion of usable size was segmented")
        if quality["lesion_area_fraction"] > t.max_area_fraction:
            failures.append("segmented region fills the frame; the lesion is not bounded")
        return failures


# ------------------------------------------------------------------- metrics
def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Rank-based AUROC with tie handling (Mann-Whitney U)."""
    pairs = sorted(zip(scores, labels), key=lambda p: p[0])
    ranks: list[float] = [0.0] * len(pairs)
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[position] = average
        index = end + 1
    positives = sum(1 for _, label in pairs if label == 1)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    rank_sum = sum(rank for rank, (_, label) in zip(ranks, pairs) if label == 1)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def sensitivity_at_specificity(
    scores: Sequence[float], labels: Sequence[int], target_specificity: float = 0.95
) -> tuple[float, float]:
    """Highest sensitivity reachable while holding specificity at or above target."""
    best = (0.0, 1.0)
    for threshold in sorted(set(scores)) + [1.01]:
        tp = sum(1 for s, l in zip(scores, labels) if l == 1 and s >= threshold)
        fn = sum(1 for s, l in zip(scores, labels) if l == 1 and s < threshold)
        fp = sum(1 for s, l in zip(scores, labels) if l == 0 and s >= threshold)
        tn = sum(1 for s, l in zip(scores, labels) if l == 0 and s < threshold)
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        if specificity >= target_specificity and sensitivity > best[0]:
            best = (sensitivity, threshold)
    return best


def specificity_at_sensitivity(
    scores: Sequence[float], labels: Sequence[int], target_sensitivity: float = 0.90
) -> tuple[float, float]:
    """Highest specificity reachable while holding sensitivity at or above target.

    The mirror of `sensitivity_at_specificity`, and the correct one for a triage
    tool. Missing a melanoma and over-referring a benign mole are not errors of
    equal cost, so the threshold is pinned to the error that matters and the
    other one is reported as the price paid.
    """
    best = (0.0, 1.01)
    for threshold in sorted(set(scores)) + [1.01]:
        tp = sum(1 for s, l in zip(scores, labels) if l == 1 and s >= threshold)
        fn = sum(1 for s, l in zip(scores, labels) if l == 1 and s < threshold)
        fp = sum(1 for s, l in zip(scores, labels) if l == 0 and s >= threshold)
        tn = sum(1 for s, l in zip(scores, labels) if l == 0 and s < threshold)
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        if sensitivity >= target_sensitivity and specificity > best[0]:
            best = (specificity, threshold)
    return best


# --------------------------------------------------------------- calibration
def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def fit_platt(
    scores: Sequence[float],
    labels: Sequence[int],
    iterations: int = 4000,
    learning_rate: float = 0.08,
) -> tuple[float, float]:
    """Fit Platt scaling  p = sigmoid(a * logit(s) + b)  by gradient descent.

    Platt's own correction for small samples is used for the targets: instead of
    regressing onto 0 and 1, which drives the fit to overconfidence on a few
    dozen points, it regresses onto (1/(N+2), (N+1)/(N+2)).
    """
    positives = sum(1 for l in labels if l == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 1.0, 0.0
    high = (positives + 1.0) / (positives + 2.0)
    low = 1.0 / (negatives + 2.0)
    targets = [high if l == 1 else low for l in labels]
    inputs = [_logit(s) for s in scores]

    a, b = 1.0, 0.0
    n = float(len(inputs))
    for _ in range(iterations):
        grad_a = 0.0
        grad_b = 0.0
        for x, t in zip(inputs, targets):
            error = _sigmoid(a * x + b) - t
            grad_a += error * x
            grad_b += error
        a -= learning_rate * grad_a / n
        b -= learning_rate * grad_b / n
    return a, b


def apply_platt(score: float, a: float, b: float) -> float:
    return _sigmoid(a * _logit(score) + b)


def expected_calibration_error(scores: Sequence[float], labels: Sequence[int], bins: int = 10) -> float:
    """ECE: mean gap between confidence and observed frequency, weighted by bin size.

    A model that says 0.9 should be right about nine times in ten. Accuracy
    alone cannot tell you whether that is true, and a triage score people act on
    is only useful if its numbers mean what they say.
    """
    if not scores:
        return float("nan")
    total = len(scores)
    error = 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        members = [
            (s, l) for s, l in zip(scores, labels) if (s > low or (b == 0 and s >= low)) and s <= high
        ]
        if not members:
            continue
        confidence = sum(s for s, _ in members) / len(members)
        frequency = sum(l for _, l in members) / len(members)
        error += (len(members) / total) * abs(confidence - frequency)
    return error
