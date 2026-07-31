"""90-day CKD-progression risk: declared coefficients, abstention, exact Shapley.

Read this docstring before the code, because the most important decision here is
a negative one.

This module does NOT train a risk model. The obvious move - fit a logistic
regression on the synthetic cohort and publish an AUROC - would be dishonest
twice over. There are twelve patients, and more importantly there are no
outcomes: nobody in this cohort has actually progressed or not progressed. A
model fitted on invented outcomes would produce a number that looks like
evidence and is not. The imaging side of this repo already learned that lesson
the expensive way (see ADR-013: an AUROC of 1.000 was a bug report, not a
result).

So the model is an explicit additive log-odds score with coefficients written
down in the source, in the direction and rough magnitude that KDIGO 2024 and the
wider CKD literature describe. It is a transparent prior, not a learned
posterior, and `validated` is False everywhere it is reported. What is real, and
what this module exists to demonstrate, is the machinery around the number:

  * it refuses when the inputs do not support an answer, rather than imputing;
  * it refuses when the inputs are present but stale relative to the patient's
    own monitoring cadence;
  * it refuses on physiologically impossible values, which are data errors;
  * but it does NOT refuse on extreme-yet-real values - see the note on the
    plausibility gate below, which was a genuine bug;
  * it abstains in an indeterminate band that WIDENS as inputs go missing;
  * every point of the score is attributed to a named factor by exact Shapley
    over all 2^7 coalitions, so the parts sum to the whole exactly;
  * every assessment is audited and traced.

A deployable version replaces `COEFFICIENTS` with a fit on real outcomes and
sets `validated`. Nothing else in the file needs to change - which is the point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Iterable

from .observability import TRACER

DISCLAIMER = (
    "Decision support only. This score uses declared, unvalidated coefficients on "
    "synthetic data. It is not a validated risk model and must not be used to make "
    "decisions about a real patient."
)

NEPHROTOXIC_RXNORM = {
    "5640": "ibuprofen (NSAID)",
    "7258": "naproxen (NSAID)",
    "4815": "gentamicin (aminoglycoside)",
}
NEPHROTOXIC_TERMS = ("ibuprofen", "naproxen", "diclofenac", "gentamicin", "ketorolac")


@dataclass(frozen=True)
class RiskFactor:
    """One named term in the score.

    `per` is the clinically meaningful step: the coefficient is the change in
    log-odds for one `per`-sized move away from `reference` in the harmful
    direction. Writing it this way means the coefficients can be read and argued
    with by a clinician, which a raw standardised weight cannot be.
    """

    key: str
    label: str
    unit: str
    reference: float
    per: float
    harmful: str  # "higher" or "lower"
    coefficient: float
    source: str
    loinc: str | None = None
    log_scale: bool = False
    max_steps: float = 4.0
    # Physiologically possible range. Outside this the value is a data error,
    # not a sick patient, and the score must refuse. Inside it, an extreme value
    # is clamped and flagged but still scored - see `clamped` vs `implausible`.
    plausible_min: float = float("-inf")
    plausible_max: float = float("inf")

    def implausible(self, value: float) -> bool:
        return not (self.plausible_min <= value <= self.plausible_max)

    def steps(self, value: float) -> float:
        """Signed distance from reference in `per` units, positive = more risk."""
        observed = math.log10(max(value, 0.1)) if self.log_scale else value
        anchor = math.log10(max(self.reference, 0.1)) if self.log_scale else self.reference
        delta = (observed - anchor) / self.per
        return delta if self.harmful == "higher" else -delta

    def clamped(self, value: float) -> tuple[float, bool]:
        raw = self.steps(value)
        bounded = max(-self.max_steps, min(self.max_steps, raw))
        return bounded, bounded != raw


# Directions and rough magnitudes follow KDIGO 2024 CKD progression risk factors.
# The values are declared, not fitted. See the module docstring.
FACTORS: tuple[RiskFactor, ...] = (
    RiskFactor("egfr", "eGFR", "mL/min/1.73m2", 60.0, 10.0, "lower", 0.55,
               "KDIGO 2024: GFR category is the primary axis of risk", loinc="33914-3",
               plausible_min=1.0, plausible_max=200.0),
    RiskFactor("egfr_slope", "eGFR trend", "mL/min per 30d", 0.0, 1.0, "lower", 0.45,
               "KDIGO 2024: sustained decline defines progression",
               plausible_min=-60.0, plausible_max=60.0),
    RiskFactor("uacr", "Urine albumin/creatinine", "mg/g", 30.0, 1.0, "higher", 0.60,
               "KDIGO 2024: albuminuria category, log-linear with risk",
               loinc="9318-7", log_scale=True, plausible_min=0.0, plausible_max=25000.0),
    RiskFactor("potassium", "Potassium", "mmol/L", 4.5, 0.5, "higher", 0.30,
               "Hyperkalaemia limits RAASi therapy and marks tubular dysfunction",
               loinc="2823-3", plausible_min=1.5, plausible_max=9.5),
    RiskFactor("haemoglobin", "Haemoglobin", "g/dL", 13.0, 1.0, "lower", 0.25,
               "Anaemia of CKD tracks with declining function", loinc="718-7",
               plausible_min=2.0, plausible_max=25.0),
    RiskFactor("age", "Age", "years", 60.0, 10.0, "higher", 0.20,
               "Age-related eGFR decline", plausible_min=0.0, plausible_max=125.0),
    RiskFactor("nephrotoxic_exposure", "Nephrotoxic medication", "present", 0.0, 1.0,
               "higher", 0.50, "Active NSAID or aminoglycoside exposure", max_steps=1.0,
               plausible_min=0.0, plausible_max=1.0),
)

FACTOR_KEYS = tuple(f.key for f in FACTORS)
INTERCEPT = -2.20


@dataclass(frozen=True)
class RiskGates:
    """Each gate is a reason to say nothing. Disarm them one at a time in the
    ablation and watch what leaks through."""

    completeness: bool = True
    staleness: bool = True
    plausibility: bool = True
    confidence: bool = True

    def without(self, name: str) -> "RiskGates":
        if name not in RISK_GATE_NAMES:
            raise ValueError(f"unknown gate {name!r}")
        return replace(self, **{name: False})


RISK_GATE_NAMES = ("completeness", "staleness", "plausibility", "confidence")


@dataclass(frozen=True)
class RiskThresholds:
    max_lag_days: float = 180.0
    review_low: float = 0.25
    review_high: float = 0.45
    # Only the factors without which the score means nothing. UACR is a strong
    # predictor but is genuinely missing for many real patients, and refusing
    # every patient with a monitoring gap would make the module useless on
    # exactly the records that most need review. Missing optional factors widen
    # the abstention band instead - see `widen_per_missing`.
    required: tuple[str, ...] = ("egfr", "potassium", "age")
    widen_per_missing: float = 0.06


@dataclass
class RiskResult:
    served: bool
    refusal_reason: str | None
    patient_id: str
    probability: float | None = None
    band: str | None = None
    baseline_probability: float | None = None
    attribution_residual: float | None = None
    attributions: list[dict[str, Any]] = field(default_factory=list)
    inputs: list[dict[str, Any]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    clamped: list[str] = field(default_factory=list)
    horizon_days: int = 90
    validated: bool = False
    calibrated: bool = False
    trace_id: str | None = None
    detail: str = ""
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "served": self.served,
            "refusalReason": self.refusal_reason,
            "patientId": self.patient_id,
            "probability": self.probability,
            "band": self.band,
            "baselineProbability": self.baseline_probability,
            "attributionResidual": self.attribution_residual,
            "attributions": self.attributions,
            "inputs": self.inputs,
            "missing": self.missing,
            "stale": self.stale,
            "clamped": self.clamped,
            "horizonDays": self.horizon_days,
            "validated": self.validated,
            "calibrated": self.calibrated,
            "traceId": self.trace_id,
            "detail": self.detail,
            "disclaimer": self.disclaimer,
        }


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


class RiskModel:
    """Additive log-odds over the declared factors, with exact Shapley.

    Calibration is supported but off by default, and `calibrated` is reported
    honestly. Calibrating a score whose coefficients were never fitted would be
    polishing a guess.
    """

    def __init__(
        self,
        factors: Iterable[RiskFactor] = FACTORS,
        intercept: float = INTERCEPT,
        calibration: tuple[float, float] | None = None,
    ) -> None:
        self.factors = tuple(factors)
        self.intercept = intercept
        self.calibration = calibration
        self.calibrated = calibration is not None

    def logit(self, steps: dict[str, float]) -> float:
        total = self.intercept
        for factor in self.factors:
            total += factor.coefficient * steps.get(factor.key, 0.0)
        return total

    def predict_raw(self, steps: dict[str, float]) -> float:
        return _sigmoid(self.logit(steps))

    def predict(self, steps: dict[str, float]) -> float:
        raw = self.predict_raw(steps)
        if not self.calibration:
            return raw
        a, b = self.calibration
        eps = 1e-6
        clipped = min(max(raw, eps), 1 - eps)
        return _sigmoid(a * math.log(clipped / (1 - clipped)) + b)

    def shapley(self, steps: dict[str, float]) -> tuple[list[float], float, float]:
        """Exact Shapley values over 2^7 = 128 coalitions.

        The baseline is every factor at its clinical reference value, so an
        attribution answers the question a clinician actually asks: how much of
        this patient's risk above a reference patient comes from each finding.
        """
        keys = [f.key for f in self.factors]
        n = len(keys)
        cache: dict[int, float] = {}

        def value(mask: int) -> float:
            if mask not in cache:
                active = {keys[i]: steps.get(keys[i], 0.0) for i in range(n) if mask & (1 << i)}
                cache[mask] = self.predict(active)
            return cache[mask]

        weights = [
            math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for size in range(n)
        ]
        phi = [0.0] * n
        for mask in range(1 << n):
            size = bin(mask).count("1")
            for i in range(n):
                if mask & (1 << i):
                    continue
                phi[i] += weights[size] * (value(mask | (1 << i)) - value(mask))
        return phi, cache[0], cache[(1 << n) - 1]

    def attribute(self, steps: dict[str, float]) -> tuple[list[dict[str, Any]], float, float]:
        phi, baseline, full = self.shapley(steps)
        contributions = [
            {
                "factor": factor.key,
                "label": factor.label,
                "steps": round(steps.get(factor.key, 0.0), 4),
                "coefficient": factor.coefficient,
                "delta": phi[index],
                "direction": "increases risk" if phi[index] >= 0 else "decreases risk",
                "source": factor.source,
                "method": "exact-shapley",
            }
            for index, factor in enumerate(self.factors)
        ]
        contributions.sort(key=lambda c: abs(c["delta"]), reverse=True)
        residual = abs((full - baseline) - sum(phi))
        return contributions, baseline, residual


class RiskAssessor:
    """Extracts factors from the record, runs the gates, then the model."""

    def __init__(
        self,
        store,
        model: RiskModel | None = None,
        gates: RiskGates | None = None,
        thresholds: RiskThresholds | None = None,
    ) -> None:
        self.store = store
        self.model = model or RiskModel()
        self.gates = gates or RiskGates()
        self.thresholds = thresholds or RiskThresholds()

    # -- extraction -------------------------------------------------------

    def factors_with_loinc(self) -> list[RiskFactor]:
        return [f for f in self.model.factors if f.loinc]

    def _latest(self, patient_id: str, loinc: str) -> tuple[float, str] | None:
        rows = [r for r in self.store.observations(patient_id, loinc) if r.get("value") is not None]
        if not rows:
            return None
        newest = max(rows, key=lambda r: r["effective_at"])
        return float(newest["value"]), newest["effective_at"]

    def _slope_per_30d(self, patient_id: str, loinc: str) -> float | None:
        rows = [r for r in self.store.observations(patient_id, loinc) if r.get("value") is not None]
        if len(rows) < 3:
            return None
        rows.sort(key=lambda r: r["effective_at"])
        origin = _parse(rows[0]["effective_at"])
        xs = [(_parse(r["effective_at"]) - origin).total_seconds() / 86400.0 for r in rows]
        ys = [float(r["value"]) for r in rows]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        denominator = sum((x - mean_x) ** 2 for x in xs)
        if denominator == 0:
            return None
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
        return slope * 30.0

    def extract(self, patient_id: str) -> dict[str, dict[str, Any]]:
        """Pull every factor the score needs, recording provenance for each."""
        patient = self.store.patient(patient_id)
        if not patient:
            raise KeyError(patient_id)

        observations = self.store.observations(patient_id)
        clock = max((r["effective_at"] for r in observations), default=None)
        found: dict[str, dict[str, Any]] = {}

        for factor in self.factors_with_loinc():
            latest = self._latest(patient_id, factor.loinc or "")
            if latest:
                value, at = latest
                found[factor.key] = {"value": value, "at": at, "source": f"LOINC {factor.loinc}"}

        slope = self._slope_per_30d(patient_id, "33914-3")
        if slope is not None:
            found["egfr_slope"] = {
                "value": slope,
                "at": clock,
                "source": "least-squares slope over all eGFR results",
            }

        birth = patient.get("birth_date")
        if birth and clock:
            years = (_parse(clock) - _parse(birth + "T00:00:00Z")).days / 365.25
            found["age"] = {"value": round(years, 1), "at": clock, "source": "patient.birth_date"}

        medications = [m for m in self.store.medications(patient_id) if m.get("status") == "active"]
        hits = [
            m["display"]
            for m in medications
            if m.get("rxnorm") in NEPHROTOXIC_RXNORM
            or any(term in (m.get("display") or "").lower() for term in NEPHROTOXIC_TERMS)
        ]
        found["nephrotoxic_exposure"] = {
            "value": 1.0 if hits else 0.0,
            "at": clock,
            "source": "; ".join(hits) if hits else "no active nephrotoxic medication",
        }
        return found

    # -- assessment -------------------------------------------------------

    def assess(
        self,
        patient_id: str,
        actor: str = "system",
        actor_role: str = "clinician",
    ) -> RiskResult:
        with TRACER.span("risk.assess", **{"patient.id": patient_id}) as span:
            found = self.extract(patient_id)
            thresholds = self.thresholds

            missing_required = [key for key in thresholds.required if key not in found]
            if self.gates.completeness and missing_required:
                return self._refuse(
                    patient_id, "insufficient_observations", span, missing=missing_required,
                    detail="required inputs absent: " + ", ".join(missing_required),
                )

            clock = max((v["at"] for v in found.values() if v.get("at")), default=None)
            stale: list[str] = []
            stale_required = False
            if clock:
                for key, entry in found.items():
                    if not entry.get("at"):
                        continue
                    lag = (_parse(clock) - _parse(entry["at"])).days
                    if lag > thresholds.max_lag_days:
                        stale.append(f"{key} ({lag}d behind latest result)")
                        if key in thresholds.required:
                            stale_required = True
            if self.gates.staleness and stale_required:
                return self._refuse(
                    patient_id, "observations_stale", span, stale=stale,
                    detail="required inputs older than the patient's own monitoring cadence: "
                    + ", ".join(stale),
                )

            steps: dict[str, float] = {}
            clamped: list[str] = []
            implausible: list[str] = []
            inputs: list[dict[str, Any]] = []
            absent = [f.key for f in self.model.factors if f.key not in found]
            for factor in self.model.factors:
                entry = found.get(factor.key)
                if entry is None:
                    continue
                raw_value = float(entry["value"])
                if factor.implausible(raw_value):
                    implausible.append(f"{factor.key}={raw_value} {factor.unit}")
                value, was_clamped = factor.clamped(raw_value)
                steps[factor.key] = value
                if was_clamped:
                    clamped.append(factor.key)
                inputs.append(
                    {
                        "factor": factor.key,
                        "label": factor.label,
                        "value": entry["value"],
                        "unit": factor.unit,
                        "reference": factor.reference,
                        "steps": round(value, 4),
                        "observedAt": entry.get("at"),
                        "provenance": entry.get("source"),
                        "clamped": was_clamped,
                    }
                )

            if self.gates.plausibility and implausible:
                # Only physiologically impossible values refuse here. An extreme
                # but real value is clamped, flagged and still scored: refusing
                # the sickest patients would invert the purpose of the tool.
                # See tests/test_risk.py::test_extreme_but_real_value_is_scored.
                return self._refuse(
                    patient_id, "implausible_value", span, clamped=clamped, inputs=inputs,
                    detail="value outside the physiologically possible range (data error): "
                    + ", ".join(implausible),
                )

            probability = self.model.predict(steps)
            contributions, baseline, residual = self.model.attribute(steps)
            TRACER.metrics.observe("nullius_risk_probability", probability)
            TRACER.metrics.observe("nullius_risk_attribution_residual", residual)
            span.set(**{
                "risk.probability": probability,
                "risk.baseline": baseline,
                "risk.attribution_residual": residual,
            })

            # Less information -> a wider band in which we decline to take a side.
            widen = thresholds.widen_per_missing * len(absent)
            low = max(0.0, thresholds.review_low - widen)
            high = min(1.0, thresholds.review_high + widen)
            if self.gates.confidence and low <= probability <= high:
                result = self._refuse(
                    patient_id, "indeterminate_needs_review", span, inputs=inputs,
                    missing=absent, clamped=clamped,
                    detail=(
                        f"score {probability:.3f} falls in the indeterminate band "
                        f"[{low:.2f}, {high:.2f}]"
                        + (f", widened by {len(absent)} missing factor(s)" if absent else "")
                    ),
                )
                result.probability = probability
                result.baseline_probability = baseline
                return result

            band = "elevated" if probability > high else "low"
            TRACER.metrics.inc("nullius_risk_served_total")
            self._audit(patient_id, actor, actor_role, span.trace_id,
                        f"risk {probability:.3f} ({band})")
            return RiskResult(
                served=True,
                refusal_reason=None,
                patient_id=patient_id,
                probability=probability,
                band=band,
                baseline_probability=baseline,
                attribution_residual=residual,
                attributions=contributions,
                inputs=inputs,
                missing=absent,
                clamped=clamped,
                stale=stale,
                validated=False,
                calibrated=self.model.calibrated,
                trace_id=span.trace_id,
                detail="declared-coefficient score; not a validated risk model",
            )

    def _refuse(self, patient_id: str, reason: str, span, **kwargs: Any) -> RiskResult:
        TRACER.metrics.inc("nullius_risk_refusals_total", reason=reason)
        span.set(**{"risk.refusal_reason": reason})
        detail = kwargs.pop("detail", "")
        return RiskResult(
            served=False,
            refusal_reason=reason,
            patient_id=patient_id,
            trace_id=span.trace_id,
            detail=detail,
            **kwargs,
        )

    def _audit(self, patient_id: str, actor: str, actor_role: str, trace_id: str, detail: str) -> None:
        if self.store is None:
            return
        self.store.audit(
            actor=actor, actor_role=actor_role, action="risk.assess",
            patient_id=patient_id, trace_id=trace_id, detail=detail,
        )
