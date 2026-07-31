"""Lab intelligence: deterministic, explainable, no LLM in the loop.

Design rule that shapes this whole file: anything that can be computed from a
reference range or a regression slope MUST NOT be delegated to a language model.
Abnormal flags, critical values and trend direction are patient-safety-critical
and must be reproducible, unit-tested and auditable. The LLM's job is to explain
these findings, never to derive them.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Literal

from .observability import TRACER

Severity = Literal["normal", "low", "high", "critical_low", "critical_high"]

# (low, high, critical_low, critical_high, unit, friendly name, higher_is_better)
REFERENCE_RANGES: dict[str, dict[str, Any]] = {
    "2160-0": {"name": "Creatinine", "low": 0.6, "high": 1.10, "crit_high": 2.5, "unit": "mg/dL", "direction": "lower_better"},
    "33914-3": {"name": "eGFR", "low": 60.0, "high": 130.0, "crit_low": 15.0, "unit": "mL/min/1.73m2", "direction": "higher_better"},
    "2823-3": {"name": "Potassium", "low": 3.5, "high": 5.0, "crit_low": 2.8, "crit_high": 5.8, "unit": "mmol/L", "direction": "in_range"},
    "2951-2": {"name": "Sodium", "low": 135.0, "high": 145.0, "crit_low": 125.0, "crit_high": 155.0, "unit": "mmol/L", "direction": "in_range"},
    "718-7": {"name": "Haemoglobin", "low": 12.0, "high": 16.0, "crit_low": 7.0, "unit": "g/dL", "direction": "higher_better"},
    "4548-4": {"name": "HbA1c", "low": 4.0, "high": 7.0, "crit_high": 10.0, "unit": "%", "direction": "lower_better"},
    "9318-7": {"name": "Urine albumin/creatinine ratio", "low": 0.0, "high": 30.0, "crit_high": 300.0, "unit": "mg/g", "direction": "lower_better"},
    "6690-2": {"name": "White cell count", "low": 4.0, "high": 11.0, "crit_low": 1.0, "crit_high": 25.0, "unit": "10*3/uL", "direction": "in_range"},
    "2524-7": {"name": "Lactate", "low": 0.5, "high": 2.0, "crit_high": 4.0, "unit": "mmol/L", "direction": "lower_better"},
    "1988-5": {"name": "C-reactive protein", "low": 0.0, "high": 5.0, "crit_high": 100.0, "unit": "mg/L", "direction": "lower_better"},
    "777-3": {"name": "Platelets", "low": 150.0, "high": 400.0, "crit_low": 50.0, "unit": "10*3/uL", "direction": "in_range"},
    "3016-3": {"name": "TSH", "low": 0.4, "high": 4.0, "crit_high": 20.0, "unit": "m[IU]/L", "direction": "in_range"},
}

# Condition (SNOMED) -> monitoring panel the guideline expects to see.
EXPECTED_PANELS: dict[str, list[str]] = {
    "433144002": ["33914-3", "2160-0", "2823-3", "718-7", "9318-7"],  # CKD stage 3
    "44054006": ["4548-4", "9318-7", "33914-3"],                        # Type 2 diabetes
    "84114007": ["2823-3", "2160-0", "2951-2"],                         # Heart failure
    "49436004": ["777-3", "3016-3"],                                     # Atrial fibrillation
}


@dataclass
class LabPoint:
    at: str
    value: float


@dataclass
class LabFinding:
    loinc: str
    name: str
    unit: str
    latest_value: float
    latest_at: str
    severity: Severity
    reference: str
    delta_from_previous: float | None
    percent_change: float | None
    trend: Literal["rising", "falling", "stable", "insufficient_data"]
    trend_slope_per_30d: float | None
    clinically_adverse_trend: bool
    series: list[LabPoint]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["series"] = [asdict(p) for p in self.series]
        return data


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def classify(loinc: str, value: float) -> tuple[Severity, str]:
    ref = REFERENCE_RANGES.get(loinc)
    if not ref:
        return "normal", "no reference range configured"
    label = f"{ref['low']}\u2013{ref['high']} {ref['unit']}"
    if "crit_high" in ref and value >= ref["crit_high"]:
        return "critical_high", label
    if "crit_low" in ref and value <= ref["crit_low"]:
        return "critical_low", label
    if value > ref["high"]:
        return "high", label
    if value < ref["low"]:
        return "low", label
    return "normal", label


def _slope_per_30d(series: list[LabPoint]) -> float | None:
    """Ordinary least squares slope in units per 30 days.

    Least squares rather than last-minus-first so a single noisy draw cannot
    manufacture a trend, which is the most common false-positive in lab alerting.
    """
    if len(series) < 3:
        return None
    t0 = _parse(series[0].at)
    xs = [(_parse(p.at) - t0).total_seconds() / 86400.0 for p in series]
    ys = [p.value for p in series]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return None
    slope_per_day = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    return slope_per_day * 30.0


def analyse_series(loinc: str, rows: list[dict[str, Any]]) -> LabFinding | None:
    rows = [r for r in rows if r.get("value") is not None]
    if not rows:
        return None
    ref = REFERENCE_RANGES.get(loinc, {})
    series = [LabPoint(at=r["effective_at"], value=float(r["value"])) for r in rows]
    series.sort(key=lambda p: p.at)
    latest = series[-1]
    severity, reference = classify(loinc, latest.value)
    delta = round(latest.value - series[-2].value, 3) if len(series) > 1 else None
    pct = (
        round((latest.value - series[-2].value) / series[-2].value * 100, 1)
        if len(series) > 1 and series[-2].value
        else None
    )
    slope = _slope_per_30d(series)
    spread = max(p.value for p in series) - min(p.value for p in series)
    noise_floor = 0.05 * (abs(latest.value) or 1.0)
    if slope is None:
        trend = "insufficient_data"
    elif spread < noise_floor:
        trend = "stable"
    elif slope > 0:
        trend = "rising"
    elif slope < 0:
        trend = "falling"
    else:
        trend = "stable"

    direction = ref.get("direction", "in_range")
    adverse = bool(
        (trend == "rising" and direction == "lower_better")
        or (trend == "falling" and direction == "higher_better")
        or (trend in {"rising", "falling"} and direction == "in_range" and severity != "normal")
    )

    bits = [f"latest {latest.value} {ref.get('unit', '')} against reference {reference}"]
    if delta is not None:
        bits.append(f"changed by {delta:+} versus previous draw")
    if slope is not None:
        bits.append(f"least-squares slope {slope:+.2f} per 30 days over {len(series)} results")
    if adverse:
        bits.append("direction of change is clinically unfavourable")

    return LabFinding(
        loinc=loinc,
        name=ref.get("name", rows[-1].get("display", loinc)),
        unit=ref.get("unit", rows[-1].get("unit", "")),
        latest_value=latest.value,
        latest_at=latest.at,
        severity=severity,
        reference=reference,
        delta_from_previous=delta,
        percent_change=pct,
        trend=trend,
        trend_slope_per_30d=round(slope, 3) if slope is not None else None,
        clinically_adverse_trend=adverse,
        series=series,
        rationale="; ".join(bits),
    )


SEVERITY_RANK = {"critical_high": 0, "critical_low": 1, "high": 2, "low": 3, "normal": 4}


def review_patient(store, patient_id: str) -> dict[str, Any]:
    """Full lab review: findings, criticals, adverse trends and monitoring gaps."""
    with TRACER.span("labs.review", **{"patient.id": patient_id}) as span:
        rows = store.observations(patient_id)
        by_code: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_code.setdefault(row["loinc"], []).append(row)

        findings = [f for f in (analyse_series(code, rs) for code, rs in by_code.items()) if f]
        findings.sort(key=lambda f: (SEVERITY_RANK[f.severity], not f.clinically_adverse_trend, f.name))

        conditions = store.conditions(patient_id)
        expected: set[str] = set()
        for cond in conditions:
            expected.update(EXPECTED_PANELS.get(cond["code"] or "", []))
        gaps = [
            {
                "loinc": code,
                "name": REFERENCE_RANGES.get(code, {}).get("name", code),
                "reason": "expected by monitoring panel for an active condition but never resulted",
            }
            for code in sorted(expected - set(by_code))
        ]

        criticals = [f for f in findings if f.severity.startswith("critical")]
        adverse = [f for f in findings if f.clinically_adverse_trend and not f.severity.startswith("critical")]

        TRACER.metrics.inc("nullius_lab_findings_total", len(findings))
        TRACER.metrics.inc("nullius_lab_criticals_total", len(criticals))
        span.set(**{
            "labs.findings": len(findings),
            "labs.criticals": len(criticals),
            "labs.adverse_trends": len(adverse),
            "labs.monitoring_gaps": len(gaps),
        })

        return {
            "patient_id": patient_id,
            "findings": [f.to_dict() for f in findings],
            "critical_values": [f.to_dict() for f in criticals],
            "adverse_trends": [f.to_dict() for f in adverse],
            "monitoring_gaps": gaps,
        }
