"""Patient timeline: one chronological, deduplicated event stream.

This is the answer to "the clinician opens a 100-page record". Encounters,
diagnoses, prescriptions and abnormal results are merged into a single ordered
stream where each event carries its own severity, so the UI can render a
reverse-chronological review without any further clinical logic. Normal results
are deliberately collapsed into a per-encounter count: showing 60 normal values
is how you hide the one abnormal one.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .labs import REFERENCE_RANGES, classify
from .observability import TRACER


def _iso(ts: str | None) -> str:
    return (ts or "")[:10]


def build_timeline(store, patient_id: str) -> list[dict[str, Any]]:
    with TRACER.span("timeline.build", **{"patient.id": patient_id}) as span:
        events: list[dict[str, Any]] = []

        for cond in store.conditions(patient_id):
            events.append(
                {
                    "at": _iso(cond["onset_at"]),
                    "kind": "diagnosis",
                    "severity": "info",
                    "title": cond["display"],
                    "detail": f"SNOMED {cond['code']} \u00b7 {cond['clinical_status']}",
                }
            )

        for med in store.medications(patient_id):
            events.append(
                {
                    "at": _iso(med["authored_on"]),
                    "kind": "medication",
                    "severity": "info",
                    "title": med["display"],
                    "detail": f"{med['dose']} \u00b7 RxNorm {med['rxnorm']} \u00b7 {med['status']}",
                }
            )

        normal_counts: dict[str, int] = {}
        for obs in store.observations(patient_id):
            if obs["value"] is None:
                continue
            severity, reference = classify(obs["loinc"], float(obs["value"]))
            day = _iso(obs["effective_at"])
            if severity == "normal":
                normal_counts[day] = normal_counts.get(day, 0) + 1
                continue
            name = REFERENCE_RANGES.get(obs["loinc"], {}).get("name", obs["display"])
            events.append(
                {
                    "at": day,
                    "kind": "lab",
                    "severity": "critical" if severity.startswith("critical") else "abnormal",
                    "title": f"{name} {obs['value']} {obs['unit']}",
                    "detail": f"{severity.replace('_', ' ')} \u00b7 reference {reference}",
                }
            )

        for enc in store.encounters(patient_id):
            day = _iso(enc["started_at"])
            kind = {"AMB": "Outpatient visit", "EMER": "Emergency presentation", "IMP": "Inpatient admission"}.get(
                enc["class"], "Encounter"
            )
            normals = normal_counts.get(day, 0)
            detail = enc["reason"] or ""
            if normals:
                detail = f"{detail} \u00b7 {normals} result(s) within reference range".strip(" \u00b7")
            events.append(
                {
                    "at": day,
                    "kind": "encounter",
                    "severity": "attention" if enc["class"] == "EMER" else "info",
                    "title": kind,
                    "detail": detail,
                }
            )

        order = {"encounter": 0, "lab": 1, "diagnosis": 2, "medication": 3}
        events.sort(key=lambda e: (e["at"], order.get(e["kind"], 9)), reverse=True)
        span.set(**{"timeline.events": len(events)})
        return events


def summarise_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        counts[event["kind"]] = counts.get(event["kind"], 0) + 1
    return counts
