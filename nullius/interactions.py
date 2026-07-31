"""Drug safety checks against a curated rule table.

Honest scoping note, stated in the README as well: this is a small hand-curated
rule set derived from published interaction guidance, NOT a licensed interaction
database. It exists to prove the integration shape - rules in, structured
severity-ranked findings out, each with a mechanism and a source - and it never
asks an LLM to invent an interaction, because that is the exact failure mode that
makes clinical AI unsafe. Production would swap this table for RxNorm +
DrugBank/ONCHigh behind the same function signature.
"""

from __future__ import annotations

from typing import Any

from .observability import TRACER

SEVERITY_ORDER = {"contraindicated": 0, "major": 1, "moderate": 2, "minor": 3}

# frozenset of RxNorm ingredient codes -> rule
INTERACTION_RULES: list[dict[str, Any]] = [
    {
        "drugs": {"29046", "9997"},
        "names": ["Lisinopril", "Spironolactone"],
        "severity": "major",
        "mechanism": "ACE inhibitor plus potassium-sparing diuretic: additive potassium retention via "
                     "reduced aldosterone activity, compounded by reduced renal excretion in CKD.",
        "effect": "Risk of severe hyperkalaemia and arrhythmia.",
        "action": "Recheck potassium and renal function; consider dose reduction or withdrawal of one agent.",
        "source": "KDIGO CKD medication safety guidance",
    },
    {
        "drugs": {"29046", "5640"},
        "names": ["Lisinopril", "Ibuprofen"],
        "severity": "major",
        "mechanism": "NSAID inhibits prostaglandin-mediated afferent arteriolar vasodilation while the ACE "
                     "inhibitor blocks efferent constriction, collapsing glomerular filtration pressure.",
        "effect": "Acute kidney injury and reduced antihypertensive effect.",
        "action": "Avoid routine NSAID use in CKD; prefer paracetamol or topical analgesia.",
        "source": "NICE CKD guidance on nephrotoxic medicines",
    },
    {
        "drugs": {"9997", "5640"},
        "names": ["Spironolactone", "Ibuprofen"],
        "severity": "moderate",
        "mechanism": "NSAIDs reduce renal potassium excretion, adding to potassium-sparing diuretic effect.",
        "effect": "Further elevation of serum potassium.",
        "action": "Monitor potassium closely if combination cannot be avoided.",
        "source": "Local formulary interaction table",
    },
    {
        "drugs": {"11289", "5640"},
        "names": ["Warfarin", "Ibuprofen"],
        "severity": "major",
        "mechanism": "Antiplatelet effect plus gastric mucosal injury on top of anticoagulation.",
        "effect": "Substantially increased gastrointestinal bleeding risk.",
        "action": "Avoid combination; if unavoidable add gastroprotection and monitor INR.",
        "source": "Local formulary interaction table",
    },
    {
        "drugs": {"6809", "4603"},
        "names": ["Metformin", "Furosemide"],
        "severity": "moderate",
        "mechanism": "Loop diuretic volume depletion can reduce renal clearance of metformin.",
        "effect": "Increased metformin exposure and lactic acidosis risk if renal function falls.",
        "action": "Monitor renal function; hold metformin during intercurrent illness.",
        "source": "Local formulary interaction table",
    },
]

# RxNorm ingredient -> (condition SNOMED, rule) for drug-disease safety.
DRUG_CONDITION_RULES: list[dict[str, Any]] = [
    {
        "drug": "6809",
        "drug_name": "Metformin",
        "condition": "433144002",
        "condition_name": "Chronic kidney disease stage 3",
        "severity": "moderate",
        "mechanism": "Metformin is renally cleared; accumulation raises lactic acidosis risk as eGFR falls.",
        "effect": "Dose review required below eGFR 45; avoid below 30 mL/min/1.73m2.",
        "action": "Check current eGFR and adjust dose to renal function.",
        "source": "NICE type 2 diabetes guidance",
    },
    {
        "drug": "5640",
        "drug_name": "Ibuprofen",
        "condition": "433144002",
        "condition_name": "Chronic kidney disease stage 3",
        "severity": "major",
        "mechanism": "NSAIDs reduce renal perfusion in patients already dependent on prostaglandin support.",
        "effect": "Accelerated loss of renal function.",
        "action": "Deprescribe where possible and document analgesic alternative.",
        "source": "NICE CKD guidance on nephrotoxic medicines",
    },
]


def check_interactions(medications: list[dict[str, Any]], conditions: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    with TRACER.span("drug_safety.check") as span:
        active = {m["rxnorm"]: m for m in medications if m.get("rxnorm") and m.get("status") == "active"}
        findings: list[dict[str, Any]] = []

        for rule in INTERACTION_RULES:
            if rule["drugs"] <= set(active):
                findings.append(
                    {
                        "type": "drug-drug",
                        "drugs": rule["names"],
                        "severity": rule["severity"],
                        "mechanism": rule["mechanism"],
                        "effect": rule["effect"],
                        "action": rule["action"],
                        "source": rule["source"],
                        "summary": f"{' + '.join(rule['names'])} ({rule['severity']}): {rule['effect']}",
                    }
                )

        condition_codes = {c["code"] for c in (conditions or [])}
        for rule in DRUG_CONDITION_RULES:
            if rule["drug"] in active and rule["condition"] in condition_codes:
                findings.append(
                    {
                        "type": "drug-disease",
                        "drugs": [rule["drug_name"]],
                        "condition": rule["condition_name"],
                        "severity": rule["severity"],
                        "mechanism": rule["mechanism"],
                        "effect": rule["effect"],
                        "action": rule["action"],
                        "source": rule["source"],
                        "summary": f"{rule['drug_name']} with {rule['condition_name']} ({rule['severity']}): {rule['effect']}",
                    }
                )

        findings.sort(key=lambda f: SEVERITY_ORDER.get(f["severity"], 9))
        span.set(**{"drug_safety.findings": len(findings), "drug_safety.active_medications": len(active)})
        TRACER.metrics.inc("nullius_interaction_findings_total", len(findings))
        return findings
