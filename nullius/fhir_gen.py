"""Deterministic synthetic FHIR R4 bundle generator (Synthea-shaped).

Why synthetic: Nullius must never touch real PHI to be demonstrable. Bundles
here use real FHIR resource types, LOINC lab codes, SNOMED condition codes and
RxNorm medication codes so the ingest layer is exercised against a realistic
shape. Clinical stories are hand-authored so the lab-intelligence and copilot
layers have known-correct expected outputs for evaluation.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from typing import Any

LOINC = {
    "creatinine": ("2160-0", "Creatinine [Mass/volume] in Serum or Plasma", "mg/dL"),
    "egfr": ("33914-3", "Glomerular filtration rate/1.73 sq M.predicted", "mL/min/1.73m2"),
    "potassium": ("2823-3", "Potassium [Moles/volume] in Serum or Plasma", "mmol/L"),
    "sodium": ("2951-2", "Sodium [Moles/volume] in Serum or Plasma", "mmol/L"),
    "hemoglobin": ("718-7", "Hemoglobin [Mass/volume] in Blood", "g/dL"),
    "hba1c": ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", "%"),
    "albumin_creat_ratio": ("9318-7", "Albumin/Creatinine [Mass Ratio] in Urine", "mg/g"),
    "wbc": ("6690-2", "Leukocytes [#/volume] in Blood", "10*3/uL"),
    "lactate": ("2524-7", "Lactate [Moles/volume] in Serum or Plasma", "mmol/L"),
    "crp": ("1988-5", "C reactive protein [Mass/volume] in Serum or Plasma", "mg/L"),
    "platelets": ("777-3", "Platelets [#/volume] in Blood", "10*3/uL"),
    "tsh": ("3016-3", "Thyrotropin [Units/volume] in Serum or Plasma", "m[IU]/L"),
}

SNOMED = {
    "ckd3": ("433144002", "Chronic kidney disease stage 3"),
    "t2dm": ("44054006", "Type 2 diabetes mellitus"),
    "htn": ("38341003", "Hypertensive disorder"),
    "afib": ("49436004", "Atrial fibrillation"),
    "hf": ("84114007", "Heart failure"),
    "copd": ("13645005", "Chronic obstructive pulmonary disease"),
    "uti": ("68566005", "Urinary tract infectious disease"),
}

RXNORM = {
    "lisinopril": ("29046", "Lisinopril 10 MG Oral Tablet"),
    "spironolactone": ("9997", "Spironolactone 25 MG Oral Tablet"),
    "metformin": ("6809", "Metformin 1000 MG Oral Tablet"),
    "empagliflozin": ("1545653", "Empagliflozin 10 MG Oral Tablet"),
    "warfarin": ("11289", "Warfarin Sodium 5 MG Oral Tablet"),
    "ibuprofen": ("5640", "Ibuprofen 400 MG Oral Tablet"),
    "furosemide": ("4603", "Furosemide 40 MG Oral Tablet"),
    "amoxicillin": ("723", "Amoxicillin 500 MG Oral Capsule"),
}


def _obs(patient_id: str, key: str, value: float, when: date, encounter: str) -> dict[str, Any]:
    code, display, unit = LOINC[key]
    return {
        "resourceType": "Observation",
        "id": f"obs-{patient_id}-{key}-{when.isoformat()}",
        "status": "final",
        "category": [
            {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory"}]}
        ],
        "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": display}], "text": display},
        "subject": {"reference": f"Patient/{patient_id}"},
        "encounter": {"reference": f"Encounter/{encounter}"},
        "effectiveDateTime": f"{when.isoformat()}T08:00:00Z",
        "valueQuantity": {"value": round(value, 2), "unit": unit, "system": "http://unitsofmeasure.org"},
    }


def _condition(patient_id: str, key: str, onset: date) -> dict[str, Any]:
    code, display = SNOMED[key]
    return {
        "resourceType": "Condition",
        "id": f"cond-{patient_id}-{key}",
        "clinicalStatus": {"coding": [{"code": "active"}]},
        "verificationStatus": {"coding": [{"code": "confirmed"}]},
        "code": {"coding": [{"system": "http://snomed.info/sct", "code": code, "display": display}], "text": display},
        "subject": {"reference": f"Patient/{patient_id}"},
        "onsetDateTime": f"{onset.isoformat()}T00:00:00Z",
    }


def _medication(patient_id: str, key: str, started: date, dose: str) -> dict[str, Any]:
    code, display = RXNORM[key]
    return {
        "resourceType": "MedicationRequest",
        "id": f"med-{patient_id}-{key}",
        "status": "active",
        "intent": "order",
        "medicationCodeableConcept": {
            "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": code, "display": display}],
            "text": display,
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "authoredOn": f"{started.isoformat()}T00:00:00Z",
        "dosageInstruction": [{"text": dose}],
    }


def _encounter(patient_id: str, idx: int, when: date, kind: str) -> dict[str, Any]:
    return {
        "resourceType": "Encounter",
        "id": f"enc-{patient_id}-{idx}",
        "status": "finished",
        "class": {"code": kind},
        "subject": {"reference": f"Patient/{patient_id}"},
        "period": {"start": f"{when.isoformat()}T08:00:00Z", "end": f"{when.isoformat()}T10:30:00Z"},
        "reasonCode": [{"text": "Routine follow-up" if kind == "AMB" else "Acute presentation"}],
    }


def _patient(patient_id: str, family: str, given: str, birth: str, gender: str) -> dict[str, Any]:
    return {
        "resourceType": "Patient",
        "id": patient_id,
        "identifier": [{"system": "urn:nullius:mrn", "value": patient_id.upper()}],
        "name": [{"family": family, "given": [given]}],
        "gender": gender,
        "birthDate": birth,
    }


def _bundle(resources: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"fullUrl": f"urn:uuid:{r['id']}", "resource": r} for r in resources],
    }


def index_patient(today: date) -> dict[str, Any]:
    """The demo patient: CKD stage 3 progressing, new hyperkalaemia, poor glycaemic control.

    Deliberately constructed so that three independent signals collide:
      * eGFR falling across four quarters (progression)
      * potassium 5.9 mmol/L (critical, actionable now)
      * lisinopril + spironolactone co-prescription (mechanistic explanation)
    A good copilot answer must connect all three and cite guidance for each.
    """
    pid = "pat-001"
    res: list[dict[str, Any]] = [_patient(pid, "Alvarez", "Marta", "1957-03-11", "female")]
    res.append(_condition(pid, "ckd3", today - timedelta(days=900)))
    res.append(_condition(pid, "t2dm", today - timedelta(days=2600)))
    res.append(_condition(pid, "htn", today - timedelta(days=3000)))
    res.append(_medication(pid, "lisinopril", today - timedelta(days=800), "10 mg once daily"))
    res.append(_medication(pid, "spironolactone", today - timedelta(days=95), "25 mg once daily"))
    res.append(_medication(pid, "metformin", today - timedelta(days=2400), "1000 mg twice daily"))
    res.append(_medication(pid, "ibuprofen", today - timedelta(days=40), "400 mg as needed"))

    series = {
        "egfr": [52.0, 47.0, 41.0, 34.0],
        "creatinine": [1.15, 1.28, 1.44, 1.72],
        "potassium": [4.4, 4.8, 5.3, 5.9],
        "sodium": [139, 138, 137, 136],
        "hemoglobin": [12.6, 12.1, 11.4, 10.6],
        "hba1c": [7.4, 7.9, 8.3, 8.8],
        "platelets": [240, 238, 232, 229],
    }
    for idx, offset in enumerate([270, 180, 90, 7]):
        when = today - timedelta(days=offset)
        enc_id = f"enc-{pid}-{idx}"
        res.append(_encounter(pid, idx, when, "AMB" if offset > 30 else "EMER"))
        for key, values in series.items():
            res.append(_obs(pid, key, values[idx], when, enc_id))
    return _bundle(res)


def cohort(n: int = 11, seed: int = 20260729, today: date | None = None) -> list[dict[str, Any]]:
    """Background cohort so aggregate views and pagination are non-trivial."""
    today = today or date.today()
    rng = random.Random(seed)
    families = [
        ("Novak", "Petra", "female"), ("Okafor", "Emeka", "male"),
        ("Lindqvist", "Sten", "male"), ("Haddad", "Nour", "female"),
        ("Fischer", "Jonas", "male"), ("Ferrari", "Chiara", "female"),
        ("Mbeki", "Thandi", "female"), ("Kowalski", "Marek", "male"),
        ("Yilmaz", "Elif", "female"), ("Dubois", "Camille", "female"),
        ("Nakamura", "Hana", "female"),
    ]
    bundles = []
    for i in range(min(n, len(families))):
        pid = f"pat-{i + 2:03d}"
        family, given, gender = families[i]
        birth_year = rng.randint(1940, 1985)
        res: list[dict[str, Any]] = [
            _patient(pid, family, given, f"{birth_year}-0{rng.randint(1, 9)}-1{rng.randint(0, 8)}", gender)
        ]
        for cond in rng.sample(list(SNOMED), rng.randint(1, 3)):
            res.append(_condition(pid, cond, today - timedelta(days=rng.randint(200, 3000))))
        for med in rng.sample(list(RXNORM), rng.randint(1, 3)):
            res.append(_medication(pid, med, today - timedelta(days=rng.randint(20, 900)), "per label"))
        for idx, offset in enumerate(sorted([rng.randint(5, 300) for _ in range(3)], reverse=True)):
            when = today - timedelta(days=offset)
            res.append(_encounter(pid, idx, when, "AMB"))
            for key in rng.sample(list(LOINC), 5):
                base = {
                    "creatinine": 1.0, "egfr": 75, "potassium": 4.2, "sodium": 139,
                    "hemoglobin": 13.0, "hba1c": 6.1, "albumin_creat_ratio": 20,
                    "wbc": 7.0, "lactate": 1.2, "crp": 4.0, "platelets": 250, "tsh": 2.0,
                }[key]
                res.append(_obs(pid, key, base * rng.uniform(0.75, 1.35), when, f"enc-{pid}-{idx}"))
        bundles.append(_bundle(res))
    return bundles


def generate_all(today: date | None = None) -> list[dict[str, Any]]:
    today = today or date.today()
    return [index_patient(today)] + cohort(today=today)
