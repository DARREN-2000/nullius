"""Tests for the risk score, weighted towards the ways it could be dangerous.

The interesting tests here are not "does it produce a number". They are:

  * does it refuse when it should, and
  * does it REFUSE TO REFUSE when refusing would be the harmful answer.

The second one is a real bug this module shipped with. The first version
treated any value outside the coefficient range as implausible and abstained -
which meant the sickest patients in the cohort (eGFR in the teens) were the ones
the tool declined to score. A triage aid that goes quiet exactly when the
patient is deteriorating is worse than no triage aid, because its silence reads
as reassurance. `test_extreme_but_real_value_is_scored_not_refused` is the
regression test for that.
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nullius.app import AccessDenied, Principal  # noqa: E402
from nullius.risk import (  # noqa: E402
    FACTOR_KEYS,
    INTERCEPT,
    RISK_GATE_NAMES,
    RiskAssessor,
    RiskGates,
    RiskModel,
    RiskThresholds,
)

NOW = "2026-07-24T08:00:00Z"
OLD = "2024-01-05T08:00:00Z"

LOINC = {
    "egfr": "33914-3",
    "uacr": "9318-7",
    "potassium": "2823-3",
    "haemoglobin": "718-7",
}


class FakeStore:
    """A store stub, so gate tests can construct exactly the record they need.

    Using the real cohort here would mean testing the generator as much as the
    gates, and there is no synthetic patient with an impossible potassium.
    """

    def __init__(self, observations, birth_date="1957-03-11", medications=None):
        self._observations = observations
        self._birth_date = birth_date
        self._medications = medications or []
        self.audits = []

    def patient(self, patient_id):
        return {"id": patient_id, "birth_date": self._birth_date}

    def observations(self, patient_id, loinc=None):
        return [r for r in self._observations if loinc is None or r["loinc"] == loinc]

    def medications(self, patient_id):
        return self._medications

    def audit(self, **kwargs):
        self.audits.append(kwargs)


def series(loinc, values, at=NOW):
    """Three results so trend estimation has something to work with."""
    return [
        {"loinc": loinc, "value": v, "effective_at": ts}
        for v, ts in zip(values, ("2026-01-24T08:00:00Z", "2026-04-24T08:00:00Z", at))
    ]


def healthy_record(**overrides):
    rows = []
    rows += series(LOINC["egfr"], overrides.get("egfr", [88.0, 87.0, 86.0]))
    rows += series(LOINC["potassium"], overrides.get("potassium", [4.2, 4.3, 4.4]))
    rows += series(LOINC["haemoglobin"], overrides.get("haemoglobin", [13.5, 13.4, 13.6]))
    rows += series(LOINC["uacr"], overrides.get("uacr", [12.0, 11.0, 10.0]))
    return rows


class RiskGateTests(unittest.TestCase):
    def test_missing_required_input_refuses_rather_than_imputing(self) -> None:
        rows = [r for r in healthy_record() if r["loinc"] != LOINC["potassium"]]
        result = RiskAssessor(FakeStore(rows)).assess("p1")
        self.assertFalse(result.served)
        self.assertEqual(result.refusal_reason, "insufficient_observations")
        self.assertIn("potassium", result.missing)
        self.assertIsNone(result.probability)

    def test_stale_required_input_refuses(self) -> None:
        rows = healthy_record()
        rows = [r for r in rows if r["loinc"] != LOINC["potassium"]]
        rows.append({"loinc": LOINC["potassium"], "value": 4.4, "effective_at": OLD})
        result = RiskAssessor(FakeStore(rows)).assess("p1")
        self.assertFalse(result.served)
        self.assertEqual(result.refusal_reason, "observations_stale")

    def test_physiologically_impossible_value_refuses(self) -> None:
        """A potassium of 42 is a unit error or a bad feed, not a patient."""
        result = RiskAssessor(FakeStore(healthy_record(potassium=[4.2, 4.3, 42.0]))).assess("p1")
        self.assertFalse(result.served)
        self.assertEqual(result.refusal_reason, "implausible_value")
        self.assertIn("potassium", result.detail)

    def test_extreme_but_real_value_is_scored_not_refused(self) -> None:
        """Regression test. An eGFR of 8 is survivable, common in late CKD, and
        exactly the patient this score exists for. It must be clamped and
        flagged, never used as an excuse to stay silent.
        """
        rows = healthy_record(egfr=[14.0, 11.0, 8.0], potassium=[5.4, 5.6, 5.9],
                              haemoglobin=[10.5, 10.2, 9.8], uacr=[600.0, 700.0, 820.0])
        result = RiskAssessor(FakeStore(rows)).assess("p1")
        self.assertTrue(result.served, f"refused the sickest patient: {result.refusal_reason}")
        self.assertIn("egfr", result.clamped, "clamping must still be disclosed")
        self.assertGreater(result.probability, 0.5)

    def test_indeterminate_scores_abstain(self) -> None:
        thresholds = RiskThresholds(review_low=0.0, review_high=1.0, widen_per_missing=0.0)
        result = RiskAssessor(FakeStore(healthy_record()), thresholds=thresholds).assess("p1")
        self.assertFalse(result.served)
        self.assertEqual(result.refusal_reason, "indeterminate_needs_review")
        self.assertIsNotNone(result.probability, "an abstention still reports what it saw")

    def test_missing_optional_factors_widen_the_abstention_band(self) -> None:
        """Less information must mean more abstention, not the same confidence."""
        rows = [r for r in healthy_record() if r["loinc"] != LOINC["uacr"]]
        store = FakeStore(rows)
        narrow = RiskThresholds(review_low=0.0, review_high=0.0, widen_per_missing=0.0)
        wide = RiskThresholds(review_low=0.0, review_high=0.0, widen_per_missing=1.0)
        self.assertTrue(RiskAssessor(store, thresholds=narrow).assess("p1").served)
        widened = RiskAssessor(store, thresholds=wide).assess("p1")
        self.assertFalse(widened.served)
        self.assertIn("widened", widened.detail)

    def test_every_gate_is_load_bearing(self) -> None:
        """Ablation: disarm one gate and the input it blocks gets through."""
        cases = {
            "completeness": [r for r in healthy_record() if r["loinc"] != LOINC["potassium"]],
            "plausibility": healthy_record(potassium=[4.2, 4.3, 42.0]),
        }
        for gate, rows in cases.items():
            with self.subTest(gate=gate):
                store = FakeStore(rows)
                armed = RiskAssessor(store).assess("p1")
                self.assertFalse(armed.served, f"{gate} gate should have blocked this")
                disarmed = RiskAssessor(store, gates=RiskGates().without(gate)).assess("p1")
                self.assertNotEqual(disarmed.refusal_reason, armed.refusal_reason)

    def test_gate_names_are_complete_and_rejects_unknown(self) -> None:
        for name in RISK_GATE_NAMES:
            self.assertFalse(getattr(RiskGates().without(name), name))
        with self.assertRaises(ValueError):
            RiskGates().without("no_such_gate")


class RiskModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = RiskModel()

    def test_reference_patient_scores_the_intercept(self) -> None:
        """Every factor at its reference value must give exactly the baseline."""
        expected = 1.0 / (1.0 + math.exp(-INTERCEPT))
        self.assertAlmostEqual(self.model.predict({}), expected, places=12)

    def test_shapley_efficiency_holds(self) -> None:
        steps = {key: 0.4 * (i + 1) for i, key in enumerate(FACTOR_KEYS)}
        phi, baseline, full = self.model.shapley(steps)
        self.assertAlmostEqual(sum(phi), full - baseline, places=12)

    def test_attribution_reports_a_real_residual(self) -> None:
        steps = {key: -0.3 * (i + 1) for i, key in enumerate(FACTOR_KEYS)}
        contributions, _, residual = self.model.attribute(steps)
        self.assertLess(residual, 1e-9)
        self.assertEqual(len(contributions), len(FACTOR_KEYS))
        self.assertTrue(all(c["source"] for c in contributions), "every factor cites a reason")

    def test_harmful_direction_is_wired_the_right_way_round(self) -> None:
        """A low eGFR must raise risk and a high one must lower it. Getting a
        sign backwards is the classic silent failure in a hand-built score."""
        sick = self.model.predict({"egfr": 3.0})
        well = self.model.predict({"egfr": -2.0})
        self.assertGreater(sick, well)

    def test_calibration_is_off_and_declared_off_by_default(self) -> None:
        self.assertFalse(self.model.calibrated)
        steps = {"egfr": 1.0}
        self.assertAlmostEqual(self.model.predict(steps), self.model.predict_raw(steps), places=12)
        calibrated = RiskModel(calibration=(0.5, 0.3))
        self.assertTrue(calibrated.calibrated)
        self.assertNotAlmostEqual(calibrated.predict(steps), calibrated.predict_raw(steps), places=6)


class RiskAccessControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from scripts.run_pipeline import build_app

        cls.tmp = tempfile.TemporaryDirectory()
        cls.app = build_app(db_path=Path(cls.tmp.name) / "risk.db", corpus_dir=ROOT / "corpus")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.app.store.close()
        cls.tmp.cleanup()

    def test_nurse_cannot_read_risk_scores(self) -> None:
        with self.assertRaises(AccessDenied):
            self.app.risk_assessment(Principal(user_id="nurse.k", role="nurse"), "pat-001")

    def test_clinician_gets_a_scored_and_audited_assessment(self) -> None:
        before = len(self.app.store.audit_trail(limit=500))
        result = self.app.risk_assessment(
            Principal(user_id="dr.alvarez", role="clinician"), "pat-001"
        )
        self.assertTrue(result["served"])
        self.assertGreater(result["probability"], 0.5, "pat-001 has stage 3 CKD and is declining")
        after = self.app.store.audit_trail(limit=500)
        self.assertGreater(len(after), before, "risk reads must be audited")
        self.assertTrue(any(e["action"] == "risk.assess" for e in after))

    def test_the_result_never_claims_to_be_validated(self) -> None:
        result = self.app.risk_assessment(
            Principal(user_id="dr.alvarez", role="clinician"), "pat-001"
        )
        self.assertFalse(result["validated"])
        self.assertIn("not a validated risk model", result["disclaimer"])
        self.assertTrue(all(i["provenance"] for i in result["inputs"]))

    def test_unknown_patient_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.app.risk_assessment(Principal(user_id="dr.alvarez", role="clinician"), "nope")


if __name__ == "__main__":
    unittest.main()
