"""Tests for the claims that are easiest to overstate.

Three of them:

  * "Shapley" is the most abused word in applied ML. Almost everything labelled
    SHAP is a sampled or kernel-regression approximation, and the difference is
    rarely stated. With nine features the exact values are computable, so these
    tests assert the defining axioms directly - efficiency, symmetry, dummy -
    rather than checking that a plot renders.
  * "Calibrated" is the second most abused. A calibrator must be fitted on the
    training split, must be monotone (it may not reorder patients), and must
    actually reduce calibration error on data it never saw.
  * A benchmark that cannot be failed proves nothing. The generator is tested
    for OVERLAP: if the two classes were trivially separable, every safety
    metric downstream would be measuring the generator instead of the system.
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nullius import dicom, imaging, onnx  # noqa: E402
from nullius.lesions import (  # noqa: E402
    LABEL_NOISE,
    SEVERITY_MEAN_BENIGN,
    SEVERITY_MEAN_SUSPICIOUS,
    generate_cohort,
)
from nullius.vision import (  # noqa: E402
    ModelBundle,
    apply_platt,
    auroc,
    expected_calibration_error,
    fit_platt,
    sensitivity_at_specificity,
    specificity_at_sensitivity,
)

N_FEATURES = len(imaging.FEATURE_NAMES)


def build_bundle(
    directory: Path,
    w1: list[list[float]],
    b1: list[float],
    w2: list[list[float]],
    b2: list[float],
    mean: list[float],
) -> ModelBundle:
    """A ModelBundle over weights the test controls exactly."""
    model_path = directory / "model.onnx"
    onnx.export_mlp(model_path, w1, b1, w2, b2)
    model_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "feature_names": list(imaging.FEATURE_NAMES),
                "feature_mean": mean,
                "feature_std": [0.1] * N_FEATURES,
                "operating_point": 0.5,
            }
        ),
        encoding="utf-8",
    )
    return ModelBundle(model_path)


class ShapleyAxiomTests(unittest.TestCase):
    """The axioms that make a number a Shapley value rather than a bar chart."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        hidden = 5
        # Deterministic, deliberately asymmetric weights with interactions.
        self.w1 = [
            [((i * 7 + j * 3) % 11 - 5) / 4.0 for j in range(hidden)] for i in range(N_FEATURES)
        ]
        self.b1 = [0.1 * j for j in range(hidden)]
        self.w2 = [[((j * 5) % 7 - 3) / 2.0] for j in range(hidden)]
        self.b2 = [0.2]
        self.mean = [0.5] * N_FEATURES
        self.bundle = build_bundle(self.dir, self.w1, self.b1, self.w2, self.b2, self.mean)
        self.x = [0.1 * ((i * 3) % 9) for i in range(N_FEATURES)]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_efficiency_axiom_holds_exactly(self) -> None:
        """Sum of attributions == model output - baseline output.

        This is the axiom sampled SHAP cannot guarantee, and the reason the
        exhaustive enumeration is worth its cost at nine features.
        """
        phi, baseline, full = self.bundle.shapley(self.x)
        self.assertAlmostEqual(sum(phi), full - baseline, places=9)

    def test_reported_residual_is_actually_zero(self) -> None:
        contributions, baseline, residual = self.bundle.attribute(
            self.x, self.bundle.predict(self.x)
        )
        self.assertLess(residual, 1e-9, "attribute() must report a real residual, not a hopeful one")
        self.assertEqual(len(contributions), N_FEATURES, "every feature must be accounted for")
        self.assertTrue(all(c["method"] == "exact-shapley" for c in contributions))
        self.assertAlmostEqual(baseline, self.bundle.predict(self.mean), places=9)

    def test_dummy_axiom_a_feature_the_model_ignores_gets_zero(self) -> None:
        """A feature wired to zero weights cannot influence the output, so its
        attribution must be exactly zero - not merely small."""
        dead = 3
        w1 = [row[:] for row in self.w1]
        w1[dead] = [0.0] * len(w1[dead])
        sub = self.dir / "dummy"
        sub.mkdir()
        bundle = build_bundle(sub, w1, self.b1, self.w2, self.b2, self.mean)
        phi, _, _ = bundle.shapley(self.x)
        self.assertAlmostEqual(phi[dead], 0.0, places=12)

    def test_symmetry_axiom_interchangeable_features_get_equal_credit(self) -> None:
        """Two features with identical weights and identical values must receive
        identical attributions, whatever the rest of the network does."""
        w1 = [row[:] for row in self.w1]
        w1[1] = w1[0][:]
        sub = self.dir / "symmetry"
        sub.mkdir()
        bundle = build_bundle(sub, w1, self.b1, self.w2, self.b2, self.mean)
        x = list(self.x)
        x[1] = x[0]
        phi, _, _ = bundle.shapley(x)
        self.assertAlmostEqual(phi[0], phi[1], places=12)

    def test_the_baseline_explains_itself_as_nothing(self) -> None:
        """Attributing the baseline point must give all zeros: there is no
        deviation from the reference to apportion."""
        phi, baseline, full = self.bundle.shapley(self.mean)
        self.assertAlmostEqual(full, baseline, places=12)
        for index, value in enumerate(phi):
            self.assertAlmostEqual(value, 0.0, places=12, msg=imaging.FEATURE_NAMES[index])

    def test_every_coalition_is_evaluated_exactly_once(self) -> None:
        """512 coalitions, not 512 * 9 model calls: the cache is load-bearing."""
        calls = {"n": 0}
        original = self.bundle.predict

        def counting(features):
            calls["n"] += 1
            return original(features)

        self.bundle.predict = counting  # type: ignore[method-assign]
        self.bundle.shapley(self.x)
        self.assertEqual(calls["n"], 2**N_FEATURES)


class CalibrationTests(unittest.TestCase):
    def test_platt_is_identity_when_one_class_is_missing(self) -> None:
        """Refusing to fit is the right answer: there is nothing to learn from a
        single class, and a silently wrong calibrator is worse than none."""
        self.assertEqual(fit_platt([0.2, 0.4, 0.6], [1, 1, 1]), (1.0, 0.0))
        self.assertEqual(fit_platt([0.2, 0.4, 0.6], [0, 0, 0]), (1.0, 0.0))

    def test_calibration_is_monotone_and_never_reorders_patients(self) -> None:
        """Calibration may change what a score means; it may not change who is
        ranked above whom, or the operating point would be meaningless."""
        a, b = 0.6, 0.65
        scores = [i / 40.0 for i in range(1, 40)]
        mapped = [apply_platt(s, a, b) for s in scores]
        self.assertEqual(mapped, sorted(mapped))
        self.assertAlmostEqual(auroc(scores, [i % 2 for i in range(39)]),
                               auroc(mapped, [i % 2 for i in range(39)]), places=9)

    def test_platt_reduces_calibration_error_on_overconfident_scores(self) -> None:
        """The point of the exercise: a model that says 0.98 and is right 70% of
        the time should be corrected towards 0.7."""
        scores, labels = [], []
        for i in range(100):
            positive = i % 10 < 7
            scores.append(0.98 if positive else 0.02)
            labels.append(1 if positive else 0)
        # Flip 30% of the confident positives: the score stays 0.98, truth does not.
        for i in range(0, 70, 3):
            labels[i] = 0
        before = expected_calibration_error(scores, labels)
        a, b = fit_platt(scores, labels)
        after = expected_calibration_error([apply_platt(s, a, b) for s in scores], labels)
        self.assertLess(after, before)

    def test_calibrated_bundle_differs_from_raw_output(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        directory = Path(tmp.name)
        try:
            w1 = [[0.4] * 3 for _ in range(N_FEATURES)]
            bundle = build_bundle(directory, w1, [0.0] * 3, [[0.9]] * 3, [0.0], [0.5] * N_FEATURES)
            self.assertFalse(bundle.calibrated, "absent metadata must mean no calibration")
            x = [0.7] * N_FEATURES
            self.assertAlmostEqual(bundle.predict(x), bundle.predict_raw(x), places=12)

            meta = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))
            meta["calibration"] = {"method": "platt", "a": 0.5, "b": 0.4}
            bundle.metadata_path.write_text(json.dumps(meta), encoding="utf-8")
            recalibrated = ModelBundle(bundle.model_path)
            self.assertTrue(recalibrated.calibrated)
            self.assertNotAlmostEqual(
                recalibrated.predict(x), recalibrated.predict_raw(x), places=6
            )
        finally:
            tmp.cleanup()


class OperatingPointTests(unittest.TestCase):
    """A triage tool must be tuned to the error that actually hurts."""

    scores = [0.10, 0.20, 0.30, 0.45, 0.50, 0.55, 0.70, 0.80, 0.90, 0.95]
    labels = [0, 0, 0, 1, 0, 1, 0, 1, 1, 1]

    def test_specificity_at_sensitivity_respects_its_constraint(self) -> None:
        specificity, threshold = specificity_at_sensitivity(self.scores, self.labels, 0.90)
        achieved = sum(
            1 for s, l in zip(self.scores, self.labels) if l == 1 and s >= threshold
        ) / sum(self.labels)
        self.assertGreaterEqual(achieved, 0.90)
        self.assertGreater(specificity, 0.0)

    def test_sensitivity_first_is_not_the_same_as_specificity_first(self) -> None:
        """On an imperfect model the two rules disagree, which is exactly why the
        choice has to be made deliberately rather than inherited from a tutorial.
        The old rule (95% specificity) drove sensitivity to zero on this data."""
        _, sens_first = specificity_at_sensitivity(self.scores, self.labels, 0.90)
        _, spec_first = sensitivity_at_specificity(self.scores, self.labels, 0.95)
        self.assertNotAlmostEqual(sens_first, spec_first)
        self.assertLess(sens_first, spec_first, "sensitivity-first must set a lower threshold")


class GeneratorHonestyTests(unittest.TestCase):
    """The generator must produce a benchmark that can be failed."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.studies = generate_cohort(cls.tmp.name, count=40, seed=4242)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_class_conditional_severity_distributions_overlap(self) -> None:
        suspicious = [s.severity for s in self.studies if s.label == 1]
        benign = [s.severity for s in self.studies if s.label == 0]
        self.assertGreater(max(benign), min(suspicious), "the classes must not be separable")
        self.assertGreater(statistics.fmean(suspicious), statistics.fmean(benign))

    def test_severity_means_match_the_documented_constants(self) -> None:
        self.assertLess(SEVERITY_MEAN_BENIGN, SEVERITY_MEAN_SUSPICIOUS)
        typical = [s for s in self.studies if not s.atypical]
        suspicious = statistics.fmean([s.severity for s in typical if s.label == 1])
        benign = statistics.fmean([s.severity for s in typical if s.label == 0])
        self.assertAlmostEqual(suspicious, SEVERITY_MEAN_SUSPICIOUS, delta=0.12)
        self.assertAlmostEqual(benign, SEVERITY_MEAN_BENIGN, delta=0.12)

    def test_some_cases_are_deliberately_atypical(self) -> None:
        """Label noise is a feature. Without it the confidence gate never fires
        and the indeterminate band is dead code."""
        atypical = [s for s in self.studies if s.atypical]
        self.assertTrue(atypical, "a cohort of 40 should contain at least one atypical case")
        self.assertLess(len(atypical) / len(self.studies), LABEL_NOISE * 4)
        for study in atypical:
            self.assertIn("ATYPICAL", study.truth_note)

    def test_no_single_feature_separates_the_classes_perfectly(self) -> None:
        """The regression test for the AUROC-of-1.000 problem. If any one
        measurement ranks the cohort perfectly, the benchmark is measuring the
        generator and every downstream number is theatre.
        """
        vectors, labels = [], []
        for study in self.studies:
            clean, _ = dicom.deidentify(dicom.read_file(study.path))
            vectors.append(imaging.preprocess(clean).vector())
            labels.append(study.label)
        for index, name in enumerate(imaging.FEATURE_NAMES):
            column = [v[index] for v in vectors]
            separation = max(auroc(column, labels), 1.0 - auroc(column, labels))
            self.assertLess(separation, 0.99, f"{name} separates the classes almost perfectly")


if __name__ == "__main__":
    unittest.main()
