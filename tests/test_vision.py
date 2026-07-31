"""Tests for the imaging path: DICOM, preprocessing, ONNX, and the gates.

The gate tests matter more than the accuracy tests. Accuracy on synthetic data
is close to meaningless; whether a bad frame gets refused is not.
"""

from __future__ import annotations

import json
import random
import statistics
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nullius import dicom, imaging, onnx  # noqa: E402
from nullius.app import AccessDenied, Principal  # noqa: E402
from nullius.lesions import generate_cohort  # noqa: E402
from nullius.vision import (  # noqa: E402
    ModelBundle,
    VisionGates,
    VisionPipeline,
    VisionThresholds,
    auroc,
    expected_calibration_error,
    sensitivity_at_specificity,
)

MODEL_PATH = ROOT / "models" / "lesion-mlp.onnx"


def fixture_bundle(directory: Path) -> ModelBundle:
    """Build a valid ModelBundle without requiring `make train` to have been run.

    A fresh clone has no models/lesion-mlp.onnx, and the previous version of this
    file skipped ten gate tests when it was missing - so the gates went
    unverified on exactly the checkout where verification matters most, and the
    suite still printed OK. The weights below are deterministic rather than
    trained, which is legitimate here because these tests assert refusal
    behaviour: whether a blank frame is rejected must not depend on how good the
    model is.
    """
    cohort = directory / "fixture-dicom"
    studies = generate_cohort(cohort, count=8, seed=7)
    vectors = []
    for study in studies:
        clean, _ = dicom.deidentify(dicom.read_file(study.path))
        vectors.append(imaging.preprocess(clean).vector())
    columns = list(zip(*vectors))

    n_in = len(imaging.FEATURE_NAMES)
    hidden = 4
    rng = random.Random(7)
    w1 = [[rng.gauss(0, 0.6) for _ in range(hidden)] for _ in range(n_in)]
    w2 = [[rng.gauss(0, 0.6)] for _ in range(hidden)]
    model_path = directory / "lesion-mlp.onnx"
    onnx.export_mlp(model_path, w1, [0.0] * hidden, w2, [0.0])
    model_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "name": "test-fixture",
                "feature_names": list(imaging.FEATURE_NAMES),
                "feature_mean": [statistics.fmean(c) for c in columns],
                "feature_std": [max(statistics.pstdev(c), 1e-3) for c in columns],
                "operating_point": 0.5,
            }
        ),
        encoding="utf-8",
    )
    return ModelBundle(model_path)


class DicomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.studies = generate_cohort(cls.tmp.name, count=4, seed=11)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_written_file_has_part10_preamble_and_magic(self) -> None:
        blob = Path(self.studies[0].path).read_bytes()
        self.assertEqual(blob[:128], b"\x00" * 128, "128-byte preamble required")
        self.assertEqual(blob[128:132], b"DICM")

    def test_decode_is_idempotent(self) -> None:
        dataset = dicom.read_file(self.studies[0].path)
        again = dicom.decode(dicom.encode(dataset.elements))
        self.assertEqual(dataset.elements, again.elements)

    def test_transfer_syntax_and_geometry(self) -> None:
        dataset = dicom.read_file(self.studies[0].path)
        self.assertEqual(dataset.meta["TransferSyntaxUID"], dicom.EXPLICIT_VR_LITTLE_ENDIAN)
        self.assertEqual(dataset.rows, 160)
        self.assertEqual(len(dataset.pixels()), 160 * 160)

    def test_rejects_non_dicom(self) -> None:
        with self.assertRaises(dicom.DicomError):
            dicom.decode(b"not a dicom file" * 20)

    def test_rejects_pixel_data_that_contradicts_the_header(self) -> None:
        dataset = dicom.read_file(self.studies[0].path)
        dataset.elements["PixelData"] = dataset.elements["PixelData"][:-4]
        with self.assertRaises(dicom.DicomError):
            dataset.pixels()

    def test_deidentification_removes_every_direct_identifier(self) -> None:
        dataset = dicom.read_file(self.studies[0].path)
        self.assertTrue(dicom.residual_identifiers(dataset), "fixture must start with PHI present")
        clean, mapping = dicom.deidentify(dataset)
        self.assertEqual(dicom.residual_identifiers(clean), [])
        self.assertEqual(clean["PatientIdentityRemoved"], "YES")
        self.assertIn("DeidentificationMethod", clean.elements)
        self.assertTrue(mapping)

    def test_deidentification_preserves_pixels(self) -> None:
        dataset = dicom.read_file(self.studies[0].path)
        clean, _ = dicom.deidentify(dataset)
        self.assertEqual(clean["PixelData"], dataset["PixelData"])

    def test_pseudonyms_are_stable_and_salted(self) -> None:
        self.assertEqual(dicom.pseudonym("pat-001"), dicom.pseudonym("pat-001"))
        self.assertNotEqual(dicom.pseudonym("pat-001"), dicom.pseudonym("pat-001", salt="other"))
        self.assertNotEqual(dicom.pseudonym("pat-001"), dicom.pseudonym("pat-002"))

    def test_dates_are_truncated_to_year(self) -> None:
        clean, _ = dicom.deidentify(dicom.read_file(self.studies[0].path))
        self.assertTrue(str(clean["PatientBirthDate"]).endswith("0101"))
        self.assertNotIn("StudyTime", clean.elements)


class ImagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.studies = generate_cohort(cls.tmp.name, count=6, seed=5)
        cls.pre = {
            study.study_id: imaging.preprocess(dicom.deidentify(dicom.read_file(study.path))[0])
            for study in cls.studies
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_preprocessing_is_deterministic(self) -> None:
        study = self.studies[0]
        again = imaging.preprocess(dicom.deidentify(dicom.read_file(study.path))[0])
        self.assertEqual(self.pre[study.study_id].features, again.features)

    def test_features_are_named_bounded_and_complete(self) -> None:
        features = self.pre[self.studies[0].study_id].features
        self.assertEqual(tuple(features), imaging.FEATURE_NAMES)
        for name, value in features.items():
            self.assertGreaterEqual(value, 0.0, name)
            self.assertLessEqual(value, 1.0, name)

    def test_every_feature_has_a_human_readable_label(self) -> None:
        for name in imaging.FEATURE_NAMES:
            self.assertIn(name, imaging.FEATURE_LABELS)

    def test_pipeline_records_its_own_steps(self) -> None:
        steps = self.pre[self.studies[0].study_id].steps
        self.assertTrue(any("median" in s for s in steps))
        self.assertTrue(any("Otsu" in s for s in steps))

    def test_segmentation_finds_one_bounded_region(self) -> None:
        for study_id, pre in self.pre.items():
            area = pre.quality["lesion_area_fraction"]
            self.assertGreater(area, 0.01, study_id)
            self.assertLess(area, 0.8, study_id)

    def test_erosion_shrinks_a_mask(self) -> None:
        mask = self.pre[self.studies[0].study_id].mask
        area = sum(1 for row in mask for v in row if v)
        eroded = sum(1 for row in imaging.erode(mask, 2) for v in row if v)
        self.assertLess(eroded, area)
        self.assertGreater(eroded, 0)

    def test_otsu_threshold_is_interior(self) -> None:
        image = self.pre[self.studies[0].study_id].image
        threshold = imaging.otsu_threshold(image)
        self.assertTrue(0.0 < threshold < 1.0)

    def test_suspicious_lesions_are_more_irregular(self) -> None:
        suspicious = [self.pre[s.study_id].features for s in self.studies if s.label == 1]
        benign = [self.pre[s.study_id].features for s in self.studies if s.label == 0]
        if not suspicious or not benign:
            self.skipTest("cohort did not contain both classes")
        mean = lambda rows, key: sum(r[key] for r in rows) / len(rows)  # noqa: E731
        self.assertGreater(mean(suspicious, "asymmetry"), mean(benign, "asymmetry"))
        self.assertGreater(mean(suspicious, "border_irregularity"), mean(benign, "border_irregularity"))

    def test_render_produces_a_valid_png(self) -> None:
        pre = self.pre[self.studies[0].study_id]
        png = imaging.render(pre.image, pre.mask)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertIn(b"IEND", png[-12:])

    def test_resize_changes_shape_only(self) -> None:
        image = [[0.5] * 20 for _ in range(20)]
        out = imaging.resize(image, 8)
        self.assertEqual(len(out), 8)
        self.assertEqual(len(out[0]), 8)
        self.assertAlmostEqual(out[3][3], 0.5, places=6)


class OnnxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tiny.onnx"
        # y = sigmoid(relu(x . W1 + b1) . W2 + b2)
        self.w1 = [[1.0, 0.0], [0.0, 1.0]]
        self.b1 = [0.0, 0.0]
        self.w2 = [[1.0], [1.0]]
        self.b2 = [0.0]
        onnx.export_mlp(self.path, self.w1, self.b1, self.w2, self.b2)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_exported_file_parses_as_the_expected_graph(self) -> None:
        graph = onnx.load_graph(self.path)
        self.assertEqual(graph.ir_version, onnx.ONNX_IR_VERSION)
        self.assertEqual(graph.opset, onnx.DEFAULT_OPSET)
        self.assertEqual([n.op_type for n in graph.nodes], list(onnx.SUPPORTED_OPS[:2]) + ["Relu", "MatMul", "Add", "Sigmoid"])
        self.assertEqual(graph.inputs, ["features"])
        self.assertEqual(graph.outputs, ["probability"])

    def test_initializers_survive_the_round_trip(self) -> None:
        graph = onnx.load_graph(self.path)
        self.assertEqual(graph.initializers["W1"].dims, [2, 2])
        self.assertEqual(graph.initializers["W1"].values, [1.0, 0.0, 0.0, 1.0])
        self.assertEqual(graph.initializers["B2"].dims, [1])

    def test_inference_matches_a_hand_computed_forward_pass(self) -> None:
        session = onnx.PythonSession(self.path)
        # x = [1, 2] -> hidden [1, 2] -> logit 3 -> sigmoid(3)
        expected = 1.0 / (1.0 + pow(2.718281828459045, -3.0))
        self.assertAlmostEqual(session.run([1.0, 2.0])[0], expected, places=6)

    def test_relu_actually_clamps(self) -> None:
        session = onnx.PythonSession(self.path)
        # Negative inputs are zeroed by Relu, so the logit is exactly b2 = 0.
        self.assertAlmostEqual(session.run([-5.0, -5.0])[0], 0.5, places=6)

    def test_shape_mismatch_is_rejected(self) -> None:
        session = onnx.PythonSession(self.path)
        with self.assertRaises(onnx.OnnxError):
            session.run([1.0, 2.0, 3.0])

    def test_unsupported_operators_are_refused_not_ignored(self) -> None:
        node = onnx._node_proto("Conv", ["features"], ["probability"], "conv")
        graph = onnx._bytes_field(1, node) + onnx._string_field(2, "bad")
        model = onnx._varint_field(1, 8) + onnx._bytes_field(7, graph)
        path = Path(self.tmp.name) / "bad.onnx"
        path.write_bytes(model)
        with self.assertRaises(onnx.OnnxError):
            onnx.load_graph(path)

    def test_float64_tensors_are_refused(self) -> None:
        tensor = onnx._varint_field(1, 2) + onnx._varint_field(2, 11) + onnx._string_field(8, "W")
        graph = onnx._string_field(2, "g") + onnx._bytes_field(5, tensor)
        model = onnx._varint_field(1, 8) + onnx._bytes_field(7, graph)
        path = Path(self.tmp.name) / "f64.onnx"
        path.write_bytes(model)
        with self.assertRaises(onnx.OnnxError):
            onnx.load_graph(path)

    @unittest.skipUnless(onnx.onnxruntime_available(), "onnxruntime is not installed")
    def test_onnxruntime_agrees_with_the_pure_python_interpreter(self) -> None:
        """The strongest available check that the serialisation is really valid."""
        import random

        rng = random.Random(3)
        big = Path(self.tmp.name) / "big.onnx"
        w1 = [[rng.gauss(0, 1) for _ in range(6)] for _ in range(9)]
        w2 = [[rng.gauss(0, 1)] for _ in range(6)]
        onnx.export_mlp(big, w1, [rng.gauss(0, 1) for _ in range(6)], w2, [0.1])
        python_session = onnx.PythonSession(big)
        runtime_session = onnx.OnnxRuntimeSession(big)
        for _ in range(50):
            features = [rng.uniform(0, 1) for _ in range(9)]
            self.assertAlmostEqual(
                python_session.run(features)[0], runtime_session.run(features)[0], places=5
            )


class MetricTests(unittest.TestCase):
    def test_auroc_matches_a_known_value(self) -> None:
        self.assertAlmostEqual(auroc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75, places=6)

    def test_auroc_handles_ties(self) -> None:
        self.assertAlmostEqual(auroc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]), 0.5, places=6)

    def test_perfect_ranking_is_one(self) -> None:
        self.assertAlmostEqual(auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0, places=6)

    def test_sensitivity_at_specificity_respects_the_constraint(self) -> None:
        scores = [0.1, 0.2, 0.3, 0.9, 0.95]
        labels = [0, 0, 0, 1, 1]
        sensitivity, threshold = sensitivity_at_specificity(scores, labels, 0.95)
        self.assertAlmostEqual(sensitivity, 1.0)
        self.assertGreater(threshold, 0.3)

    def test_calibration_error_is_zero_when_confidence_matches_frequency(self) -> None:
        scores = [0.0] * 10 + [1.0] * 10
        labels = [0] * 10 + [1] * 10
        self.assertAlmostEqual(expected_calibration_error(scores, labels), 0.0, places=6)

    def test_calibration_error_catches_overconfidence(self) -> None:
        scores = [0.99] * 10
        labels = [0] * 5 + [1] * 5
        self.assertGreater(expected_calibration_error(scores, labels), 0.4)


class VisionPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        # Never skip: fall back to a fixture model on a cold clone.
        cls.bundle = ModelBundle(MODEL_PATH) if MODEL_PATH.exists() else fixture_bundle(root)
        cls.studies = generate_cohort(root / "cohort", count=4, seed=99)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def pipeline(self, **kwargs) -> VisionPipeline:
        return VisionPipeline(self.bundle, **kwargs)

    def test_a_normal_study_is_classified_with_an_explanation(self) -> None:
        result = self.pipeline().classify(self.studies[0].path)
        if not result.served:
            self.skipTest(f"study fell into {result.refusal_reason}")
        self.assertIsNotNone(result.probability)
        self.assertTrue(result.attributions, "a served result must explain itself")
        self.assertIn(result.attributions[0]["feature"], imaging.FEATURE_NAMES)
        self.assertIn("not a diagnosis", result.disclaimer)

    def test_result_never_carries_a_direct_identifier(self) -> None:
        result = self.pipeline().classify(self.studies[0].path)
        payload = str(result.to_dict())
        for study in self.studies:
            self.assertNotIn(study.patient_name, payload)

    def test_unreadable_input_is_refused_not_guessed(self) -> None:
        result = self.pipeline().classify(b"garbage that is not dicom" * 10)
        self.assertFalse(result.served)
        self.assertEqual(result.refusal_reason, "unreadable_dicom")

    def test_blank_frame_is_refused_on_quality(self) -> None:
        dataset = dicom.read_file(self.studies[0].path)
        flat = struct.pack("<%dH" % (160 * 160), *([2048] * (160 * 160)))
        dataset.elements["PixelData"] = flat
        result = self.pipeline().classify(dicom.encode(dataset.elements))
        self.assertFalse(result.served)
        self.assertIn(result.refusal_reason, {"image_quality_insufficient", "out_of_distribution"})

    def test_distribution_gate_blocks_extrapolation(self) -> None:
        strict = self.pipeline(thresholds=VisionThresholds(max_feature_z=0.0))
        result = strict.classify(self.studies[0].path)
        self.assertFalse(result.served)
        self.assertEqual(result.refusal_reason, "out_of_distribution")

    def test_indeterminate_scores_are_referred_to_a_human(self) -> None:
        cautious = self.pipeline(thresholds=VisionThresholds(review_low=0.0, review_high=1.0))
        result = cautious.classify(self.studies[0].path)
        self.assertFalse(result.served)
        self.assertEqual(result.refusal_reason, "indeterminate_needs_review")
        self.assertEqual(result.triage, "human review")
        self.assertIsNotNone(result.probability, "a referral should still show its score")

    def test_disarming_a_gate_changes_the_outcome(self) -> None:
        thresholds = VisionThresholds(max_feature_z=0.0)
        blocked = self.pipeline(thresholds=thresholds).classify(self.studies[0].path)
        allowed = self.pipeline(
            thresholds=thresholds, gates=VisionGates().without("distribution")
        ).classify(self.studies[0].path)
        self.assertFalse(blocked.served)
        self.assertNotEqual(allowed.refusal_reason, "out_of_distribution")

    def test_gates_are_immutable_and_typo_resistant(self) -> None:
        gates = VisionGates()
        self.assertTrue(VisionGates().distribution)
        self.assertFalse(gates.without("distribution").distribution)
        self.assertTrue(gates.distribution, "without() must not mutate in place")
        with self.assertRaises(ValueError):
            gates.without("not_a_gate")

    def test_attribution_names_a_real_measurement(self) -> None:
        result = self.pipeline().classify(self.studies[0].path)
        if not result.served:
            self.skipTest("study was refused")
        top = result.attributions[0]
        self.assertIn(top["direction"], {"raises", "lowers", "neutral"})
        self.assertEqual(top["label"], imaging.FEATURE_LABELS[top["feature"]])

    def test_model_bundle_exposes_its_provenance(self) -> None:
        self.assertIn("caveat", self.bundle.metadata)
        self.assertIn("synthetic", self.bundle.metadata["caveat"].lower())
        self.assertEqual(self.bundle.feature_names, list(imaging.FEATURE_NAMES))


class ImagingAccessControlTests(unittest.TestCase):
    def test_nurse_cannot_read_imaging(self) -> None:
        with self.assertRaises(AccessDenied):
            Principal("nurse-1", "nurse").require("imaging.read")

    def test_radiologist_can_read_imaging_but_not_ask_the_copilot(self) -> None:
        Principal("rad-1", "radiologist").require("imaging.read")
        with self.assertRaises(AccessDenied):
            Principal("rad-1", "radiologist").require("copilot.ask")

    def test_clinician_can_read_imaging(self) -> None:
        Principal("doc-1", "clinician").require("imaging.read")


if __name__ == "__main__":
    unittest.main()
