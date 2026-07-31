"""Regression tests for three defects found by cold-start probing, not by the suite.

All three were invisible to the existing tests because the goldset only asks
questions the corpus can answer. Every one of them produced confident, fully
cited, high-groundedness output, which is the most dangerous failure mode this
project claims to prevent -- so each gets a test that fails if it returns.

  1. The coverage gate could be bypassed. It returned max(question_coverage,
     clinical_signal_coverage), so any patient with a rich record cleared the
     threshold no matter what was asked. "What is the capital of France?"
     returned eight sentences of nephrology guidance, cited, refused=False.

  2. The extractive provider degenerated. Sentences were ranked by question
     overlap, but zero-overlap sentences were kept, so when nothing matched the
     sort fell through to length and every question received the same two
     shortest sentences per block.

  3. The polarity gate silently disarmed. It compared a claim's negation cues
     against the whole cited chunk. Chunks nearly always contain a negation
     somewhere, so the set difference was empty and an inverted claim passed.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nullius.app import Principal, build_app  # noqa: E402
from nullius.llm import ExtractiveProvider, TamperedProvider  # noqa: E402
from nullius.verification import check_claim  # noqa: E402


class ScopeGateTests(unittest.TestCase):
    """The coverage gate must judge the question, not the patient's record."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.app = build_app(
            corpus_dir=str(ROOT / "corpus"),
            db_path=str(Path(cls.tmp.name) / "scope.db"),
            provider="extractive",
        )
        cls.principal = Principal(user_id="dr.test", role="clinician")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.app.store.close()
        cls.tmp.cleanup()

    def ask(self, question: str) -> dict:
        return self.app.ask(principal=self.principal, question=question, patient_id="pat-001")

    def test_out_of_scope_questions_are_refused_even_for_a_rich_patient(self) -> None:
        for question in (
            "What is the capital of France?",
            "Who won the 1998 World Cup?",
            "Write me a poem about the sea",
            "How do I rebuild a carburettor?",
        ):
            with self.subTest(question=question):
                answer = self.ask(question)
                self.assertTrue(answer["refused"], f"answered an out-of-scope question: {question}")
                self.assertEqual(answer["refusal_reason"], "insufficient_query_coverage")

    def test_a_deictic_question_is_still_answerable(self) -> None:
        """The counterweight. Blocking off-topic questions must not block the
        vague ones clinicians actually ask, which carry no topical terms at all
        and are answered from the patient's own record."""
        answer = self.ask("Why is this happening?")
        self.assertFalse(answer["refused"], answer["refusal_reason"])
        self.assertTrue(answer["answer"])

    def test_clinical_questions_are_still_answered(self) -> None:
        for question in (
            "How should severe hyperkalaemia be treated?",
            "At what eGFR should metformin be stopped?",
            "When should anaemia in chronic kidney disease be investigated?",
        ):
            with self.subTest(question=question):
                self.assertFalse(self.ask(question)["refused"], question)

    def test_different_questions_receive_different_answers(self) -> None:
        """The symptom that exposed the degenerate provider: five unrelated
        questions returned one identical paragraph."""
        answers = {
            self.ask(q)["answer"]
            for q in (
                "How should severe hyperkalaemia be treated?",
                "When should anaemia in chronic kidney disease be investigated?",
                "How often should stage 3b chronic kidney disease be monitored?",
            )
        }
        self.assertEqual(len(answers), 3, "distinct questions collapsed to the same answer")


class ExtractiveSelectionTests(unittest.TestCase):
    BLOCKS = [
        "[1] KDIGO\nPotassium above 6.0 mmol/L is a medical emergency requiring urgent "
        "electrocardiography. Ferritin and transferrin saturation should be measured before "
        "starting an erythropoiesis-stimulating agent."
    ]

    def test_only_sentences_relevant_to_the_question_are_selected(self) -> None:
        response = ExtractiveProvider().generate(
            question="When is potassium an emergency?", context_blocks=self.BLOCKS, system=""
        )
        self.assertIn("potassium", response.text.lower())
        self.assertNotIn("ferritin", response.text.lower(), "unrelated sentence was emitted")

    def test_a_question_sharing_no_vocabulary_falls_back_to_a_summary(self) -> None:
        """Pass 2 exists for deictic questions, so it must still produce text."""
        response = ExtractiveProvider().generate(
            question="Why is this happening?", context_blocks=self.BLOCKS, system=""
        )
        self.assertTrue(response.text.strip())


class PolarityGateTests(unittest.TestCase):
    def test_inversion_is_caught_when_the_chunk_negates_elsewhere(self) -> None:
        """The exact shape of the bug: the claim inverts its own source sentence,
        while an unrelated sentence in the same chunk already contains 'not'.
        Pooling cues across the chunk made the difference empty."""
        cited = (
            "Serum potassium above 5.5 mmol/L requires prompt review. "
            "Metformin should not be started below an eGFR of 30."
        )
        claim = "Serum potassium above 5.5 mmol/L requires not prompt review [1]."
        self.assertTrue(check_claim(claim, cited).polarity_conflict)

    def test_a_faithful_claim_is_not_flagged(self) -> None:
        cited = "Metformin should not be started below an eGFR of 30 mL/min/1.73m2."
        claim = "Metformin should not be started below an eGFR of 30 mL/min/1.73m2 [1]."
        self.assertFalse(check_claim(claim, cited).polarity_conflict)

    def test_dropping_a_negation_is_left_to_the_overlap_score(self) -> None:
        cited = "Ferritin should not be used alone. Transferrin saturation is also measured."
        claim = "Transferrin saturation is also measured [1]."
        self.assertFalse(check_claim(claim, cited).polarity_conflict)


class TamperGuaranteeTests(unittest.TestCase):
    """A red-team arm that sometimes does nothing measures nothing."""

    NO_DIGITS = ["[1] Guideline\nFerritin and transferrin saturation should be measured before treatment."]

    def test_numeric_tamper_is_never_a_no_op(self) -> None:
        clean = ExtractiveProvider().generate(
            question="Which iron studies are measured?", context_blocks=self.NO_DIGITS, system=""
        )
        tampered = TamperedProvider("numeric").generate(
            question="Which iron studies are measured?", context_blocks=self.NO_DIGITS, system=""
        )
        self.assertNotEqual(clean.text, tampered.text, "numeric arm emitted untampered text")

    def test_polarity_tamper_is_never_a_no_op(self) -> None:
        clean = ExtractiveProvider().generate(
            question="Which iron studies are measured?", context_blocks=self.NO_DIGITS, system=""
        )
        tampered = TamperedProvider("polarity").generate(
            question="Which iron studies are measured?", context_blocks=self.NO_DIGITS, system=""
        )
        self.assertNotEqual(clean.text, tampered.text, "polarity arm emitted untampered text")


if __name__ == "__main__":
    unittest.main(verbosity=2)
