"""Clinical copilot: retrieve -> generate -> verify -> (answer | refuse).

The verification step is the point of this file. An unverified RAG answer is a
liability in a clinical setting, so the pipeline enforces four gates before any
text reaches a clinician:

  1. Retrieval gate   - no supporting chunk above threshold => refuse, do not generate.
  2. Coverage gate    - at least one retrieved passage must address a meaningful share
                        of the question's own terms. Without this, BM25 happily returns
                        a hyperkalaemia passage for a question about ventilator settings
                        because both contain the word "severe", and the model then writes
                        a fluent, perfectly cited, completely irrelevant answer. This is
                        the single most dangerous RAG failure mode: high groundedness
                        against the wrong evidence.
  3. Citation gate    - every sentence must carry a marker resolving to a retrieved chunk.
  4. Groundedness gate - each sentence's content tokens must overlap its cited chunk.
  5. Numeric gate     - every number in a sentence must appear in the evidence it cites.
                        Thresholds and doses are the entire clinical payload, and a
                        fabricated one passes token overlap easily because every other
                        word in the sentence is genuine.
  6. Polarity gate    - a sentence may not introduce negation its evidence does not
                        contain. "Potassium above 6.0 is not an emergency" shares nearly
                        all its tokens with the passage saying the opposite.
  7. Scope gate       - dosing/diagnosis instructions are reframed as clinician-facing
                        suggestions; the copilot advises, it does not prescribe.

Each gate is individually switchable, not for configurability but for measurement:
`evaluate.ablation` disables one at a time and reports what it was worth. A gate
that cannot be shown to catch anything does not belong in a safety argument.

Refusing is a success state, not an error. Every answer carries the trace id and
the exact evidence it used, and every request is written to the audit log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .interactions import check_interactions
from .labs import review_patient
from .llm import LLMProvider, LLMResponse, build_provider, split_sentences
from .observability import TRACER
from .retrieval import Retriever, ScoredChunk, tokenize
from .verification import check_claim

SYSTEM_PROMPT = (
    "You are Nullius, a clinical decision support assistant for licensed clinicians. "
    "Answer only from the numbered evidence blocks provided. "
    "Every sentence must end with a citation marker such as [1] identifying the block it came from. "
    "If the evidence does not answer the question, reply exactly: INSUFFICIENT_EVIDENCE. "
    "Never state a diagnosis as fact and never issue a prescription; surface options and their evidence "
    "for the responsible clinician to accept or reject."
)

REFUSAL_TEXT = (
    "I cannot answer this from the indexed guideline corpus. No retrieved passage met the "
    "evidence threshold, so answering would mean generating unsupported clinical content."
)

CITATION = re.compile(r"\[(\d+)\]")

# Phrases that turn advice into an order. Rewritten, not blocked, so the clinical
# content survives while the authority stays with the clinician.
PRESCRIPTIVE_REWRITES = [
    (re.compile(r"\byou should\b", re.I), "the guideline recommends that the clinician"),
    (re.compile(r"\bmust be stopped\b", re.I), "is recommended for review and possible discontinuation"),
    (re.compile(r"\bstop the\b", re.I), "consider discontinuing the"),
    (re.compile(r"\bprescribe\b", re.I), "consider prescribing"),
]


@dataclass
class SentenceCheck:
    sentence: str
    citations: list[int]
    support: float
    grounded: bool
    numbers: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    polarity_conflict: bool = False
    nli_entailed: bool = True


@dataclass(frozen=True)
class Gates:
    """Which verification gates are active.

    Exists so the evaluation harness can quantify each gate's contribution by
    ablation. Production runs use the default: all on.
    """

    coverage: bool = True
    citation: bool = True
    groundedness: bool = True
    numeric: bool = True
    polarity: bool = True
    nli: bool = False

    def without(self, name: str) -> "Gates":
        return Gates(**{**self.__dict__, name: False})


@dataclass
class CopilotAnswer:
    question: str
    patient_id: str | None
    answer: str
    refused: bool
    refusal_reason: str | None
    evidence: list[dict[str, Any]]
    sentence_checks: list[dict[str, Any]]
    groundedness: float
    citation_coverage: float
    confidence: str
    trace_id: str
    usage: dict[str, Any]
    patient_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "patient_id": self.patient_id,
            "answer": self.answer,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "evidence": self.evidence,
            "sentence_checks": self.sentence_checks,
            "groundedness": round(self.groundedness, 3),
            "citation_coverage": round(self.citation_coverage, 3),
            "confidence": self.confidence,
            "trace_id": self.trace_id,
            "usage": self.usage,
            "patient_context": self.patient_context,
        }


def support_score(sentence: str, cited_text: str) -> float:
    """Fraction of the sentence's content tokens present in its cited evidence.

    Token overlap is a weak proxy for entailment, and that is an accepted
    limitation (docs/ADR-003): it is cheap, deterministic, needs no judge model,
    and reliably catches the failure that matters most here - a fluent sentence
    citing a passage that does not contain it.
    """
    tokens = [t for t in tokenize(sentence) if not CITATION.fullmatch(t)]
    if not tokens:
        return 1.0
    cited = set(tokenize(cited_text))
    return sum(1 for t in tokens if t in cited) / len(tokens)


class Copilot:
    def __init__(
        self,
        store,
        retriever: Retriever,
        provider: LLMProvider | None = None,
        judge_provider: LLMProvider | None = None,
        *,
        min_retrieval_score: float = 1.0,
        groundedness_threshold: float = 0.55,
        min_question_coverage: float = 0.35,
        top_k: int = 4,
        gates: Gates | None = None,
    ) -> None:
        self.gates = gates or Gates()
        self.store = store
        self.retriever = retriever
        self.provider = provider or build_provider("extractive")
        self.judge_provider = judge_provider
        self.min_retrieval_score = min_retrieval_score
        self.groundedness_threshold = groundedness_threshold
        self.min_question_coverage = min_question_coverage
        self.top_k = top_k

    @staticmethod
    def _best_coverage(terms: set[str], hits: list[ScoredChunk]) -> float:
        if not terms:
            return 0.0
        best = 0.0
        for hit in hits:
            chunk_terms = set(tokenize(f"{hit.chunk.doc_title} {hit.chunk.section} {hit.chunk.text}"))
            best = max(best, len(terms & chunk_terms) / len(terms))
        return best

    @staticmethod
    def _clinical_signal_terms(context: dict[str, Any]) -> set[str]:
        """Terms drawn from *verified* patient data: active problems, abnormal
        analytes and interacting drug names. These are database facts, not model
        output, so they are legitimate evidence that a passage is on-topic."""
        parts: list[str] = []
        for cond in context.get("conditions", []):
            parts.append(cond.get("display") or "")
        for finding in context.get("critical_values", []) + context.get("adverse_trends", []):
            parts.append(finding.get("name") or "")
        for interaction in context.get("interactions", []):
            parts.extend(interaction.get("drugs", []))
        return set(tokenize(" ".join(parts)))

    # Words that carry no topic of their own. A question built only from these is
    # deictic ("why is this happening?") and must be answered from the patient's
    # record. A question containing anything else is topical, and if none of those
    # topical terms exist anywhere in the corpus, it is out of scope.
    DEICTIC_TERMS = frozenset({
        "happening", "happen", "going", "next", "now", "here", "case", "situation",
        "concern", "concerns", "concerning", "worried", "about", "mean", "means",
        "think", "thoughts", "anything", "else", "wrong", "look", "looks", "seems",
        "tell", "explain", "summarise", "summarize", "summary", "overview", "review",
    })

    @classmethod
    def _topical_terms(cls, question: str) -> set[str]:
        return set(tokenize(question)) - cls.DEICTIC_TERMS

    def _is_off_topic(self, question: str, hits: list[ScoredChunk], context: dict[str, Any] | None) -> bool:
        """True when the question is about something this system has no evidence for.

        Without this check the clinical-signal fallback below is a hole in the
        coverage gate: any patient with a rich record scores highly on their own
        conditions, so *every* question clears the threshold regardless of what was
        actually asked. Asking a nephrology corpus for the capital of France would
        return eight sentences about chronic kidney disease, fully cited.
        """
        topical = self._topical_terms(question)
        if not topical:
            return False
        known: set[str] = set(self._clinical_signal_terms(context or {}))
        for hit in hits:
            known |= set(tokenize(f"{hit.chunk.doc_title} {hit.chunk.section} {hit.chunk.text}"))
        return not (topical & known)

    def question_coverage(self, question: str, hits: list[ScoredChunk], context: dict[str, Any] | None = None) -> float:
        """Relevance gate input: how much of what we are actually asking about does
        the best retrieved passage cover?

        Two term sets are scored independently and the better one wins:

          * the question's own content terms - handles explicit questions such as
            "what potassium level requires urgent review";
          * the patient's verified clinical signals - handles the deictic questions
            clinicians really ask ("why is this happening?"), which contain almost
            no content words of their own.

        Scoring them separately rather than as one pooled set matters: pooling lets
        a large context dilute a precise question, and a vague question dilute a
        strong context match.
        """
        own = self._best_coverage(set(tokenize(question)), hits)
        if self._is_off_topic(question, hits, context):
            # No context rescue for a question naming things the corpus has never
            # heard of. Scored on its own terms, which is what gets it refused.
            return own
        return max(own, self._best_coverage(self._clinical_signal_terms(context or {}), hits))

    # --------------------------------------------------------------- context
    def patient_context(self, patient_id: str) -> dict[str, Any]:
        with TRACER.span("copilot.patient_context", **{"patient.id": patient_id}):
            patient = self.store.patient(patient_id)
            if not patient:
                return {}
            labs = review_patient(self.store, patient_id)
            meds = self.store.medications(patient_id)
            return {
                "patient": patient,
                "conditions": self.store.conditions(patient_id),
                "medications": meds,
                "critical_values": labs["critical_values"],
                "adverse_trends": labs["adverse_trends"],
                "monitoring_gaps": labs["monitoring_gaps"],
                "interactions": check_interactions(meds, conditions=self.store.conditions(patient_id)),
            }

    @staticmethod
    def _expand_query(question: str, context: dict[str, Any]) -> str:
        """Add the patient's active problems and abnormal analytes to the query.

        Without this, "why is this happening?" retrieves nothing useful. Expansion
        is additive and logged, so retrieval stays explainable.
        """
        extras: list[str] = []
        for cond in context.get("conditions", [])[:4]:
            if cond.get("display"):
                extras.append(cond["display"])
        for finding in (context.get("critical_values", []) + context.get("adverse_trends", []))[:4]:
            extras.append(finding["name"])
        for interaction in context.get("interactions", [])[:3]:
            extras.extend(interaction["drugs"])
        return f"{question} {' '.join(dict.fromkeys(extras))}".strip()

    # ----------------------------------------------------------------- answer
    def ask(self, question: str, patient_id: str | None = None, *, actor: str = "dr.demo",
            actor_role: str = "clinician") -> CopilotAnswer:
        with TRACER.span("copilot.ask", **{"copilot.question": question, "patient.id": patient_id or ""}) as root:
            context = self.patient_context(patient_id) if patient_id else {}
            query = self._expand_query(question, context) if context else question
            root.set(**{"copilot.expanded_query": query})

            hits: list[ScoredChunk] = self.retriever.search(query, top_k=self.top_k, min_score=self.min_retrieval_score)
            if not hits:
                answer = self._refuse(question, patient_id, context, root.trace_id, "no_retrieval_hits")
                self._audit(actor, actor_role, patient_id, root.trace_id, "copilot.refused:no_retrieval_hits")
                return answer

            coverage = self.question_coverage(question, hits, context)
            root.set(**{"retrieval.question_coverage": round(coverage, 3)})
            TRACER.metrics.observe("nullius_question_coverage", coverage)
            if self.gates.coverage and coverage < self.min_question_coverage:
                answer = self._refuse(question, patient_id, context, root.trace_id, "insufficient_query_coverage", hits)
                self._audit(actor, actor_role, patient_id, root.trace_id, "copilot.refused:insufficient_query_coverage")
                return answer

            blocks = [f"[{i}] {h.chunk.citation}\n{h.chunk.text}" for i, h in enumerate(hits, start=1)]
            with TRACER.span("llm.generate", **{"llm.provider": self.provider.name, "llm.model": self.provider.model}) as gen:
                # Generation scores sentences against the raw question, not the
                # expanded query. Expansion is right for retrieval -- it is how a
                # vague question reaches the correct documents -- but wrong for
                # sentence selection, because the added context terms outnumber the
                # question's own and drown it: asking about metformin dosing and
                # asking about hyperkalaemia treatment both returned whichever
                # sentences best matched the patient's problem list.
                response: LLMResponse = self.provider.generate(
                    question=question, context_blocks=blocks, system=SYSTEM_PROMPT
                )
                gen.set(**{
                    "llm.prompt_tokens": response.prompt_tokens,
                    "llm.completion_tokens": response.completion_tokens,
                    "llm.latency_ms": round(response.latency_ms, 2),
                })
            TRACER.metrics.inc("nullius_llm_tokens_total", response.total_tokens, provider=self.provider.name)
            TRACER.metrics.observe("nullius_llm_latency_ms", response.latency_ms, provider=self.provider.name)

            text = response.text.strip()
            if not text or "INSUFFICIENT_EVIDENCE" in text:
                answer = self._refuse(question, patient_id, context, root.trace_id, "model_declined", hits, response)
                self._audit(actor, actor_role, patient_id, root.trace_id, "copilot.refused:model_declined")
                return answer

            text = self._soften_prescriptive(text)
            checks = self._verify(text, hits)
            cited = [c for c in checks if c.citations]
            citation_coverage = len(cited) / len(checks) if checks else 0.0
            groundedness = (sum(c.support for c in cited) / len(cited)) if cited else 0.0
            root.set(**{
                "quality.groundedness": round(groundedness, 3),
                "quality.citation_coverage": round(citation_coverage, 3),
                "quality.sentences": len(checks),
            })
            TRACER.metrics.observe("nullius_answer_groundedness", groundedness, provider=self.provider.name)
            TRACER.metrics.observe("nullius_answer_citation_coverage", citation_coverage, provider=self.provider.name)

            claim_failures = [c for c in checks if c.unsupported_numbers]
            polarity_failures = [c for c in checks if c.polarity_conflict]
            root.set(**{
                "quality.unsupported_numeric_sentences": len(claim_failures),
                "quality.polarity_conflicts": len(polarity_failures),
            })

            # Ordered most-specific first: a fabricated threshold is a more useful
            # refusal reason than a generic groundedness failure, and the audit log
            # should record which check actually fired.
            if self.gates.numeric and claim_failures:
                TRACER.metrics.inc("nullius_numeric_blocks_total", provider=self.provider.name)
                answer = self._refuse(
                    question, patient_id, context, root.trace_id,
                    "unsupported_numeric_claim", hits, response, checks, groundedness, citation_coverage,
                )
                self._audit(actor, actor_role, patient_id, root.trace_id, "copilot.refused:unsupported_numeric_claim")
                return answer

            if self.gates.polarity and polarity_failures:
                TRACER.metrics.inc("nullius_polarity_blocks_total", provider=self.provider.name)
                answer = self._refuse(
                    question, patient_id, context, root.trace_id,
                    "polarity_conflict", hits, response, checks, groundedness, citation_coverage,
                )
                self._audit(actor, actor_role, patient_id, root.trace_id, "copilot.refused:polarity_conflict")
                return answer

            fails_groundedness = self.gates.groundedness and groundedness < self.groundedness_threshold
            fails_nli = self.gates.nli and any(not c.nli_entailed for c in checks if c.citations)
            fails_citation = self.gates.citation and citation_coverage < 0.99
            if fails_groundedness or fails_citation or fails_nli:
                reason = "failed_nli_gate" if fails_nli else "failed_groundedness_gate"
                TRACER.metrics.inc("nullius_hallucination_blocks_total", provider=self.provider.name)
                answer = self._refuse(
                    question, patient_id, context, root.trace_id,
                    reason, hits, response, checks, groundedness, citation_coverage,
                )
                self._audit(actor, actor_role, patient_id, root.trace_id, f"copilot.refused:{reason}")
                return answer

            TRACER.metrics.inc("nullius_answers_served_total", provider=self.provider.name)
            self._audit(actor, actor_role, patient_id, root.trace_id, f"copilot.answered:{len(hits)}_sources")
            return CopilotAnswer(
                question=question,
                patient_id=patient_id,
                answer=text,
                refused=False,
                refusal_reason=None,
                evidence=[h.to_dict() for h in hits],
                sentence_checks=[c.__dict__ for c in checks],
                groundedness=groundedness,
                citation_coverage=citation_coverage,
                confidence=self._confidence(groundedness, hits),
                trace_id=root.trace_id,
                usage={
                    "provider": response.provider,
                    "model": response.model,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "latency_ms": round(response.latency_ms, 2),
                },
                patient_context=self._summarise_context(context),
            )

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _soften_prescriptive(text: str) -> str:
        for pattern, replacement in PRESCRIPTIVE_REWRITES:
            text = pattern.sub(replacement, text)
        return text

    def _verify(self, text: str, hits: list[ScoredChunk]) -> list[SentenceCheck]:
        with TRACER.span("copilot.verify") as span:
            checks: list[SentenceCheck] = []
            for sentence in split_sentences(text):
                markers = [int(m) for m in CITATION.findall(sentence)]
                valid = [m for m in markers if 1 <= m <= len(hits)]
                cited_text = " ".join(hits[m - 1].chunk.text for m in valid)
                support = support_score(CITATION.sub("", sentence), cited_text) if valid else 0.0
                claim = check_claim(sentence, cited_text) if valid else check_claim(sentence, "")
                
                nli_entailed = True
                if self.gates.nli and self.judge_provider and valid:
                    system = "You are a strict clinical Natural Language Inference judge. Determine if the sentence is fully supported by the evidence. Output exactly YES or NO."
                    resp = self.judge_provider.generate(
                        question=f"Sentence: {CITATION.sub('', sentence)}",
                        context_blocks=[f"Evidence: {cited_text}"],
                        system=system
                    )
                    nli_entailed = "YES" in resp.text.upper()

                checks.append(
                    SentenceCheck(
                        sentence=sentence,
                        citations=valid,
                        support=round(support, 3),
                        grounded=bool(valid) and support >= self.groundedness_threshold and claim.ok and nli_entailed,
                        numbers=claim.numbers,
                        unsupported_numbers=claim.unsupported_numbers,
                        polarity_conflict=claim.polarity_conflict,
                        nli_entailed=nli_entailed,
                    )
                )
            span.set(**{
                "verify.sentences": len(checks),
                "verify.unsupported_numbers": sum(len(c.unsupported_numbers) for c in checks),
                "verify.polarity_conflicts": sum(1 for c in checks if c.polarity_conflict),
                "verify.ungrounded": sum(1 for c in checks if not c.grounded),
            })
            return checks

    @staticmethod
    def _confidence(groundedness: float, hits: list[ScoredChunk]) -> str:
        top = hits[0].score if hits else 0.0
        if groundedness >= 0.8 and top >= 4.0 and len(hits) >= 2:
            return "high"
        if groundedness >= 0.65 and top >= 2.0:
            return "moderate"
        return "low"

    @staticmethod
    def _summarise_context(context: dict[str, Any]) -> dict[str, Any]:
        if not context:
            return {}
        return {
            "conditions": [c["display"] for c in context.get("conditions", [])],
            "critical_values": [
                f"{f['name']} {f['latest_value']} {f['unit']}" for f in context.get("critical_values", [])
            ],
            "adverse_trends": [
                f"{f['name']} {f['trend']} ({f['trend_slope_per_30d']:+}/30d)"
                for f in context.get("adverse_trends", [])
                if f.get("trend_slope_per_30d") is not None
            ],
            "interactions": [i["summary"] for i in context.get("interactions", [])],
            "monitoring_gaps": [g["name"] for g in context.get("monitoring_gaps", [])],
        }

    def _refuse(self, question: str, patient_id: str | None, context: dict[str, Any], trace_id: str,
                reason: str, hits: list[ScoredChunk] | None = None, response: LLMResponse | None = None,
                checks: list[SentenceCheck] | None = None, groundedness: float = 0.0,
                citation_coverage: float = 0.0) -> CopilotAnswer:
        TRACER.metrics.inc("nullius_refusals_total", reason=reason)
        return CopilotAnswer(
            question=question,
            patient_id=patient_id,
            answer=REFUSAL_TEXT,
            refused=True,
            refusal_reason=reason,
            evidence=[h.to_dict() for h in (hits or [])],
            sentence_checks=[c.__dict__ for c in (checks or [])],
            groundedness=groundedness,
            citation_coverage=citation_coverage,
            confidence="refused",
            trace_id=trace_id,
            usage={
                "provider": response.provider if response else self.provider.name,
                "model": response.model if response else self.provider.model,
                "prompt_tokens": response.prompt_tokens if response else 0,
                "completion_tokens": response.completion_tokens if response else 0,
                "latency_ms": round(response.latency_ms, 2) if response else 0.0,
            },
            patient_context=self._summarise_context(context),
        )

    def _audit(self, actor: str, actor_role: str, patient_id: str | None, trace_id: str, detail: str) -> None:
        self.store.audit(
            actor=actor, actor_role=actor_role, action="copilot.ask",
            patient_id=patient_id, trace_id=trace_id, detail=detail,
        )
