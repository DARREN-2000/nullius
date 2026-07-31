"""Claim-level verification beyond token overlap.

Token overlap answers "does this sentence reuse the vocabulary of its evidence?"
That is necessary but not sufficient, and the two ways it fails are both worse
than an ordinary hallucination because the output *looks* well-cited:

  1. **Fabricated numbers.** "Withhold spironolactone when potassium exceeds 4.9
     mmol/L [1]" scores highly against a passage whose real threshold is 6.0.
     Every content word is present in the evidence; only the number is invented,
     and the number is the entire clinical payload. In a decision-support setting
     a wrong threshold or dose is the most dangerous single token that can be
     emitted.

  2. **Inverted polarity.** "Potassium above 6.0 mmol/L is not an emergency [1]"
     shares almost all of its tokens with the passage stating that it is. Overlap
     scores it as well supported; the meaning is reversed.

Both are checked here with deterministic, dependency-free logic, and both are
proven to discriminate by the `numeric-tamper` control arm in the evaluation
harness rather than asserted to work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matches integers and decimals, including ranges written as 3.5-5.0.
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# Citation markers and the ordinals inside them are not clinical claims.
_CITATION = re.compile(r"\[\d+\]")

# Negation and polarity cues. Deliberately small and explicit: a hand-checked list
# beats a clever heuristic that nobody can audit, and this list is testable.
NEGATION_CUES = frozenset(
    {
        "not", "no", "never", "without", "cannot", "none", "neither", "nor",
        "avoid", "contraindicated", "unnecessary", "withhold", "stop", "exclude",
        "unsafe", "inappropriate", "discontinue",
    }
)


def extract_numbers(text: str) -> list[str]:
    """Clinical numbers in a sentence, with citation markers stripped first.

    Values are compared as normalised strings rather than floats so that 5.0 and
    5 match, while 5.0 and 50 do not.
    """
    cleaned = _CITATION.sub(" ", text)
    return [_normalise(match.group()) for match in _NUMBER.finditer(cleaned)]


def _normalise(value: str) -> str:
    try:
        number = float(value)
    except ValueError:
        return value
    return f"{number:.4f}".rstrip("0").rstrip(".") or "0"


@dataclass
class ClaimCheck:
    """Result of the non-overlap checks for a single sentence."""

    numbers: list[str]
    unsupported_numbers: list[str]
    polarity_conflict: bool

    @property
    def ok(self) -> bool:
        return not self.unsupported_numbers and not self.polarity_conflict


def check_claim(sentence: str, cited_text: str) -> ClaimCheck:
    """Verify a sentence's numbers and polarity against the text it cites."""
    evidence_numbers = set(extract_numbers(cited_text))
    numbers = extract_numbers(sentence)
    # Small integers are usually enumeration or duration prose rather than a
    # threshold, but they are still checked: a fabricated "3 months" matters. The
    # only exemption is a number that also appears in the evidence, which is the
    # check itself, so no exemption list is needed.
    unsupported = [n for n in numbers if n not in evidence_numbers]
    return ClaimCheck(
        numbers=numbers,
        unsupported_numbers=unsupported,
        polarity_conflict=_polarity_conflict(sentence, cited_text),
    )


def _cues(text: str) -> set[str]:
    words = re.findall(r"[a-z']+", text.lower())
    return {w for w in words if w in NEGATION_CUES}


_SENTENCE_SPLIT = re.compile(r"(?<=[.;])\s+")


def _closest_evidence_sentence(sentence: str, cited_text: str) -> str:
    """The single source sentence a claim is most plausibly derived from.

    Polarity must be judged against this, not against the whole passage. A
    retrieved chunk is several hundred words and almost always contains a
    negation *somewhere* -- "do not", "avoid", "stop" -- and comparing a claim's
    cues against that pooled set made the difference empty for any negation the
    chunk happened to mention elsewhere. The gate then reported no conflict while
    the claim said the opposite of its own source sentence, which is precisely the
    failure it exists to catch.
    """
    claim_terms = set(re.findall(r"[a-z']+", _CITATION.sub(" ", sentence).lower()))
    best, best_score = "", -1.0
    for candidate in _SENTENCE_SPLIT.split(cited_text):
        terms = set(re.findall(r"[a-z']+", candidate.lower()))
        if not terms:
            continue
        score = len(claim_terms & terms) / len(terms | claim_terms)
        if score > best_score:
            best, best_score = candidate, score
    return best or cited_text


def _polarity_conflict(sentence: str, cited_text: str) -> bool:
    """True when the sentence introduces negation its evidence does not contain.

    Asymmetric on purpose. A claim that negates where the evidence does not is a
    meaning inversion and is blocked. A claim that drops a negation the evidence
    carries is usually legitimate extraction of a positive sub-clause, so it is
    left to the overlap score - blocking it would refuse a large share of correct
    answers, and a gate that fires on correct output gets switched off in
    practice, which makes the system less safe rather than more.

    Compared sentence-to-sentence rather than sentence-to-chunk; see
    `_closest_evidence_sentence` for why the pooled comparison was unsound.
    """
    return bool(_cues(sentence) - _cues(_closest_evidence_sentence(sentence, cited_text)))
