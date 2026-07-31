"""Guideline retrieval: heading-aware chunking + BM25, with a pluggable scorer.

Why BM25 rather than embeddings: the sandbox has no network, so a hosted
embedding model is unavailable, and BM25 is the correct baseline anyway. Clinical
queries are dense with rare exact terms ("hyperkalaemia", "spironolactone",
"eGFR"), which lexical scoring handles well. `Retriever` takes a `scorer`, so a
vector or hybrid scorer drops in without touching the copilot. Chunk identity is
stable (`doc_id#slug`) because citations are only trustworthy if they are
addressable and reproducible.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Iterable

from .observability import TRACER

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does", "for", "from",
    "has", "have", "how", "if", "in", "into", "is", "it", "its", "may", "of", "on", "or", "should",
    "that", "the", "their", "then", "there", "these", "this", "to", "was", "were", "what", "when",
    "which", "who", "why", "will", "with", "you", "your", "i", "we", "my", "patient", "patients",
}

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-/\.]*")


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    source: str
    section: str
    text: str
    citation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    matched_terms: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"score": round(self.score, 4), "matched_terms": self.matched_terms, **self.chunk.to_dict()}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def load_corpus(corpus_dir: str | Path) -> list[Chunk]:
    """Chunk on `##` headings.

    Heading boundaries beat fixed-size windows here because clinical guidance is
    already written in self-contained recommendation blocks; splitting mid-block
    is how you end up citing a threshold without its qualifying condition.
    """
    chunks: list[Chunk] = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta: dict[str, str] = {}
        body = raw
        if raw.startswith("---"):
            _, front, body = raw.split("---", 2)
            for line in front.strip().splitlines():
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
        doc_id = meta.get("id", path.stem)
        title = meta.get("title", path.stem)
        source = meta.get("source", "local protocol")
        current = "Overview"
        buffer: list[str] = []

        def flush() -> None:
            text = "\n".join(buffer).strip()
            if not text:
                return
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{_slug(current)}",
                    doc_id=doc_id,
                    doc_title=title,
                    source=source,
                    section=current,
                    text=text,
                    citation=f"{title} \u2014 {current} ({source})",
                )
            )

        for line in body.splitlines():
            if line.startswith("## "):
                flush()
                buffer = []
                current = line[3:].strip()
            else:
                buffer.append(line)
        flush()
    return chunks


Scorer = Callable[[list[str], "Retriever"], list[tuple[int, float, list[str]]]]


class Retriever:
    """BM25 index. k1/b are the standard defaults; short chunks make b matter."""

    def __init__(self, chunks: Iterable[Chunk], k1: float = 1.5, b: float = 0.75, scorer: Scorer | None = None) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.scorer = scorer or bm25_scorer
        self.doc_tokens = [tokenize(f"{c.doc_title} {c.section} {c.text}") for c in self.chunks]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.term_freq: list[dict[str, int]] = []
        self.doc_freq: dict[str, int] = {}
        for tokens in self.doc_tokens:
            freq: dict[str, int] = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
            self.term_freq.append(freq)
            for token in freq:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

    def search(self, query: str, top_k: int = 4, min_score: float = 0.8) -> list[ScoredChunk]:
        with TRACER.span("retrieval.search", **{"retrieval.query": query, "retrieval.top_k": top_k}) as span:
            terms = tokenize(query)
            ranked = self.scorer(terms, self)
            ranked.sort(key=lambda item: -item[1])
            hits = [
                ScoredChunk(chunk=self.chunks[idx], score=score, matched_terms=matched)
                for idx, score, matched in ranked[:top_k]
                if score >= min_score
            ]
            span.set(**{
                "retrieval.candidates": len(ranked),
                "retrieval.hits": len(hits),
                "retrieval.top_score": round(hits[0].score, 3) if hits else 0.0,
                "retrieval.chunk_ids": [h.chunk.chunk_id for h in hits],
            })
            TRACER.metrics.observe("nullius_retrieval_hits", len(hits))
            if not hits:
                TRACER.metrics.inc("nullius_retrieval_empty_total")
            return hits


def bm25_scorer(terms: list[str], index: Retriever) -> list[tuple[int, float, list[str]]]:
    n = len(index.chunks)
    results: list[tuple[int, float, list[str]]] = []
    for idx, freq in enumerate(index.term_freq):
        score = 0.0
        matched: list[str] = []
        for term in set(terms):
            tf = freq.get(term, 0)
            if tf == 0:
                continue
            matched.append(term)
            df = index.doc_freq.get(term, 0)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            norm = tf * (index.k1 + 1) / (
                tf + index.k1 * (1 - index.b + index.b * (index.doc_len[idx] / (index.avg_len or 1)))
            )
            score += idf * norm
        if score > 0:
            results.append((idx, score, sorted(matched)))
    return results
