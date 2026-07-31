"""Minimal OpenTelemetry-shaped tracing + metrics with zero dependencies.

Why hand-rolled: the sandbox has no network access, and the point of this module
is to prove the *shape* of AI observability (traces, spans, attributes, token
and latency metrics, per-span quality attributes) rather than to depend on a
vendor SDK. `Tracer.export_otlp_like()` emits the same span fields an OTLP
exporter would carry, so swapping in the real SDK is a mechanical change.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


def _new_id(n: int = 16) -> str:
    return uuid.uuid4().hex[:n]


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    start_unix_nano: int
    end_unix_nano: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "UNSET"

    @property
    def duration_ms(self) -> float:
        if self.end_unix_nano is None:
            return 0.0
        return (self.end_unix_nano - self.start_unix_nano) / 1_000_000

    def set(self, **attrs: Any) -> "Span":
        self.attributes.update(attrs)
        return self

    def event(self, name: str, **attrs: Any) -> "Span":
        self.events.append(
            {"name": name, "time_unix_nano": time.time_ns(), "attributes": attrs}
        )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_span_id,
            "startTimeUnixNano": self.start_unix_nano,
            "endTimeUnixNano": self.end_unix_nano,
            "durationMs": round(self.duration_ms, 3),
            "status": {"code": self.status},
            "attributes": self.attributes,
            "events": self.events,
        }


class Metrics:
    """Counters and histograms with Prometheus-style text exposition."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, float] = {}
        self.histograms: dict[str, list[float]] = {}

    def inc(self, name: str, value: float = 1.0, **labels: Any) -> None:
        key = _label_key(name, labels)
        with self._lock:
            self.counters[key] = self.counters.get(key, 0.0) + value

    def observe(self, name: str, value: float, **labels: Any) -> None:
        key = _label_key(name, labels)
        with self._lock:
            self.histograms.setdefault(key, []).append(value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            hist = {}
            for key, values in self.histograms.items():
                ordered = sorted(values)
                hist[key] = {
                    "count": len(ordered),
                    "mean": round(sum(ordered) / len(ordered), 3),
                    "p50": round(_quantile(ordered, 0.50), 3),
                    "p95": round(_quantile(ordered, 0.95), 3),
                    "max": round(ordered[-1], 3),
                }
            return {"counters": dict(self.counters), "histograms": hist}

    def prometheus(self) -> str:
        lines: list[str] = []
        snap = self.snapshot()
        for key, value in sorted(snap["counters"].items()):
            lines.append(f"{key} {value}")
        for key, stats in sorted(snap["histograms"].items()):
            base, _, labels = key.partition("{")
            suffix = ("{" + labels) if labels else ""
            for stat, value in stats.items():
                lines.append(f"{base}_{stat}{suffix} {value}")
        return "\n".join(lines) + "\n"


def _label_key(name: str, labels: dict[str, Any]) -> str:
    if not labels:
        return name
    inner = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{name}{{{inner}}}"


def _quantile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


class Tracer:
    def __init__(self, service_name: str = "nullius") -> None:
        self.service_name = service_name
        self.spans: list[Span] = []
        self.metrics = Metrics()
        self._local = threading.local()

    @property
    def _stack(self) -> list[Span]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        parent = self._stack[-1] if self._stack else None
        span = Span(
            name=name,
            trace_id=parent.trace_id if parent else _new_id(32),
            span_id=_new_id(),
            parent_span_id=parent.span_id if parent else None,
            start_unix_nano=time.time_ns(),
            attributes={"service.name": self.service_name, **attributes},
        )
        self._stack.append(span)
        try:
            yield span
            span.status = "OK"
        except Exception as exc:  # pragma: no cover - defensive
            span.status = "ERROR"
            span.set(**{"exception.type": type(exc).__name__, "exception.message": str(exc)})
            self.metrics.inc("nullius_span_errors_total", span=name)
            raise
        finally:
            span.end_unix_nano = time.time_ns()
            self._stack.pop()
            self.spans.append(span)
            self.metrics.inc("nullius_spans_total", span=name)
            self.metrics.observe("nullius_span_duration_ms", span.duration_ms, span=name)

    def export_otlp_like(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for span in self.spans:
                fh.write(json.dumps(span.to_dict()) + "\n")
        return out

    def traces(self) -> list[dict[str, Any]]:
        """Group spans into traces, root first, for the observability UI."""
        grouped: dict[str, list[Span]] = {}
        for span in self.spans:
            grouped.setdefault(span.trace_id, []).append(span)
        result = []
        for trace_id, spans in grouped.items():
            spans = sorted(spans, key=lambda s: s.start_unix_nano)
            root = next((s for s in spans if s.parent_span_id is None), spans[0])
            result.append(
                {
                    "traceId": trace_id,
                    "root": root.name,
                    "durationMs": round(root.duration_ms, 2),
                    "status": root.status,
                    "spanCount": len(spans),
                    "spans": [s.to_dict() for s in spans],
                }
            )
        return sorted(result, key=lambda t: -t["durationMs"])


TRACER = Tracer()
