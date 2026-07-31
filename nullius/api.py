"""HTTP API on the standard library.

Why stdlib and not FastAPI: the whole repo must run with `python3 -m nullius.api`
and no install step, which keeps CI and reviewers unblocked. The routing table
below maps one-to-one onto FastAPI path operations, and `docs/ADR-004` records
the swap. Auth is a demo header pair (X-User / X-Role) resolved into the same
`Principal` the rest of the system uses - no separate code path, so RBAC and
audit logging apply identically.
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from .app import AccessDenied, Nullius, Principal, build_app
from .observability import TRACER

ROUTES: list[tuple[str, re.Pattern[str], str]] = []


def route(method: str, pattern: str, handler_name: str) -> None:
    ROUTES.append((method, re.compile(f"^{pattern}$"), handler_name))


route("GET", r"/health", "health")
route("GET", r"/metrics", "metrics")
route("GET", r"/traces", "traces")
route("GET", r"/audit", "audit")
route("GET", r"/patients", "patients")
route("GET", r"/patients/([\w\-]+)", "patient_summary")
route("GET", r"/patients/([\w\-]+)/timeline", "timeline")
route("GET", r"/patients/([\w\-]+)/labs", "labs")
route("GET", r"/patients/([\w\-]+)/risk", "risk")
route("POST", r"/copilot/ask", "ask")
route("GET", r"/studies", "studies")
route("GET", r"/studies/([\w\-]+)", "study_detail")
route("GET", r"/studies/([\w\-]+)/preview\.png", "study_preview")
route("GET", r"/vision/model", "vision_model")
route("POST", r"/vision/classify", "vision_classify")


class Handler(BaseHTTPRequestHandler):
    app: Nullius
    server_version = "Nullius/1.0"

    # ------------------------------------------------------------------ plumbing
    def log_message(self, fmt: str, *args: Any) -> None:  # quieter test output
        return

    def _principal(self) -> Principal:
        return Principal(
            user_id=self.headers.get("X-User", "anonymous"),
            role=self.headers.get("X-Role", "clinician"),
        )

    def _send(self, status: int, payload: Any, content_type: str = "application/json") -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-User, X-Role")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode() or "{}")

    def _dispatch(self, method: str) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/health"
        for verb, pattern, handler_name in ROUTES:
            if verb != method:
                continue
            match = pattern.match(path)
            if not match:
                continue
            with TRACER.span("http.request", **{"http.method": method, "http.route": pattern.pattern}) as span:
                try:
                    getattr(self, handler_name)(*match.groups())
                    span.set(**{"http.status_code": 200})
                except AccessDenied as exc:
                    span.set(**{"http.status_code": 403})
                    self._send(403, {"error": "forbidden", "detail": str(exc)})
                except KeyError as exc:
                    span.set(**{"http.status_code": 404})
                    self._send(404, {"error": "not_found", "detail": str(exc)})
                except Exception as exc:  # pragma: no cover
                    span.set(**{"http.status_code": 500})
                    self._send(500, {"error": "internal", "detail": str(exc)})
            return
        self._send(404, {"error": "no_such_route", "path": path})

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-User, X-Role")
        self.end_headers()

    # ------------------------------------------------------------------ handlers
    def health(self) -> None:
        self._send(200, {"status": "ok", "patients": len(self.app.store.patients()), "chunks": len(self.app.retriever.chunks)})

    def metrics(self) -> None:
        self._send(200, self.app.metrics().get("counters") and self.app.metrics() or self.app.metrics())

    def traces(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        limit = int((query.get("limit") or ["20"])[0])
        self._send(200, {"traces": TRACER.traces()[:limit]})

    def audit(self) -> None:
        self._send(200, {"events": self.app.audit_trail(self._principal())})

    def patients(self) -> None:
        self._send(200, {"patients": self.app.patients(self._principal())})

    def patient_summary(self, patient_id: str) -> None:
        self._send(200, self.app.patient_summary(self._principal(), patient_id))

    def timeline(self, patient_id: str) -> None:
        summary = self.app.patient_summary(self._principal(), patient_id)
        self._send(200, {"patient_id": patient_id, "timeline": summary["timeline"]})

    def labs(self, patient_id: str) -> None:
        summary = self.app.patient_summary(self._principal(), patient_id)
        self._send(200, summary["labs"])

    def risk(self, patient_id: str) -> None:
        # A refusal is a 200 with served=false, not an error: "we decline to
        # score this patient, and here is exactly why" is a successful answer.
        self._send(200, self.app.risk_assessment(self._principal(), patient_id))

    def ask(self) -> None:
        body = self._body()
        question = body.get("question", "").strip()
        if not question:
            self._send(400, {"error": "question_required"})
            return
        self._send(200, self.app.ask(self._principal(), question, body.get("patient_id")))

    # -------------------------------------------------------------- imaging
    def studies(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        patient_id = (query.get("patient_id") or [None])[0]
        self._send(200, {"studies": self.app.studies(self._principal(), patient_id)})

    def study_detail(self, study_id: str) -> None:
        from .dicom import deidentify, read_file

        principal = self._principal()
        principal.require("imaging.read")
        study = self.app.study(study_id)
        clean, _ = deidentify(read_file(study["path"]))
        self._send(
            200,
            {
                "study": {k: v for k, v in study.items() if k not in {"label", "truth_note"}},
                "header": clean.summary(),
            },
        )

    def study_preview(self, study_id: str) -> None:
        """The preprocessed frame with the segmentation outline, as a PNG.

        Serving the image the model actually saw - not the original - is the
        point: it makes preprocessing bugs visible instead of invisible.
        """
        from .dicom import deidentify, read_file
        from .imaging import preprocess, render

        principal = self._principal()
        principal.require("imaging.read")
        study = self.app.study(study_id)
        pre = preprocess(deidentify(read_file(study["path"]))[0])
        self._send(200, render(pre.image, pre.mask), content_type="image/png")

    def vision_model(self) -> None:
        self._send(200, self.app.model_card())

    def vision_classify(self) -> None:
        body = self._body()
        study_id = body.get("study_id") or body.get("studyId")
        path = body.get("path")
        if study_id:
            path = self.app.study(study_id)["path"]
        if not path:
            self._send(400, {"error": "study_id_or_path_required"})
            return
        self._send(200, self.app.classify_image(self._principal(), path, body.get("patient_id")))


def serve(host: str = "0.0.0.0", port: int = 8080, corpus_dir: str | Path = "corpus",
          db_path: str | Path = "data/nullius.db", provider: str = "extractive",
          nli_judge: bool = False) -> None:
    Handler.app = build_app(db_path=db_path, corpus_dir=corpus_dir, provider=provider, nli_judge=nli_judge)
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Nullius API listening on http://{host}:{port} (provider={provider}, nli_judge={nli_judge})")
    print("  GET  /health  /patients  /patients/pat-001  /patients/pat-001/labs  /metrics  /traces  /audit")
    print("  GET  /patients/pat-001/risk  (clinician role required)")
    print("  POST /copilot/ask  {\"question\": \"...\", \"patient_id\": \"pat-001\"}")
    print("  GET  /studies  /studies/stu-000  /studies/stu-000/preview.png  /vision/model")
    print("  POST /vision/classify  {\"study_id\": \"stu-000\"}")
    server.serve_forever()


if __name__ == "__main__":
    import argparse

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run the Nullius API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--corpus", default=str(root / "corpus"))
    parser.add_argument("--db", default=str(root / "out" / "nullius.db"))
    parser.add_argument("--provider", default="extractive")
    parser.add_argument("--nli", action="store_true", help="Enable OpenAI NLI judge for groundedness checks")
    args = parser.parse_args()
    serve(args.host, args.port, args.corpus, args.db, args.provider, args.nli)
