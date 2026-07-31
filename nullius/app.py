"""Application facade: wiring, role-based access control, and the read models
the UI and API consume.

Everything a caller can do goes through `Nullius` so that access control and
audit logging cannot be bypassed by reaching into the store directly. Access
control is enforced here rather than in the HTTP layer because the pipeline,
the tests and the API all share this path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .copilot import Copilot
from .interactions import check_interactions
from .labs import review_patient
from .llm import build_provider
from .observability import TRACER
from .retrieval import Retriever, load_corpus
from .risk import RiskAssessor
from .store import Store
from .timeline import build_timeline, summarise_counts

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "clinician": {"patient.read", "copilot.ask", "labs.read", "audit.read", "imaging.read",
                  "risk.read"},
    "nurse": {"patient.read", "labs.read"},
    "radiologist": {"patient.read", "imaging.read", "audit.read"},
    "researcher": {"cohort.read"},
    "auditor": {"audit.read"},
}


class AccessDenied(PermissionError):
    pass


@dataclass
class Principal:
    user_id: str
    role: str

    def require(self, permission: str) -> None:
        allowed = ROLE_PERMISSIONS.get(self.role, set())
        if permission not in allowed:
            TRACER.metrics.inc("nullius_access_denied_total", role=self.role, permission=permission)
            raise AccessDenied(f"role '{self.role}' lacks permission '{permission}'")


class Nullius:
    def __init__(
        self,
        store: Store,
        retriever: Retriever,
        copilot: Copilot,
        vision: Any = None,
        studies: list[dict[str, Any]] | None = None,
    ) -> None:
        self.store = store
        self.retriever = retriever
        self.copilot = copilot
        self.vision = vision
        self.risk = RiskAssessor(store)
        self._studies = studies or []

    # ------------------------------------------------------------------ reads
    def patients(self, principal: Principal) -> list[dict[str, Any]]:
        principal.require("patient.read")
        return self.store.patients()

    def patient_summary(self, principal: Principal, patient_id: str) -> dict[str, Any]:
        """The clinician's landing view: who, what is wrong now, what changed."""
        principal.require("patient.read")
        with TRACER.span("app.patient_summary", **{"patient.id": patient_id}) as span:
            patient = self.store.patient(patient_id)
            if not patient:
                raise KeyError(patient_id)
            labs = review_patient(self.store, patient_id)
            medications = self.store.medications(patient_id)
            conditions = self.store.conditions(patient_id)
            interactions = check_interactions(medications, conditions=conditions)
            timeline = build_timeline(self.store, patient_id)
            self.store.audit(
                actor=principal.user_id, actor_role=principal.role, action="patient.read",
                patient_id=patient_id, trace_id=span.trace_id, detail="opened patient summary",
            )
            return {
                "patient": patient,
                "conditions": conditions,
                "medications": medications,
                "encounters": self.store.encounters(patient_id),
                "labs": labs,
                "interactions": interactions,
                "timeline": timeline,
                "timeline_counts": summarise_counts(timeline),
                "priority": self._priority(labs, interactions),
            }

    @staticmethod
    def _priority(labs: dict[str, Any], interactions: list[dict[str, Any]]) -> dict[str, Any]:
        """Deterministic triage banner. Not a risk model - a rule the clinician can audit."""
        reasons: list[str] = []
        for finding in labs["critical_values"]:
            reasons.append(f"{finding['name']} {finding['latest_value']} {finding['unit']} is a critical value")
        for interaction in interactions:
            if interaction["severity"] in {"contraindicated", "major"}:
                reasons.append(interaction["summary"])
        for finding in labs["adverse_trends"]:
            reasons.append(f"{finding['name']} {finding['trend']} unfavourably ({finding['trend_slope_per_30d']:+}/30d)")
        level = "urgent" if labs["critical_values"] else ("attention" if reasons else "routine")
        return {"level": level, "reasons": reasons[:6], "reason_count": len(reasons)}

    def ask(self, principal: Principal, question: str, patient_id: str | None = None) -> dict[str, Any]:
        principal.require("copilot.ask")
        answer = self.copilot.ask(
            question, patient_id, actor=principal.user_id, actor_role=principal.role
        )
        return answer.to_dict()

    # ---------------------------------------------------------------- imaging
    def studies(self, principal: Principal, patient_id: str | None = None) -> list[dict[str, Any]]:
        """Catalogue of available imaging studies.

        Ground-truth labels are deliberately withheld here. They belong to the
        evaluation harness; an API that hands out the answer key invites
        benchmarking against itself.
        """
        principal.require("imaging.read")
        rows = [
            {k: v for k, v in study.items() if k not in {"label", "truth_note", "features", "quality"}}
            for study in self._studies
        ]
        if patient_id:
            rows = [row for row in rows if row.get("patient_id") == patient_id]
        return rows

    def study(self, study_id: str) -> dict[str, Any]:
        for study in self._studies:
            if study["study_id"] == study_id:
                return study
        raise KeyError(study_id)

    def classify_image(
        self, principal: Principal, source: Any, patient_id: str | None = None
    ) -> dict[str, Any]:
        principal.require("imaging.read")
        if self.vision is None:
            raise KeyError("no imaging model loaded; run scripts/train_lesion_model.py first")
        result = self.vision.classify(
            source, actor=principal.user_id, actor_role=principal.role, patient_id=patient_id
        )
        return result.to_dict()

    def risk_assessment(self, principal: Principal, patient_id: str) -> dict[str, Any]:
        """90-day CKD-progression score, or a documented refusal to produce one."""
        principal.require("risk.read")
        if not self.store.patient(patient_id):
            raise KeyError(patient_id)
        result = self.risk.assess(
            patient_id, actor=principal.user_id, actor_role=principal.role
        )
        return result.to_dict()

    def model_card(self) -> dict[str, Any]:
        if self.vision is None:
            return {"loaded": False}
        metadata = dict(self.vision.bundle.metadata)
        metadata["loaded"] = True
        metadata["backend"] = self.vision.bundle.backend
        metadata["modelPath"] = str(self.vision.bundle.model_path)
        return metadata

    def audit_trail(self, principal: Principal, limit: int = 50) -> list[dict[str, Any]]:
        principal.require("audit.read")
        return self.store.audit_trail(limit)

    def metrics(self) -> dict[str, Any]:
        return TRACER.metrics.snapshot()


def load_vision(model_dir: str | Path = "models", store: Store | None = None) -> tuple[Any, list[dict[str, Any]]]:
    """Load the imaging model if it has been trained. Absence is not an error.

    The text path must keep working on a checkout where nobody has run the
    trainer yet, so a missing model degrades to "imaging unavailable" rather
    than breaking startup.
    """
    import json

    from .vision import ModelBundle, VisionPipeline

    directory = Path(model_dir)
    try:
        bundle = ModelBundle(directory / "lesion-mlp.onnx")
    except (FileNotFoundError, ValueError):
        return None, []
    catalogue_path = directory / "lesion-features.json"
    catalogue: list[dict[str, Any]] = []
    if catalogue_path.exists():
        catalogue = json.loads(catalogue_path.read_text(encoding="utf-8"))
    return VisionPipeline(bundle, store=store), catalogue


def build_app(
    db_path: str | Path = ":memory:",
    corpus_dir: str | Path = "corpus",
    provider: str = "extractive",
    bundles: list[dict[str, Any]] | None = None,
    model_dir: str | Path = "models",
    nli_judge: bool = False,
) -> Nullius:
    """Compose the system. `bundles=None` generates the synthetic FHIR cohort."""
    from .fhir_gen import generate_all
    from .copilot import Gates

    store = Store(db_path)
    store.ingest_bundles(bundles if bundles is not None else generate_all())
    retriever = Retriever(load_corpus(corpus_dir))
    gates = Gates(nli=True) if nli_judge else Gates()
    judge = build_provider("openai") if nli_judge else None
    copilot = Copilot(store, retriever, build_provider(provider), judge_provider=judge, gates=gates)
    vision, studies = load_vision(model_dir, store=store)
    return Nullius(store, retriever, copilot, vision=vision, studies=studies)
