"""FHIR -> relational ingest over SQLite.

Why SQLite: v1 has one writer and a few thousand rows. Postgres is a one-line
change (see docs/ADR-002) and buys nothing until we need concurrency or
partitioning. Ingest is idempotent by design: every row is keyed on its FHIR
resource id and written with INSERT OR REPLACE, so replaying a bundle is safe.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .observability import TRACER

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY,
    mrn TEXT,
    family TEXT,
    given TEXT,
    gender TEXT,
    birth_date TEXT
);
CREATE TABLE IF NOT EXISTS encounters (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    class TEXT,
    started_at TEXT,
    ended_at TEXT,
    reason TEXT
);
CREATE TABLE IF NOT EXISTS conditions (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    code TEXT,
    display TEXT,
    clinical_status TEXT,
    onset_at TEXT
);
CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    encounter_id TEXT,
    loinc TEXT,
    display TEXT,
    value REAL,
    unit TEXT,
    effective_at TEXT
);
CREATE TABLE IF NOT EXISTS medications (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    rxnorm TEXT,
    display TEXT,
    status TEXT,
    dose TEXT,
    authored_on TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    actor TEXT NOT NULL,
    actor_role TEXT NOT NULL,
    action TEXT NOT NULL,
    patient_id TEXT,
    trace_id TEXT,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_obs_patient_code ON observations(patient_id, loinc, effective_at);
CREATE INDEX IF NOT EXISTS idx_enc_patient ON encounters(patient_id, started_at);
CREATE INDEX IF NOT EXISTS idx_audit_patient ON audit_log(patient_id, at);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------ ingest
    def close(self) -> None:
        """Release the SQLite handle. Safe to call more than once."""
        if getattr(self, "conn", None) is not None:
            self.conn.close()
            self.conn = None  # type: ignore[assignment]

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def ingest_bundles(self, bundles: Iterable[dict[str, Any]]) -> dict[str, int]:
        counts = {"Patient": 0, "Encounter": 0, "Condition": 0, "Observation": 0, "MedicationRequest": 0, "skipped": 0}
        with TRACER.span("ingest.bundles") as span:
            for bundle in bundles:
                for entry in bundle.get("entry", []):
                    resource = entry.get("resource", {})
                    kind = resource.get("resourceType")
                    handler = getattr(self, f"_upsert_{kind.lower()}", None) if kind else None
                    if handler is None:
                        counts["skipped"] += 1
                        continue
                    handler(resource)
                    counts[kind] = counts.get(kind, 0) + 1
            self.conn.commit()
            span.set(**{f"ingest.{k.lower()}": v for k, v in counts.items()})
        for kind, value in counts.items():
            TRACER.metrics.inc("nullius_ingested_resources_total", value, resource=kind)
        return counts

    @staticmethod
    def _ref_id(resource: dict[str, Any], key: str) -> str | None:
        ref = resource.get(key, {}).get("reference")
        return ref.split("/", 1)[1] if ref and "/" in ref else None

    @staticmethod
    def _coding(resource: dict[str, Any], key: str) -> tuple[str | None, str | None]:
        codings = resource.get(key, {}).get("coding") or [{}]
        return codings[0].get("code"), codings[0].get("display") or resource.get(key, {}).get("text")

    def _upsert_patient(self, r: dict[str, Any]) -> None:
        name = (r.get("name") or [{}])[0]
        mrn = (r.get("identifier") or [{}])[0].get("value")
        self.conn.execute(
            "INSERT OR REPLACE INTO patients (id, mrn, family, given, gender, birth_date) VALUES (?,?,?,?,?,?)",
            (r["id"], mrn, name.get("family"), (name.get("given") or [None])[0], r.get("gender"), r.get("birthDate")),
        )

    def _upsert_encounter(self, r: dict[str, Any]) -> None:
        period = r.get("period", {})
        reason = (r.get("reasonCode") or [{}])[0].get("text")
        self.conn.execute(
            "INSERT OR REPLACE INTO encounters (id, patient_id, class, started_at, ended_at, reason) VALUES (?,?,?,?,?,?)",
            (r["id"], self._ref_id(r, "subject"), r.get("class", {}).get("code"), period.get("start"), period.get("end"), reason),
        )

    def _upsert_condition(self, r: dict[str, Any]) -> None:
        code, display = self._coding(r, "code")
        status = ((r.get("clinicalStatus", {}).get("coding") or [{}])[0]).get("code")
        self.conn.execute(
            "INSERT OR REPLACE INTO conditions (id, patient_id, code, display, clinical_status, onset_at) VALUES (?,?,?,?,?,?)",
            (r["id"], self._ref_id(r, "subject"), code, display, status, r.get("onsetDateTime")),
        )

    def _upsert_observation(self, r: dict[str, Any]) -> None:
        code, display = self._coding(r, "code")
        qty = r.get("valueQuantity", {})
        self.conn.execute(
            "INSERT OR REPLACE INTO observations (id, patient_id, encounter_id, loinc, display, value, unit, effective_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (
                r["id"], self._ref_id(r, "subject"), self._ref_id(r, "encounter"), code, display,
                qty.get("value"), qty.get("unit"), r.get("effectiveDateTime"),
            ),
        )

    def _upsert_medicationrequest(self, r: dict[str, Any]) -> None:
        code, display = self._coding(r, "medicationCodeableConcept")
        dose = (r.get("dosageInstruction") or [{}])[0].get("text")
        self.conn.execute(
            "INSERT OR REPLACE INTO medications (id, patient_id, rxnorm, display, status, dose, authored_on) VALUES (?,?,?,?,?,?,?)",
            (r["id"], self._ref_id(r, "subject"), code, display, r.get("status"), dose, r.get("authoredOn")),
        )

    # ------------------------------------------------------------------- reads
    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        return [dict(row) for row in self.conn.execute(sql, params).fetchall()]

    def patients(self) -> list[dict[str, Any]]:
        return self.query(
            """SELECT p.*, 
                      (SELECT COUNT(*) FROM observations o WHERE o.patient_id = p.id) AS observation_count,
                      (SELECT COUNT(*) FROM conditions c WHERE c.patient_id = p.id) AS condition_count,
                      (SELECT MAX(started_at) FROM encounters e WHERE e.patient_id = p.id) AS last_seen
               FROM patients p ORDER BY p.id"""
        )

    def patient(self, patient_id: str) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM patients WHERE id = ?", (patient_id,))
        return rows[0] if rows else None

    def observations(self, patient_id: str, loinc: str | None = None) -> list[dict[str, Any]]:
        if loinc:
            return self.query(
                "SELECT * FROM observations WHERE patient_id = ? AND loinc = ? ORDER BY effective_at",
                (patient_id, loinc),
            )
        return self.query("SELECT * FROM observations WHERE patient_id = ? ORDER BY effective_at", (patient_id,))

    def conditions(self, patient_id: str) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM conditions WHERE patient_id = ? ORDER BY onset_at", (patient_id,))

    def medications(self, patient_id: str) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM medications WHERE patient_id = ? ORDER BY authored_on", (patient_id,))

    def encounters(self, patient_id: str) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM encounters WHERE patient_id = ? ORDER BY started_at", (patient_id,))

    # ------------------------------------------------------------------- audit
    def audit(self, *, actor: str, actor_role: str, action: str, patient_id: str | None = None,
              trace_id: str | None = None, detail: str | None = None) -> None:
        from datetime import datetime, timezone

        self.conn.execute(
            "INSERT INTO audit_log (at, actor, actor_role, action, patient_id, trace_id, detail) VALUES (?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), actor, actor_role, action, patient_id, trace_id, detail),
        )
        self.conn.commit()
        TRACER.metrics.inc("nullius_audit_events_total", action=action)

    def audit_trail(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.query("SELECT * FROM audit_log ORDER BY seq DESC LIMIT ?", (limit,))
