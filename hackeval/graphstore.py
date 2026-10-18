"""FalkorDB graph store: every evaluated submission becomes a small sub-graph.

    (:Run {id})-[:EVALUATED]->(:Submission {id, file_name, total_score, verdict, ...})
    (:Submission)-[:OF_PROJECT]->(:Project {name})
    (:Submission)-[:SCORED {score, justification}]->(:Criterion {id, label, weight})
    (:Submission)-[:CITES]->(:Evidence {quote, verified, agent})-[:SUPPORTS]->(:Criterion)
    (:Submission)-[:HAS_STRENGTH|HAS_WEAKNESS|HAS_RECOMMENDATION]->(:Insight {kind, text})

The store is optional: ``open_graph_store`` returns ``None`` when ``FALKORDB_URL`` is unset,
and every write is isolated by the caller so a graph outage never fails an evaluation.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

from .config import Settings
from .schemas import CRITERIA_LABELS, CRITERIA_WEIGHTS, DocumentEvaluationResult

logger = logging.getLogger(__name__)

INSIGHT_RELS = {"strength": "HAS_STRENGTH", "weakness": "HAS_WEAKNESS", "recommendation": "HAS_RECOMMENDATION"}
MAX_QUERY_CHARS = 4000

SAMPLE_QUERIES: dict[str, str] = {
    "Leaderboard (all runs)": (
        "MATCH (r:Run)-[:EVALUATED]->(s:Submission) WHERE s.total_score IS NOT NULL "
        "RETURN r.id AS run, s.file_name AS file, s.project_title AS project, s.total_score AS total, "
        "s.verdict AS verdict ORDER BY total DESC LIMIT 25"
    ),
    "Average score per criterion": (
        "MATCH (:Submission)-[sc:SCORED]->(c:Criterion) "
        "RETURN c.label AS criterion, c.weight AS weight, round(avg(sc.score) * 10) / 10 AS avg_score, count(sc) AS n "
        "ORDER BY avg_score DESC"
    ),
    "Weakest criterion per submission": (
        "MATCH (s:Submission)-[sc:SCORED]->(c:Criterion) WITH s, c, sc ORDER BY sc.score ASC "
        "WITH s, collect(c.label)[0] AS weakest, collect(sc.score)[0] AS score "
        "RETURN s.file_name AS file, s.project_title AS project, weakest, score ORDER BY score ASC LIMIT 25"
    ),
    "Unverified evidence quotes": (
        "MATCH (s:Submission)-[:CITES]->(e:Evidence)-[:SUPPORTS]->(c:Criterion) WHERE e.verified = false "
        "RETURN s.file_name AS file, c.label AS criterion, e.agent AS agent, e.quote AS quote LIMIT 50"
    ),
    "Most common weaknesses": (
        "MATCH (:Submission)-[:HAS_WEAKNESS]->(i:Insight) "
        "RETURN i.text AS weakness, count(*) AS submissions ORDER BY submissions DESC LIMIT 25"
    ),
    "Projects evaluated in more than one run": (
        "MATCH (p:Project)<-[:OF_PROJECT]-(s:Submission)<-[:EVALUATED]-(r:Run) "
        "WITH p, collect(DISTINCT r.id) AS runs, collect(s.total_score) AS scores WHERE size(runs) > 1 "
        "RETURN p.name AS project, runs, scores ORDER BY size(runs) DESC"
    ),
}


class GraphStoreError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def redact_url(url: str) -> str:
    """Hide credentials so the URL can be shown in /health-style endpoints."""
    try:
        u = urlsplit(url)
    except ValueError:
        return "<invalid url>"
    if not u.hostname:
        return url
    host = u.hostname + (f":{u.port}" if u.port else "")
    netloc = f"{u.username}:***@{host}" if u.username or u.password else host
    return urlunsplit((u.scheme, netloc, u.path, "", ""))


def _to_plain(v: Any) -> Any:
    """Convert falkordb Node/Edge/Path objects to JSON-friendly values."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_to_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _to_plain(x) for k, x in v.items()}
    labels = getattr(v, "labels", None)
    props = getattr(v, "properties", None)
    if props is not None:
        out: dict[str, Any] = {}
        if labels:
            out["labels"] = list(labels)
        rel = getattr(v, "relation", None)
        if rel:
            out["type"] = rel
        out.update(_to_plain(props))
        return out
    return str(v)


class GraphStore:
    """Thin wrapper over a FalkorDB graph. ``graph`` needs ``query`` and ``ro_query``."""

    def __init__(self, graph, name: str, url: str = ""):
        self.graph, self.name, self.url = graph, name, url
        self._schema_ready = False
        self._lock = threading.Lock()

    # ---------- setup ----------

    def ensure_schema(self) -> None:
        with self._lock:
            if self._schema_ready:
                return
            for label, prop in (("Run", "id"), ("Submission", "id"), ("Criterion", "id"), ("Project", "name")):
                try:
                    self.graph.query(f"CREATE INDEX FOR (n:{label}) ON (n.{prop})")
                except Exception as e:  # noqa: BLE001 - index already exists
                    logger.debug("Index %s.%s: %s", label, prop, e)
            self.graph.query(
                "UNWIND $criteria AS c MERGE (n:Criterion {id: c.id}) SET n.label = c.label, n.weight = c.weight",
                {"criteria": [{"id": c, "label": CRITERIA_LABELS[c], "weight": w}
                              for c, w in CRITERIA_WEIGHTS.items()]},
            )
            self._schema_ready = True

    # ---------- writes ----------

    def store_result(self, run_id: str, r: DocumentEvaluationResult) -> str:
        """Upsert one submission (and its scores/evidence/insights). Returns the Submission id."""
        self.ensure_schema()
        sid = f"{run_id}:{r.file_name}"
        self.graph.query(
            "MATCH (s:Submission {id: $id}) "
            "OPTIONAL MATCH (s)-[:CITES]->(e:Evidence) "
            "OPTIONAL MATCH (s)-[]->(i:Insight) "
            "DETACH DELETE s, e, i",
            {"id": sid},
        )
        self.graph.query(
            "MERGE (run:Run {id: $run_id}) ON CREATE SET run.started_at = $now "
            "CREATE (s:Submission {id: $id, file_name: $file_name, file_type: $file_type, status: $status, "
            "project_title: $project_title, total_score: $total_score, verdict: $verdict, summary: $summary, "
            "char_count: $char_count, truncated: $truncated, elapsed_seconds: $elapsed_seconds, "
            "evidence_checked: $evidence_checked, evidence_verified: $evidence_verified, errors: $errors, "
            "evaluated_at: $now}) "
            "CREATE (run)-[:EVALUATED]->(s)",
            {
                "run_id": run_id, "id": sid, "now": _now(), "file_name": r.file_name, "file_type": r.file_type,
                "status": r.status, "project_title": r.project_title, "total_score": r.total_score,
                "verdict": r.verdict, "summary": r.summary, "char_count": r.char_count, "truncated": r.truncated,
                "elapsed_seconds": r.elapsed_seconds, "evidence_checked": r.evidence_checked,
                "evidence_verified": r.evidence_verified, "errors": list(r.errors),
            },
        )
        if r.project_title and r.project_title.lower() != "unknown":
            self.graph.query(
                "MATCH (s:Submission {id: $id}) MERGE (p:Project {name: $name}) CREATE (s)-[:OF_PROJECT]->(p)",
                {"id": sid, "name": r.project_title.strip()},
            )
        scores = [
            {"criterion": c, "score": float(v), "justification": r.criterion_justifications.get(c, "")}
            for c, v in r.criterion_scores.items() if c in CRITERIA_WEIGHTS
        ]
        if scores:
            self.graph.query(
                "MATCH (s:Submission {id: $id}) UNWIND $scores AS sc MATCH (c:Criterion {id: sc.criterion}) "
                "CREATE (s)-[:SCORED {score: sc.score, justification: sc.justification}]->(c)",
                {"id": sid, "scores": scores},
            )
        evidence = [e.model_dump() for e in r.evidence if e.criterion in CRITERIA_WEIGHTS]
        if evidence:
            self.graph.query(
                "MATCH (s:Submission {id: $id}) UNWIND $evidence AS ev MATCH (c:Criterion {id: ev.criterion}) "
                "CREATE (s)-[:CITES]->(e:Evidence {quote: ev.quote, verified: ev.verified, agent: ev.agent})"
                "-[:SUPPORTS]->(c)",
                {"id": sid, "evidence": evidence},
            )
        insights = (("strength", r.strengths), ("weakness", r.weaknesses), ("recommendation", r.recommendations))
        for kind, items in insights:
            texts = [t.strip() for t in items if t and t.strip()]
            if texts:
                self.graph.query(
                    f"MATCH (s:Submission {{id: $id}}) UNWIND $texts AS t "
                    f"CREATE (s)-[:{INSIGHT_RELS[kind]}]->(:Insight {{kind: $kind, text: t}})",
                    {"id": sid, "texts": texts, "kind": kind},
                )
        return sid

    # ---------- reads ----------

    def _rows(self, result) -> list[dict[str, Any]]:
        cols = [h[1] if isinstance(h, (list, tuple)) else str(h) for h in (result.header or [])]
        return [dict(zip(cols, (_to_plain(v) for v in row))) for row in result.result_set]

    def ro_query(self, cypher: str, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Run a read-only Cypher query (FalkorDB rejects writes via GRAPH.RO_QUERY)."""
        cypher = (cypher or "").strip()
        if not cypher:
            raise GraphStoreError("Empty query.")
        if len(cypher) > MAX_QUERY_CHARS:
            raise GraphStoreError(f"Query exceeds {MAX_QUERY_CHARS} characters.")
        self.ensure_schema()  # RO queries fail on a graph key that does not exist yet
        try:
            return self._rows(self.graph.ro_query(cypher, params or {}))
        except Exception as e:  # noqa: BLE001 - surface the server's message
            raise GraphStoreError(str(e)) from e

    def stats(self) -> dict[str, Any]:
        self.ensure_schema()
        counts = {}
        for label in ("Run", "Submission", "Project", "Criterion", "Evidence", "Insight"):
            res = self.graph.ro_query(f"MATCH (n:{label}) RETURN count(n)")
            counts[label.lower()] = int(res.result_set[0][0]) if res.result_set else 0
        return {"graph": self.name, "url": redact_url(self.url), "nodes": counts}

    def leaderboard(self, run_id: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        self.ensure_schema()
        where = "WHERE s.total_score IS NOT NULL" + (" AND r.id = $run_id" if run_id else "")
        q = (
            "MATCH (r:Run)-[:EVALUATED]->(s:Submission) " + where + " "
            "OPTIONAL MATCH (s)-[sc:SCORED]->(c:Criterion) "
            "WITH r, s, collect({criterion: c.id, score: sc.score}) AS scores "
            "RETURN r.id AS run_id, s.id AS submission_id, s.file_name AS file_name, s.project_title AS project_title, "
            "s.total_score AS total_score, s.verdict AS verdict, s.status AS status, scores "
            "ORDER BY s.total_score DESC, s.file_name ASC LIMIT $limit"
        )
        rows = self._rows(self.graph.ro_query(q, {"run_id": run_id or "", "limit": max(1, min(int(limit), 500))}))
        for row in rows:
            row["criterion_scores"] = {
                d["criterion"]: d["score"] for d in row.pop("scores", []) if isinstance(d, dict) and d.get("criterion")
            }
        return rows

    def submission(self, submission_id: str) -> Optional[dict[str, Any]]:
        self.ensure_schema()
        rows = self._rows(self.graph.ro_query(
            "MATCH (r:Run)-[:EVALUATED]->(s:Submission {id: $id}) "
            "OPTIONAL MATCH (s)-[sc:SCORED]->(c:Criterion) "
            "WITH r, s, collect({criterion: c.id, score: sc.score, justification: sc.justification}) AS scores "
            "OPTIONAL MATCH (s)-[:CITES]->(e:Evidence)-[:SUPPORTS]->(ec:Criterion) "
            "WITH r, s, scores, "
            "collect({criterion: ec.id, quote: e.quote, verified: e.verified, agent: e.agent}) AS evidence "
            "OPTIONAL MATCH (s)-[]->(i:Insight) "
            "RETURN r.id AS run_id, s AS submission, scores, evidence, "
            "collect({kind: i.kind, text: i.text}) AS insights",
            {"id": submission_id},
        ))
        if not rows:
            return None
        row = rows[0]
        row["scores"] = [d for d in row["scores"] if d.get("criterion")]
        row["evidence"] = [d for d in row["evidence"] if d.get("quote")]
        row["insights"] = [d for d in row["insights"] if d.get("text")]
        return row


# ---------- factory ----------

_STORES: dict[tuple[str, str], GraphStore] = {}
_STORES_LOCK = threading.Lock()


def open_graph_store(settings: Settings) -> Optional[GraphStore]:
    """Return a cached GraphStore for the configured FalkorDB, or None when disabled.

    Raises GraphStoreError if FALKORDB_URL is set but the server cannot be reached.
    """
    if not settings.falkordb_url:
        return None
    key = (settings.falkordb_url, settings.falkordb_graph)
    with _STORES_LOCK:
        store = _STORES.get(key)
        if store is None:
            try:
                from falkordb import FalkorDB
            except ImportError as e:  # pragma: no cover
                raise GraphStoreError("Install the 'falkordb' package to enable the graph store.") from e
            try:
                db = FalkorDB.from_url(settings.falkordb_url, socket_connect_timeout=5, socket_timeout=30)
                db.connection.ping()
                graph = db.select_graph(settings.falkordb_graph)
            except Exception as e:  # noqa: BLE001
                raise GraphStoreError(f"Cannot connect to FalkorDB at {redact_url(settings.falkordb_url)}: {e}") from e
            store = _STORES[key] = GraphStore(graph, settings.falkordb_graph, settings.falkordb_url)
    return store
