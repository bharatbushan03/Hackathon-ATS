import os
import re

import pytest
from fastapi.testclient import TestClient

from hackeval import api
from hackeval.agents import evidence_quotes
from hackeval.graphstore import GraphStore, GraphStoreError, _to_plain, open_graph_store, redact_url
from hackeval.schemas import CRITERIA_WEIGHTS, AgentReport, CriterionScore, DocumentEvaluationResult, EvidenceQuote
from hackeval.service import run_evaluation
from tests.helpers import FakeLLM, make_pdf, make_zip


class _Result:
    def __init__(self, header=(), rows=()):
        self.header, self.result_set = list(header), list(rows)


class FakeGraph:
    """Records Cypher statements; answers reads with canned rows. Raises when ``broken``."""

    def __init__(self, broken=False):
        self.writes: list[tuple[str, dict]] = []
        self.reads: list[tuple[str, dict]] = []
        self.broken = broken

    def query(self, q, params=None):
        if self.broken:
            raise ConnectionError("falkordb down")
        self.writes.append((q, params or {}))
        return _Result()

    def ro_query(self, q, params=None):
        if self.broken:
            raise ConnectionError("falkordb down")
        self.reads.append((q, params or {}))
        if "count(n)" in q:
            return _Result([[1, "count(n)"]], [[3]])
        if "syntax" in q.lower():
            raise RuntimeError("Syntax error at offset 0")
        return _Result([[1, "file_name"], [1, "total_score"], [1, "scores"]],
                       [["a.pdf", 80.0, [{"criterion": "innovation", "score": 8.0}]]])


def _result(**kw) -> DocumentEvaluationResult:
    base = dict(
        file_name="a.pdf", file_type="pdf", status="completed", project_title="Demo", total_score=80.0,
        verdict="Excellent", criterion_scores={c: 8.0 for c in CRITERIA_WEIGHTS},
        criterion_justifications={c: "j" for c in CRITERIA_WEIGHTS}, summary="sum",
        strengths=["a"], weaknesses=["b"], recommendations=["c"],
        evidence=[EvidenceQuote(criterion="innovation", quote="rural farmers", agent="problem_innovation",
                                verified=True)],
        evidence_checked=1, evidence_verified=1,
    )
    base.update(kw)
    return DocumentEvaluationResult(**base)


def test_store_result_writes_full_subgraph():
    g = FakeGraph()
    store = GraphStore(g, "t")
    sid = store.store_result("run1", _result())
    assert sid == "run1:a.pdf"
    text = "\n".join(q for q, _ in g.writes)
    for frag in ("MERGE (n:Criterion", "DETACH DELETE", "MERGE (run:Run", "CREATE (run)-[:EVALUATED]->(s)",
                 "MERGE (p:Project", "[:SCORED {", "(e:Evidence {", "[:HAS_STRENGTH]", "[:HAS_WEAKNESS]",
                 "[:HAS_RECOMMENDATION]"):
        assert frag in text, frag
    params = [p for _, p in g.writes]
    scores = next(p["scores"] for p in params if "scores" in p)
    assert {s["criterion"] for s in scores} == set(CRITERIA_WEIGHTS)
    ev = next(p["evidence"] for p in params if "evidence" in p)
    assert ev == [{"criterion": "innovation", "quote": "rural farmers", "agent": "problem_innovation",
                   "verified": True}]
    # schema is created once per store
    n_schema = sum("MERGE (n:Criterion" in q for q, _ in g.writes)
    store.store_result("run1", _result(file_name="b.pdf"))
    assert sum("MERGE (n:Criterion" in q for q, _ in g.writes) == n_schema == 1


def test_store_result_skips_optional_parts_for_failed_docs():
    g = FakeGraph()
    GraphStore(g, "t").store_result("run1", _result(status="failed", project_title="", total_score=None, verdict="",
                                                     criterion_scores={}, criterion_justifications={}, evidence=[],
                                                     strengths=[], weaknesses=[], recommendations=[], errors=["boom"]))
    text = "\n".join(q for q, _ in g.writes)
    assert "CREATE (run)-[:EVALUATED]->(s)" in text
    for frag in ("MERGE (p:Project", "[:SCORED", "(e:Evidence {", "HAS_"):
        assert frag not in text, frag


def test_reads_and_query_guardrails():
    store = GraphStore(FakeGraph(), "t", "falkor://user:secret@db:6379")
    assert store.stats()["url"] == "falkor://user:***@db:6379"
    assert store.stats()["nodes"]["submission"] == 3
    rows = store.leaderboard(run_id="run1", limit=5)
    assert rows[0]["criterion_scores"] == {"innovation": 8.0} and "scores" not in rows[0]
    assert store.ro_query("MATCH (n) RETURN n")[0]["file_name"] == "a.pdf"
    with pytest.raises(GraphStoreError):
        store.ro_query("   ")
    with pytest.raises(GraphStoreError):
        store.ro_query("x" * 5000)
    with pytest.raises(GraphStoreError, match="Syntax"):
        store.ro_query("syntax")


def test_helpers():
    assert redact_url("redis://:pw@host:1234/0") == "redis://:***@host:1234/0"
    assert redact_url("falkor://localhost:6379") == "falkor://localhost:6379"

    class Node:
        labels, properties = ["Submission"], {"id": "x", "n": [1, 2]}

    assert _to_plain([Node(), {"k": 1}, None]) == [{"labels": ["Submission"], "id": "x", "n": [1, 2]}, {"k": 1}, None]


def test_evidence_quotes_per_agent_verification():
    rep = AgentReport(criteria=[CriterionScore(criterion="innovation", score=7, justification="j",
                                               evidence=["rural farmers predict", "not in the document at all",
                                                         "tiny"])],
                      strengths=[], weaknesses=[], summary="s")
    quotes = evidence_quotes("Our platform helps rural farmers predict crop disease.",
                             {"problem_innovation": rep, "technical": None})
    assert [(q.verified, q.agent) for q in quotes] == [(True, "problem_innovation"), (False, "problem_innovation")]


def test_pipeline_writes_graph_and_isolates_failures(tmp_path, settings):
    z = make_zip(tmp_path / "in.zip", {"a.pdf": make_pdf(), "skip.txt": b"x"})
    g = FakeGraph()
    s = run_evaluation(z, tmp_path / "job", settings, FakeLLM(), run_id="g1", graph_store=GraphStore(g, "t"))
    ids = [p["id"] for q, p in g.writes if "CREATE (run)-[:EVALUATED]" in q]
    assert ids == ["g1:a.pdf", "g1:skip.txt"]
    assert s.results[0].evidence and all(e.verified for e in s.results[0].evidence)
    assert len(s.results[0].evidence) == s.results[0].evidence_checked == 6

    broken = run_evaluation(z, tmp_path / "job2", settings, FakeLLM(), run_id="g2",
                            graph_store=GraphStore(FakeGraph(broken=True), "t"))
    r = broken.results[0]
    assert r.status == "completed" and r.total_score == 80.0
    assert any(e.startswith("Graph store:") for e in r.errors)


def test_open_graph_store_disabled_or_unreachable(settings, monkeypatch):
    assert open_graph_store(settings) is None
    monkeypatch.setenv("FALKORDB_URL", "falkor://127.0.0.1:1")
    from hackeval.config import get_settings
    with pytest.raises(GraphStoreError, match="Cannot connect"):
        open_graph_store(get_settings())


def test_graph_api(settings, monkeypatch):
    c = TestClient(api.app)
    st = c.get("/graph/status").json()
    assert st == {"enabled": False, "connected": False, "graph": "hackeval", "url": ""}
    assert c.get("/graph/leaderboard").status_code == 503
    assert c.post("/graph/query", json={"cypher": "RETURN 1"}).status_code == 503

    monkeypatch.setenv("FALKORDB_URL", "falkor://db:6379")
    store = GraphStore(FakeGraph(), "hackeval", "falkor://db:6379")
    monkeypatch.setattr(api, "open_graph_store", lambda s: store)
    st = c.get("/graph/status").json()
    assert st["connected"] and st["nodes"]["run"] == 3 and "Leaderboard (all runs)" in st["sample_queries"]
    lb = c.get("/graph/leaderboard", params={"run_id": "run1", "limit": 10}).json()
    assert lb["count"] == 1 and lb["results"][0]["file_name"] == "a.pdf"
    q = c.post("/graph/query", json={"cypher": "MATCH (n) RETURN n"}).json()
    assert q["count"] == 1
    assert c.post("/graph/query", json={"cypher": "syntax"}).status_code == 400
    assert c.get("/graph/leaderboard", params={"limit": 0}).status_code == 422


# ---------- real server (opt-in): FALKORDB_TEST_URL=falkor://localhost:6379 pytest ----------

@pytest.mark.skipif(not os.getenv("FALKORDB_TEST_URL"), reason="FALKORDB_TEST_URL not set")
def test_against_real_falkordb():
    from falkordb import FalkorDB

    from hackeval.graphstore import SAMPLE_QUERIES

    name = "hackeval_test_" + re.sub(r"\W", "", os.urandom(4).hex())
    db = FalkorDB.from_url(os.environ["FALKORDB_TEST_URL"])
    graph = db.select_graph(name)
    try:
        store = GraphStore(graph, name, os.environ["FALKORDB_TEST_URL"])
        store.store_result("r1", _result())
        store.store_result("r1", _result(file_name="b.pdf", total_score=55.0, verdict="Promising",
                                         evidence=[EvidenceQuote(criterion="innovation", quote="x y z made up",
                                                                 agent="technical", verified=False)]))
        store.store_result("r1", _result())  # idempotent re-write
        store.store_result("r2", _result(status="failed", project_title="", total_score=None, verdict="",
                                         criterion_scores={}, criterion_justifications={}, evidence=[],
                                         strengths=[], weaknesses=[], recommendations=[], errors=["boom"]))
        st = store.stats()["nodes"]
        assert st == {"run": 2, "submission": 3, "project": 1, "criterion": 6, "evidence": 2, "insight": 6}
        lb = store.leaderboard()
        assert [r["file_name"] for r in lb] == ["a.pdf", "b.pdf"]
        assert lb[0]["criterion_scores"] == {c: 8.0 for c in CRITERIA_WEIGHTS}
        assert store.leaderboard(run_id="r2") == []
        sub = store.submission("r1:b.pdf")
        assert sub["run_id"] == "r1" and sub["submission"]["total_score"] == 55.0
        assert len(sub["scores"]) == 6 and sub["evidence"][0]["verified"] is False
        assert {i["kind"] for i in sub["insights"]} == {"strength", "weakness", "recommendation"}
        assert store.submission("nope") is None
        for q in SAMPLE_QUERIES.values():
            store.ro_query(q)
        with pytest.raises(GraphStoreError):
            store.ro_query("CREATE (n:Evil) RETURN n")
    finally:
        graph.delete()
