import csv

from hackeval.service import run_evaluation
from tests.helpers import FakeLLM, make_pdf, make_pptx, make_zip


def _zip(tmp_path):
    return make_zip(tmp_path / "in.zip", {"a.pdf": make_pdf(), "b.pptx": make_pptx(),
                                          "broken.pdf": b"garbage", "skip.txt": b"x"})


def test_full_pipeline(tmp_path, settings):
    llm = FakeLLM(score=8.0)
    s = run_evaluation(_zip(tmp_path), tmp_path / "job", settings, llm, run_id="t1")
    by = {r.file_name: r for r in s.results}
    assert by["a.pdf"].status == "completed" and by["a.pdf"].total_score == 80.0
    assert by["a.pdf"].verdict == "Strong" and by["a.pdf"].adjustments == []
    assert by["a.pdf"].evidence_verified == by["a.pdf"].evidence_checked == 6
    assert by["broken.pdf"].status == "failed"
    assert by["skip.txt"].status == "skipped"
    # 3 specialists + final per valid doc, no cross-document leakage
    assert sorted(llm.calls).count("final") == 2 and len(llm.calls) == 8
    rows = list(csv.DictReader(open(settings.csv_path, encoding="utf-8-sig")))
    assert len(rows) == 3
    assert (tmp_path / "job" / "hackathon_evaluation_t1.xlsx").exists()


def test_partial_when_one_agent_fails(tmp_path, settings):
    z = make_zip(tmp_path / "in.zip", {"a.pdf": make_pdf()})
    s = run_evaluation(z, tmp_path / "job", settings, FakeLLM(fail_agents={"technical"}), run_id="t2")
    r = s.results[0]
    # technical specialist failed -> technical_depth/completeness have no verified evidence -> capped at 4
    assert r.status == "partial" and r.total_score == 66.0
    assert r.criterion_scores["technical_depth"] == 4.0 and len(r.adjustments) == 2
    assert any("Technical" in e for e in r.errors)


def test_failed_when_all_fail(tmp_path, settings):
    z = make_zip(tmp_path / "in.zip", {"a.pdf": make_pdf()})
    llm = FakeLLM(fail_agents={"technical", "problem_innovation", "impact_presentation", "final"})
    r = run_evaluation(z, tmp_path / "job", settings, llm, run_id="t3").results[0]
    assert r.status == "failed" and r.total_score is None
