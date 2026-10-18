from hackeval.agents import FABRICATED_CAP, THIN_CAP, UNSUPPORTED_CAP, apply_strict_scoring
from hackeval.config import get_settings
from hackeval.schemas import CRITERIA_WEIGHTS, CriterionScore, EvidenceQuote, verdict_for
from hackeval.service import run_evaluation
from tests.helpers import FakeLLM, make_pdf, make_zip


def _crit(score=8.0):
    return [CriterionScore(criterion=c, score=score, justification="j", evidence=[]) for c in CRITERIA_WEIGHTS]


def _q(c, ok):
    return EvidenceQuote(criterion=c, quote="q", agent="x", verified=ok)


def _scores(out):
    return {c.criterion: c.score for c in out}


def test_unsupported_criterion_capped():
    quotes = [_q(c, True) for c in CRITERIA_WEIGHTS if c != "technical_depth"]
    out, adj = apply_strict_scoring(_crit(), quotes, 5000, 1500)
    assert _scores(out)["technical_depth"] == UNSUPPORTED_CAP and _scores(out)["innovation"] == 8.0
    assert adj == ["technical_depth: 8 -> 4 (capped at 4: no verified evidence)"]
    assert "[strict:" in next(c for c in out if c.criterion == "technical_depth").justification


def test_fabricated_evidence_capped_lower():
    quotes = [_q(c, True) for c in CRITERIA_WEIGHTS if c != "innovation"] + [_q("innovation", False)]
    out, _ = apply_strict_scoring(_crit(), quotes, 5000, 1500)
    assert _scores(out)["innovation"] == FABRICATED_CAP


def test_thin_submission_capped():
    out, adj = apply_strict_scoring(_crit(9.0), [_q(c, True) for c in CRITERIA_WEIGHTS], 400, 1500)
    assert all(c.score == THIN_CAP for c in out) and adj[0].startswith("Thin submission (400 chars")


def test_integrity_penalty_when_most_quotes_fake():
    quotes = [_q(c, True) for c in CRITERIA_WEIGHTS] + [_q(c, False) for c in CRITERIA_WEIGHTS for _ in range(2)]
    out, adj = apply_strict_scoring(_crit(), quotes, 5000, 1500)
    assert all(c.score == 7.0 for c in out) and adj[0].startswith("Evidence integrity: only 6/18")


def test_low_scores_untouched():
    out, adj = apply_strict_scoring(_crit(3.0), [], 5000, 1500)
    assert all(c.score == 3.0 and "[strict" not in c.justification for c in out) and adj == []


def test_verdict_thresholds_tightened():
    assert verdict_for(85) == "Excellent" and verdict_for(84.9) == "Strong"
    assert verdict_for(70) == "Strong" and verdict_for(69.9) == "Promising"
    assert verdict_for(55) == "Promising" and verdict_for(54.9) == "Needs Improvement"


def test_pipeline_fabricated_quotes_and_toggle(tmp_path, settings, monkeypatch):
    z = make_zip(tmp_path / "in.zip", {"a.pdf": make_pdf()})
    llm = FakeLLM(quote="this sentence is not in the deck")
    r = run_evaluation(z, tmp_path / "j1", settings, llm, run_id="s1").results[0]
    # every quote fabricated: cap 3.0 then -1 integrity -> 2.0 per criterion
    assert r.total_score == 20.0 and r.verdict == "Needs Improvement" and len(r.adjustments) == 7
    monkeypatch.setenv("STRICT_SCORING", "false")
    r2 = run_evaluation(z, tmp_path / "j2", get_settings(), llm, run_id="s2").results[0]
    assert r2.total_score == 80.0 and r2.adjustments == []
