from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel

from .agents import total_score, verify_evidence
from .config import Settings
from .graph import build_graph
from .ingestion import IngestionError, extract_zip, parse_document
from .reporting import append_to_csv, build_excel
from .schemas import CRITERIA_WEIGHTS, DocumentEvaluationResult, verdict_for

logger = logging.getLogger(__name__)

ProgressCB = Callable[[int, int, str], None]


class RunSummary(BaseModel):
    run_id: str
    results: list[DocumentEvaluationResult]
    skipped: list[str]
    report_path: str
    csv_path: str


def _assemble(state: dict, doc, elapsed: float) -> DocumentEvaluationResult:
    errors = list(state.get("errors", []))
    fin = state.get("final")
    base = dict(
        file_name=doc.name, file_type=doc.file_type, char_count=doc.char_count,
        truncated=doc.truncated, elapsed_seconds=round(elapsed, 1), errors=errors,
    )
    if fin is None:
        return DocumentEvaluationResult(status="failed", **base)
    reports = [state[k] for k in ("problem_innovation", "technical", "impact_presentation") if state.get(k)]
    verified, checked = verify_evidence(doc.text, reports)
    total = total_score(fin.criteria)
    return DocumentEvaluationResult(
        status="partial" if errors else "completed",
        project_title=fin.project_title,
        total_score=total,
        verdict=verdict_for(total),
        criterion_scores={c.criterion: round(c.score, 1) for c in fin.criteria},
        criterion_justifications={c.criterion: c.justification for c in fin.criteria},
        summary=fin.summary, strengths=fin.strengths, weaknesses=fin.weaknesses,
        recommendations=fin.recommendations,
        evidence_checked=checked, evidence_verified=verified, **base,
    )


def run_evaluation(
    zip_path: Path,
    job_dir: Path,
    settings: Settings,
    llm,
    run_id: Optional[str] = None,
    on_progress: Optional[ProgressCB] = None,
    on_result: Optional[Callable[[DocumentEvaluationResult], None]] = None,
) -> RunSummary:
    """Process every document in the ZIP independently (fresh graph state per document)."""
    run_id = run_id or uuid.uuid4().hex[:12]
    job_dir = Path(job_dir)
    extraction = extract_zip(zip_path, job_dir / "extracted", settings)
    graph = build_graph(llm, settings)
    total = len(extraction.files)
    results: list[DocumentEvaluationResult] = []

    for i, ef in enumerate(extraction.files):
        if on_progress:
            on_progress(i, total, ef.original_name)
        t0 = time.time()
        try:
            doc = parse_document(ef.path, ef.original_name, settings.max_doc_chars)
        except IngestionError as e:
            res = DocumentEvaluationResult(
                file_name=ef.original_name, file_type=ef.path.suffix.lstrip(".").lower(),
                status="failed", errors=[f"Ingestion: {e}"],
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Unexpected parse failure for %s", ef.original_name)
            res = DocumentEvaluationResult(
                file_name=ef.original_name, file_type=ef.path.suffix.lstrip(".").lower(),
                status="failed", errors=[f"Ingestion: unexpected error: {e}"],
            )
        else:
            try:
                state = graph.invoke({"document": doc, "errors": []})
            except Exception as e:  # noqa: BLE001
                logger.exception("Graph failure for %s", ef.original_name)
                state = {"final": None, "errors": [f"Workflow error: {e}"]}
            res = _assemble(state, doc, time.time() - t0)
        append_to_csv(res, run_id, settings.csv_path)
        results.append(res)
        if on_result:
            on_result(res)
        if on_progress:
            on_progress(i + 1, total, ef.original_name)

    for s in extraction.skipped:
        results.append(DocumentEvaluationResult(
            file_name=s.split(":")[0], file_type="", status="skipped", errors=[s]))

    report = build_excel(results, job_dir / f"hackathon_evaluation_{run_id}.xlsx", run_id)
    return RunSummary(
        run_id=run_id, results=results, skipped=extraction.skipped,
        report_path=str(report), csv_path=str(settings.csv_path),
    )
