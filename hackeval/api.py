from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import get_settings
from .drive import create_folder_zip, parse_folder_url
from .ingestion import IngestionError
from .llm import build_llms
from .service import run_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hackeval.api")

app = FastAPI(title="AI Hackathon Evaluator API")

JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


def _update(job_id: str, **kw) -> None:
    with _LOCK:
        JOBS[job_id].update(kw)


def _worker(job_id: str, zip_path: Path | None, job_dir: Path, folder_url: str | None = None) -> None:
    settings = get_settings()
    _update(job_id, status="running")
    try:
        if folder_url:
            _update(job_id, current="Downloading Google Drive folder")
            zip_path = create_folder_zip(folder_url, job_dir / "drive_upload.zip", settings)
        if zip_path is None:
            raise IngestionError("No submission source was provided.")
        llm = build_llms(settings)

        def prog(done: int, total: int, name: str) -> None:
            _update(job_id, done=done, total=total, current=name)

        def on_result(r) -> None:
            with _LOCK:
                JOBS[job_id]["results"].append(r.model_dump())

        summary = run_evaluation(zip_path, job_dir, settings, llm, run_id=job_id,
                                 on_progress=prog, on_result=on_result)
        _update(job_id, status="completed", report_path=summary.report_path,
                skipped=summary.skipped, current="",
                results=[r.model_dump() for r in summary.results])
    except IngestionError as e:
        _update(job_id, status="failed", error=str(e))
    except Exception as e:  # noqa: BLE001
        logger.exception("Job %s failed", job_id)
        _update(job_id, status="failed", error=f"{type(e).__name__}: {e}")


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {"status": "ok", "provider": s.llm_provider, "model": s.llm_model, "models": s.agent_models}


class DriveJobRequest(BaseModel):
    folder_url: str


@app.post("/jobs")
async def create_job(file: UploadFile = File(...)) -> dict:
    s = get_settings()
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(400, "Upload a .zip file.")
    job_id = uuid.uuid4().hex[:12]
    job_dir = s.data_dir / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    zip_path = job_dir / "upload.zip"
    limit, size = s.max_zip_mb * 1024 * 1024, 0
    with open(zip_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                f.close()
                zip_path.unlink(missing_ok=True)
                raise HTTPException(413, f"ZIP exceeds {s.max_zip_mb} MB limit.")
            f.write(chunk)
    with _LOCK:
        JOBS[job_id] = dict(job_id=job_id, status="queued", done=0, total=0, current="",
                            results=[], skipped=[], error=None, report_path=None)
    threading.Thread(target=_worker, args=(job_id, zip_path, job_dir), daemon=True).start()
    return {"job_id": job_id}


@app.post("/jobs/drive")
def create_drive_job(request: DriveJobRequest) -> dict:
    s = get_settings()
    if not s.google_drive_api_key:
        raise HTTPException(503, "Set GOOGLE_DRIVE_API_KEY to enable Google Drive folder evaluation.")
    try:
        parse_folder_url(request.folder_url)
    except IngestionError as e:
        raise HTTPException(400, str(e)) from e
    job_id = uuid.uuid4().hex[:12]
    job_dir = s.data_dir / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        JOBS[job_id] = dict(job_id=job_id, status="queued", done=0, total=0, current="",
                            results=[], skipped=[], error=None, report_path=None)
    threading.Thread(target=_worker, args=(job_id, None, job_dir, request.folder_url), daemon=True).start()
    return {"job_id": job_id}


def _get(job_id: str) -> dict:
    with _LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job.")
        return dict(job, results=list(job["results"]))


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _get(job_id)
    job.pop("report_path", None)
    return job


@app.get("/jobs/{job_id}/report")
def job_report(job_id: str):
    job = _get(job_id)
    if job["status"] != "completed" or not job["report_path"]:
        raise HTTPException(409, "Report not ready.")
    return FileResponse(
        job["report_path"], filename=Path(job["report_path"]).name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.get("/evaluation.csv")
def full_csv():
    p = get_settings().csv_path
    if not p.exists():
        raise HTTPException(404, "No evaluations recorded yet.")
    return FileResponse(p, filename="evaluation.csv", media_type="text/csv")
