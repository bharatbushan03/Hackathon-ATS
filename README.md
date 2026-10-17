# AI Hackathon Evaluator

Upload one ZIP of `.pptx` / `.pdf` submissions or provide a publicly shared Google Drive folder URL. Drive folders are searched recursively, and Google Slides are exported to `.pptx`. Each document goes through a LangGraph workflow:

```
START -> [Problem & Innovation | Technical | Impact & Presentation]  (parallel) -> Final Evaluator -> END
```

- Each document runs in a fresh graph state. No cross-project leakage.
- Specialists score 2 criteria each. The final agent moderates all 6.
- Total (/100) and verdict are computed in code from fixed weights, not by the LLM.
- Evidence quotes are checked verbatim against the document text (reported as verified/checked).
- Every result is appended to `data/evaluation.csv` (header created if missing).
- Output: ranked Excel report (Leaderboard, Details, Issues, Rubric).
- Failures are isolated: bad file -> `failed`; one agent down -> `partial`; unsupported file -> `skipped`.

## Rubric (weights)
Problem Clarity 15, Innovation 20, Technical Depth 25, Completeness 10, Impact & Viability 20, Presentation Quality 10.
Verdict: >=80 Excellent, >=65 Strong, >=50 Promising, else Needs Improvement.

## Run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # set AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT (or PROJECT_URL), LLM_MODEL
python run.py                                          # API :8000 + UI :8501
```
Or separately:
```bash
uvicorn hackeval.api:app --port 8000
streamlit run streamlit_app.py
```
UI: http://localhost:8501

## Configuration
See `.env.example`. Default provider is Azure OpenAI. Set one `AZURE_OPENAI_API_KEY` and one project/resource URL (`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_PROJECT_URL`, or `FOUNDRY_PROJECT_ENDPOINT`). That pair is used for every deployment in the project.

To evaluate a Drive folder, enable the Google Drive API in Google Cloud, set `GOOGLE_DRIVE_API_KEY`, and share the folder as “Anyone with the link” (Viewer). Drive folder enumeration uses the [Drive API's public-folder listing](https://developers.google.com/workspace/drive/api/guides/search-files). The API key is only needed for Drive jobs; ZIP uploads work without it. Private Drive folders require an authenticated connection and are not supported yet.

`LLM_MODEL` is the default deployment name. Optional per-agent overrides (same key + URL): `LLM_MODEL_PROBLEM_INNOVATION`, `LLM_MODEL_TECHNICAL`, `LLM_MODEL_IMPACT_PRESENTATION`, `LLM_MODEL_FINAL`. Each name must match a deployment in that project and support structured output / tool calling.

Other providers: `openai`, `anthropic`, `google` (needs `langchain-google-genai`), `ollama` (needs `langchain-ollama`).

## API
- `POST /jobs` (multipart `file`=zip) -> `{job_id}`
- `POST /jobs/drive` (`{"folder_url": "https://drive.google.com/drive/folders/..."}`) -> `{job_id}`
- `GET /jobs/{id}` -> status, progress, per-document results
- `GET /jobs/{id}/report` -> Excel file
- `GET /evaluation.csv` -> cumulative CSV
- `GET /health`

## Tests
```bash
pytest -q
```
Uses a fake LLM. No API key needed.

## Limits and known gaps
- Scanned/image-only PDFs have no extractable text and are marked failed (no OCR).
- Supported Drive files are PDFs and PowerPoint `.pptx` files; Google Slides are exported as `.pptx`. Legacy `.ppt` files are not supported.
- Images and charts inside slides are counted, not interpreted.
- Long documents are truncated (70% head / 30% tail) at `MAX_DOC_CHARS`.
- Job state is in memory. Restarting the API loses job status (files and CSV persist).
- `evaluation.csv` is cumulative across runs. The Excel report covers one run only.
