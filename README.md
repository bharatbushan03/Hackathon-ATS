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
- Optional [FalkorDB](https://www.falkordb.com/) graph store: submissions, scores, evidence quotes and insights are written as a graph and can be queried with Cypher (see below).
- Output: ranked Excel report (Leaderboard, Details, Issues, Rubric).
- Failures are isolated: bad file -> `failed`; one agent down -> `partial`; unsupported file -> `skipped`.

## Rubric (weights)
Problem Clarity 15, Innovation 20, Technical Depth 25, Completeness 10, Impact & Viability 20, Presentation Quality 10.
Verdict: >=85 Excellent, >=70 Strong, >=55 Promising, else Needs Improvement.

### Strict scoring (default on)
Scores are evidence-gated in code after the LLM pass, so a deck cannot score well on claims alone:
- a criterion with **no verbatim quote found in the document** is capped at 4/10; if quotes were cited but none exist in the document it is capped at 3/10;
- if fewer than half of all cited quotes are verified, every criterion loses 1 point (evidence-integrity penalty);
- submissions shorter than `STRICT_MIN_CHARS` (default 1500 characters of extracted text) are capped at 5/10 on every criterion.
Every cap/penalty is listed under *Strict adjustments* in the UI, the Excel Details sheet and the graph. Set `STRICT_SCORING=false` to disable.

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

To evaluate a Drive folder, the person submitting the link enters their own Google Drive API key in the UI (create it in Google Cloud with the Drive API enabled) and shares the folder as “Anyone with the link” (Viewer). The key is used only for that job and is never stored; there is no server-side Drive key. Drive folder enumeration uses the [Drive API's public-folder listing](https://developers.google.com/workspace/drive/api/guides/search-files). ZIP uploads need no key. Private Drive folders require an authenticated connection and are not supported yet.

`LLM_MODEL` is the default deployment name. Optional per-agent overrides (same key + URL): `LLM_MODEL_PROBLEM_INNOVATION`, `LLM_MODEL_TECHNICAL`, `LLM_MODEL_IMPACT_PRESENTATION`, `LLM_MODEL_FINAL`. Each name must match a deployment in that project and support structured output / tool calling.

Other providers: `openai`, `anthropic`, `google` (needs `langchain-google-genai`), `ollama` (needs `langchain-ollama`).

## FalkorDB graph store (optional)
Set `FALKORDB_URL` (e.g. `falkor://localhost:6379`, `falkor://user:pass@host:6379`, `falkors://…` for TLS) and optionally `FALKORDB_GRAPH` (default `hackeval`). Every evaluated document is then also written to FalkorDB:

```
(:Run {id})-[:EVALUATED]->(:Submission {id, file_name, project_title, total_score, verdict, status, …})
(:Submission)-[:OF_PROJECT]->(:Project {name})
(:Submission)-[:SCORED {score, justification}]->(:Criterion {id, label, weight})
(:Submission)-[:CITES]->(:Evidence {quote, verified, agent})-[:SUPPORTS]->(:Criterion)
(:Submission)-[:HAS_STRENGTH|HAS_WEAKNESS|HAS_RECOMMENDATION]->(:Insight {kind, text})
```

- `Submission.id` is `<run_id>:<file_name>`; re-evaluating the same file in the same run replaces its sub-graph.
- Graph writes never fail an evaluation: if FalkorDB is down the run completes and the document gets a `Graph store: …` error note.
- The Streamlit sidebar shows the graph status and a **Graph Explorer** panel runs read-only Cypher (with sample queries: leaderboard across runs, average per criterion, weakest criterion, unverified evidence, most common weaknesses, projects seen in several runs).

Quick start with Docker: `docker run -p 6379:6379 -p 3000:3000 falkordb/falkordb` (FalkorDB Browser at http://localhost:3000), then set `FALKORDB_URL=falkor://localhost:6379`. `docker compose up --build` starts FalkorDB and the evaluator together.

Example queries:
```cypher
// Cross-run leaderboard
MATCH (r:Run)-[:EVALUATED]->(s:Submission) WHERE s.total_score IS NOT NULL
RETURN r.id, s.file_name, s.total_score, s.verdict ORDER BY s.total_score DESC LIMIT 10

// Evidence quotes the judges cited but that were not found verbatim in the document
MATCH (s:Submission)-[:CITES]->(e:Evidence {verified: false})-[:SUPPORTS]->(c:Criterion)
RETURN s.file_name, c.label, e.agent, e.quote
```

## API
- `POST /jobs` (multipart `file`=zip) -> `{job_id}`
- `POST /jobs/drive` (`{"folder_url": "https://drive.google.com/drive/folders/...", "api_key": "<uploader's Drive API key>"}`, both required) -> `{job_id}`
- `GET /jobs/{id}` -> status, progress, per-document results
- `GET /jobs/{id}/report` -> Excel file
- `GET /evaluation.csv` -> cumulative CSV
- `GET /health`
- `GET /graph/status` -> FalkorDB connection state, node counts, sample queries
- `GET /graph/leaderboard?run_id=&limit=` -> ranked submissions from the graph (all runs by default)
- `GET /graph/submissions/{run_id}:{file_name}` -> one submission with scores, evidence and insights
- `POST /graph/query` (`{"cypher": "...", "params": {}}`) -> read-only Cypher (`GRAPH.RO_QUERY`; writes are rejected)

## Tests
```bash
pytest -q
```
Uses a fake LLM. No API key needed. Graph-store tests use a fake graph; to also run them against a live server: `FALKORDB_TEST_URL=falkor://localhost:6379 pytest -q`.

## Limits and known gaps
- Scanned/image-only PDFs have no extractable text and are marked failed (no OCR).
- Supported Drive files are PDFs and PowerPoint `.pptx` files; Google Slides are exported as `.pptx`. Legacy `.ppt` files are not supported.
- Images and charts inside slides are counted, not interpreted.
- Long documents are truncated (70% head / 30% tail) at `MAX_DOC_CHARS`.
- Job state is in memory. Restarting the API loses job status (files and CSV persist).
- `evaluation.csv` is cumulative across runs. The Excel report covers one run only.
