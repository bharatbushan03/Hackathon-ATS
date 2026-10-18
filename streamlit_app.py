from __future__ import annotations

import os
import time

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
CRITERIA = ["problem_clarity", "innovation", "technical_depth", "completeness",
            "impact_viability", "presentation_quality"]

st.set_page_config(page_title="AI Evaluator", layout="wide")
st.title("AI Evaluator")
st.markdown(
    "Upload a ZIP or provide a Google Drive folder link containing hackathon submissions (`.pptx` and `.pdf`). "
    "Each document is scored independently by three specialist agents "
    "(Problem & Innovation, Technical, Impact & Presentation). A fourth agent moderates their "
    "outputs into a final evaluation. Totals are computed in code from a weighted rubric. "
    "Download the ranked Excel report when processing finishes. With FalkorDB configured, every "
    "submission, score, evidence quote and insight is also stored as a graph you can query in Cypher."
)


def graph_status() -> dict:
    try:
        return requests.get(f"{API_URL}/graph/status", timeout=5).json()
    except Exception as e:  # noqa: BLE001
        return {"enabled": False, "connected": False, "error": str(e)}


with st.sidebar:
    st.subheader("Backend")
    try:
        h = requests.get(f"{API_URL}/health", timeout=3).json()
        models = h.get("models") or {}
        extra = ", ".join(sorted({m for m in models.values() if m and m != h.get("model")}))
        st.success(f"Connected: {h['provider']} / {h['model']}" + (f" (+ {extra})" if extra else ""))
    except Exception:
        st.error(f"API unreachable at {API_URL}")
    st.subheader("Graph store (FalkorDB)")
    gs = graph_status()
    if not gs.get("enabled"):
        st.caption("Disabled. Set FALKORDB_URL on the API to store evaluations as a graph.")
    elif gs.get("connected"):
        n = gs.get("nodes", {})
        st.success(f"Connected: {gs.get('url')} / graph `{gs.get('graph')}`")
        st.caption(f"{n.get('run', 0)} runs · {n.get('submission', 0)} submissions · "
                   f"{n.get('evidence', 0)} evidence quotes · {n.get('insight', 0)} insights")
    else:
        st.error(f"FalkorDB unreachable: {gs.get('error', 'unknown error')}")

for k, v in {"job_id": None, "report": None, "final": None}.items():
    st.session_state.setdefault(k, v)

zip_tab, drive_tab = st.tabs(["Upload ZIP", "Google Drive folder"])
with zip_tab:
    upload = st.file_uploader("ZIP file with PDFs and PPTX files", type=["zip"])
    start_zip = st.button("Evaluate ZIP", type="primary", disabled=upload is None)
with drive_tab:
    drive_url = st.text_input("Google Drive folder URL", placeholder="https://drive.google.com/drive/folders/…")
    st.caption("The folder must be shared as ‘Anyone with the link’. The backend also needs GOOGLE_DRIVE_API_KEY.")
    start_drive = st.button("Evaluate Drive folder", type="primary", disabled=not drive_url.strip())


def results_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"File": r["file_name"], "Project": r["project_title"],
               "Total": r["total_score"], "Verdict": r["verdict"], "Status": r["status"]}
        row.update({c.replace("_", " ").title(): r["criterion_scores"].get(c) for c in CRITERIA})
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Total", ascending=False, na_position="last").reset_index(drop=True)
    return df


def show_results(job: dict) -> None:
    results = job["results"]
    if not results:
        return
    st.subheader("Results")
    st.dataframe(results_table(results), use_container_width=True, hide_index=True)
    for r in sorted(results, key=lambda x: -(x["total_score"] or -1)):
        label = f"{r['file_name']} — " + (
            f"{r['total_score']}/100 ({r['verdict']})" if r["total_score"] is not None else r["status"].upper())
        with st.expander(label):
            if r["summary"]:
                st.write(r["summary"])
                c1, c2, c3 = st.columns(3)
                c1.markdown("**Strengths**\n" + "\n".join(f"- {x}" for x in r["strengths"]))
                c2.markdown("**Weaknesses**\n" + "\n".join(f"- {x}" for x in r["weaknesses"]))
                c3.markdown("**Recommendations**\n" + "\n".join(f"- {x}" for x in r["recommendations"]))
                st.markdown("**Score justifications**")
                for c, t in r["criterion_justifications"].items():
                    st.markdown(f"- `{c}` {r['criterion_scores'].get(c)}/10: {t}")
                if r["evidence_checked"]:
                    st.caption(f"Evidence quotes verified in document: {r['evidence_verified']}/{r['evidence_checked']}")
            for e in r["errors"]:
                st.error(e)


if start_zip or start_drive:
    st.session_state.update(report=None, final=None)
    try:
        if start_zip and upload is not None:
            resp = requests.post(
                f"{API_URL}/jobs",
                files={"file": (upload.name, upload.getvalue(), "application/zip")},
                timeout=120,
            )
        else:
            resp = requests.post(f"{API_URL}/jobs/drive", json={"folder_url": drive_url.strip()}, timeout=30)
        resp.raise_for_status()
        st.session_state.job_id = resp.json()["job_id"]
    except requests.HTTPError as e:
        st.error(f"Could not start evaluation: {e.response.text}")
        st.stop()
    except Exception as e:
        st.error(f"Cannot reach API: {e}")
        st.stop()

    bar, status, live = st.progress(0.0), st.empty(), st.empty()
    while True:
        job = requests.get(f"{API_URL}/jobs/{st.session_state.job_id}", timeout=30).json()
        total = job["total"] or 1
        bar.progress(min(job["done"] / total, 1.0))
        status.info(f"{job['status'].upper()} — {job['done']}/{job['total']} documents"
                    + (f" — processing: {job['current']}" if job["current"] else ""))
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(1.5)
    if job["status"] == "failed":
        status.error(f"Job failed: {job['error']}")
    else:
        status.success(f"Done. {len(job['results'])} item(s) processed.")
        st.session_state.final = job
        st.session_state.report = requests.get(f"{API_URL}/jobs/{job['job_id']}/report", timeout=60).content

if st.session_state.final:
    show_results(st.session_state.final)
    if st.session_state.report:
        st.download_button(
            "Download Excel report", st.session_state.report,
            file_name=f"hackathon_evaluation_{st.session_state.final['job_id']}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    if st.session_state.final["skipped"]:
        with st.expander("Skipped files"):
            for s in st.session_state.final["skipped"]:
                st.write(s)

if gs.get("connected"):
    st.divider()
    st.subheader("Graph Explorer")
    st.caption(
        "Evaluations live in FalkorDB as `(:Run)-[:EVALUATED]->(:Submission)-[:SCORED]->(:Criterion)`, "
        "with `(:Evidence)` quotes (verified flag) and `(:Insight)` strengths / weaknesses / recommendations. "
        "Queries run read-only."
    )
    samples = gs.get("sample_queries") or {}
    pick = st.selectbox("Sample query", ["Custom"] + list(samples))
    default = samples.get(pick, st.session_state.get("cypher", ""))
    cypher = st.text_area("Cypher", value=default, height=120, key=f"cypher_{pick}")
    if st.button("Run query", disabled=not cypher.strip()):
        try:
            resp = requests.post(f"{API_URL}/graph/query", json={"cypher": cypher}, timeout=60)
            if resp.status_code != 200:
                st.error(resp.json().get("detail", resp.text))
            else:
                rows = resp.json()["results"]
                st.caption(f"{len(rows)} row(s)")
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"Cannot reach API: {e}")
