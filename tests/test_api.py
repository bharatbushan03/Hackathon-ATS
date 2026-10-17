import time

from fastapi.testclient import TestClient

from hackeval import api
from tests.helpers import FakeLLM, make_pdf, make_zip


def test_api_flow(tmp_path, settings, monkeypatch):
    monkeypatch.setattr(api, "build_llms", lambda s: FakeLLM())
    c = TestClient(api.app)
    z = make_zip(tmp_path / "in.zip", {"a.pdf": make_pdf()})
    jid = c.post("/jobs", files={"file": ("in.zip", open(z, "rb"), "application/zip")}).json()["job_id"]
    for _ in range(100):
        j = c.get(f"/jobs/{jid}").json()
        if j["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert j["status"] == "completed" and j["results"][0]["total_score"] == 80.0
    r = c.get(f"/jobs/{jid}/report")
    assert r.status_code == 200 and r.content[:2] == b"PK"
    assert c.get("/evaluation.csv").status_code == 200
    assert c.post("/jobs", files={"file": ("x.txt", b"x", "text/plain")}).status_code == 400
    assert c.get("/jobs/nope").status_code == 404
