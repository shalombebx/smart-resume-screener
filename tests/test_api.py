"""
tests/test_api.py — smoke tests for the HTTP layer (no OpenAI key needed).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from main import app
from app.services.analyzer import resume_analyzer

client = TestClient(app)


def setup_function(_):
    resume_analyzer.clear_resumes()


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert set(r.json()) == {"status", "version", "openai_ready"}


def test_frontend_served():
    assert client.get("/").status_code == 200


def test_list_resumes_empty():
    assert client.get("/api/resumes/").json() == []


def test_upload_rejects_unsupported_type():
    r = client.post("/api/resumes/upload", files=[("files", ("cv.txt", b"hello", "text/plain"))])
    assert r.status_code == 201
    assert r.json()[0]["status"] == "error"


def test_analysis_without_resumes_is_400():
    r = client.post("/api/analysis/run", json={"job_description": "x" * 60})
    assert r.status_code == 400


def test_analysis_short_jd_is_422():
    assert client.post("/api/analysis/run", json={"job_description": "short"}).status_code == 422


def test_unknown_session_is_404():
    assert client.get("/api/results/nope").status_code == 404
    assert client.get("/api/results/nope/export").status_code == 404


def test_delete_unknown_resume_is_404():
    assert client.delete("/api/resumes/does-not-exist").status_code == 404
