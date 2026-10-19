from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from src.api.app import create_app
from src.registry.storage import init_db, insert_batch


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Fixture providing a FastAPI TestClient configured with an isolated test DB."""
    test_db = tmp_path / "api_test.db"
    init_db(test_db)
    app = create_app(db_path=test_db)
    return TestClient(app)


def test_docs_and_openapi_accessible(client: TestClient) -> None:
    """Check acceptance criteria: /docs is accessible (HTTP 200)."""
    resp_docs = client.get("/docs")
    assert resp_docs.status_code == 200
    assert "swagger" in resp_docs.text.lower() or "html" in resp_docs.text.lower()

    resp_openapi = client.get("/openapi.json")
    assert resp_openapi.status_code == 200
    data = resp_openapi.json()
    assert data["info"]["title"] == "FreshDoc Registry API"
    assert "/status" in data["paths"]
    assert "/sections/history" in data["paths"]
    assert "/verify" in data["paths"]


def test_health(client: TestClient) -> None:
    """Check health endpoint."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_status_empty_db(client: TestClient) -> None:
    """Check /status on empty database returns empty items list."""
    resp = client.get("/status", params={"repo": "owner/repo", "branch": "main"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["repo"] == "owner/repo"
    assert data["branch"] == "main"
    assert data["commit"] is None
    assert data["items"] == []


def test_status_with_data_matches_r12_schema(tmp_path: Path) -> None:
    """Check that /status response format matches R12 example exactly."""
    test_db = tmp_path / "r12_test.db"
    init_db(test_db)

    # Insert sample row matching R12 example
    insert_batch(
        test_db,
        [
            {
                "repo": "owner/repo",
                "branch": "main",
                "commit_sha": "abc123",
                "section_id": "pkg/mod.py::fetch#docstring",
                "symbol": "fetch",
                "risk": 0.82,
                "urgency": 0.79,
                "impact": 0.4,
                "drift_type": "param_renamed",
                "evidence": {"diff": "-def fetch(url…)\n+def fetch(endpoint…)"},
                "verified_commit": "9f2c1",
                "doc_path": "pkg/mod.py",
            }
        ],
    )

    app = create_app(db_path=test_db)
    c = TestClient(app)

    resp = c.get(
        "/status",
        params={"repo": "owner/repo", "branch": "main", "commit": "abc123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["repo"] == "owner/repo"
    assert data["branch"] == "main"
    assert data["commit"] == "abc123"
    assert len(data["items"]) == 1

    item = data["items"][0]
    assert item["section_id"] == "pkg/mod.py::fetch#docstring"
    assert item["symbol"] == "fetch"
    assert item["risk"] == 0.82
    assert item["urgency"] == 0.79
    assert item["impact"] == 0.4
    assert item["drift_type"] == "param_renamed"
    assert item["evidence"] == {"diff": "-def fetch(url…)\n+def fetch(endpoint…)"}
    assert item["verified_commit"] == "9f2c1"


def test_verify_with_query_params(client: TestClient) -> None:
    """Check POST /verify via query parameters per R12."""
    resp = client.post(
        "/verify",
        params={
            "repo": "owner/repo",
            "branch": "main",
            "section_id": "pkg/mod.py::fetch#docstring",
            "commit": "c_verified",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "verified"
    assert data["record"]["section_id"] == "pkg/mod.py::fetch#docstring"
    assert data["record"]["risk"] == 0.0
    assert data["record"]["verified_commit"] == "c_verified"

    # Status should now reflect verified state
    status_resp = client.get(
        "/status", params={"repo": "owner/repo", "branch": "main"}
    )
    assert status_resp.status_code == 200
    items = status_resp.json()["items"]
    assert len(items) == 1
    assert items[0]["risk"] == 0.0


def test_verify_with_json_body(client: TestClient) -> None:
    """Check POST /verify via JSON body."""
    body = {
        "repo": "owner/repo",
        "branch": "main",
        "section_id": "pkg/mod.py::other#docstring",
        "commit": "c_verified_body",
        "verified_by": "qa_team",
    }
    resp = client.post("/verify", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "verified"
    assert data["record"]["section_id"] == "pkg/mod.py::other#docstring"
    assert data["record"]["risk"] == 0.0


def test_verify_missing_parameters_returns_400(client: TestClient) -> None:
    """Check POST /verify fails with 400 when required fields are missing."""
    resp = client.post("/verify", params={"repo": "owner/repo"})
    assert resp.status_code == 400
    assert "Missing required parameter" in resp.json()["detail"]


def test_sections_history(client: TestClient) -> None:
    """Check GET /sections/history returns chronological history."""
    sec_id = "pkg/mod.py::track#docstring"

    # Add two verification events
    client.post(
        "/verify",
        params={
            "repo": "owner/repo",
            "branch": "main",
            "section_id": sec_id,
            "commit": "c1",
        },
    )
    client.post(
        "/verify",
        params={
            "repo": "owner/repo",
            "branch": "main",
            "section_id": sec_id,
            "commit": "c2",
        },
    )

    resp = client.get(
        "/sections/history",
        params={"section_id": sec_id, "repo": "owner/repo", "branch": "main"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["section_id"] == sec_id
    assert len(data["items"]) == 2
    assert data["items"][0]["verified_commit"] == "c1"
    assert data["items"][1]["verified_commit"] == "c2"


def test_status_exposes_p_stale(tmp_path: Path) -> None:
    """Check that p_stale stored in the registry is returned by /status."""
    from src.registry.storage import insert

    test_db = tmp_path / "p_stale_test.db"
    init_db(test_db)
    insert(
        test_db,
        {
            "repo": "owner/repo",
            "branch": "main",
            "commit_sha": "abc123",
            "section_id": "pkg/mod.py::fetch#docstring",
            "symbol": "fetch",
            "p_stale": 0.91,
            "risk": 0.82,
            "doc_path": "pkg/mod.py",
        },
    )

    app = create_app(db_path=test_db)
    c = TestClient(app)
    resp = c.get("/status", params={"repo": "owner/repo", "branch": "main"})
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["p_stale"] == 0.91
