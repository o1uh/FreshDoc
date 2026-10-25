from __future__ import annotations

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.dashboard.core import (
    execute_verify,
    get_available_branches,
    get_available_commits,
    get_available_projects,
    get_risk_level_info,
    load_freshness_data,
    load_section_history,
)
from src.registry.storage import init_db, insert_batch


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    """Fixture providing populated SQLite registry DB with two branches."""
    db = tmp_path / "test_dashboard.db"
    init_db(db)

    # Insert records into two distinct branches with different states
    records = [
        # Branch 'main'
        {
            "repo": "owner/repo",
            "branch": "main",
            "commit_sha": "c101",
            "doc_path": "pkg/mod_a.py",
            "section_id": "pkg/mod_a.py::func_a#docstring",
            "symbol": "func_a",
            "p_stale": 0.8,
            "drift_type": "param_renamed",
            "risk": 0.85,
            "urgency": 1.25,
            "impact": 0.5,
            "evidence": {"diff": "-def func_a(old)\n+def func_a(new)"},
            "event": "risk_flagged",
            "computed_at": "2026-02-01T10:00:00Z",
        },
        {
            "repo": "owner/repo",
            "branch": "main",
            "commit_sha": "c101",
            "doc_path": "pkg/mod_b.py",
            "section_id": "pkg/mod_b.py::func_b#docstring",
            "symbol": "func_b",
            "p_stale": 0.1,
            "drift_type": None,
            "risk": 0.15,
            "urgency": 0.2,
            "impact": 0.3,
            "evidence": None,
            "event": "doc_updated",
            "computed_at": "2026-02-01T10:00:00Z",
        },
        # Branch 'feature' - has different sections and different risk
        {
            "repo": "owner/repo",
            "branch": "feature",
            "commit_sha": "c201",
            "doc_path": "pkg/mod_a.py",
            "section_id": "pkg/mod_a.py::func_a#docstring",
            "symbol": "func_a",
            "p_stale": 0.0,
            "drift_type": None,
            "risk": 0.0,
            "urgency": 0.0,
            "impact": 0.5,
            "evidence": None,
            "event": "verified",
            "computed_at": "2026-02-02T12:00:00Z",
        },
        {
            "repo": "owner/repo",
            "branch": "feature",
            "commit_sha": "c202",
            "doc_path": "pkg/mod_c.py",
            "section_id": "pkg/mod_c.py::func_c#docstring",
            "symbol": "func_c",
            "p_stale": 0.9,
            "drift_type": "signature_changed",
            "risk": 0.95,
            "urgency": 1.8,
            "impact": 0.8,
            "evidence": {"diff": "-def func_c()\n+def func_c(x, y)"},
            "event": "risk_flagged",
            "computed_at": "2026-02-03T14:00:00Z",
        },
    ]
    insert_batch(db, records)
    return db


def test_get_projects_and_branches(test_db: Path) -> None:
    """Check discovery of projects, branches, and commits."""
    projects = get_available_projects(test_db)
    assert projects == ["owner/repo"]

    branches = get_available_branches(test_db, "owner/repo")
    assert branches == ["feature", "main"]

    commits_main = get_available_commits(test_db, "owner/repo", "main")
    assert len(commits_main) == 1
    assert commits_main[0]["commit_sha"] == "c101"

    commits_feat = get_available_commits(test_db, "owner/repo", "feature")
    assert len(commits_feat) == 2
    assert commits_feat[0]["commit_sha"] == "c202"
    assert commits_feat[1]["commit_sha"] == "c201"


def test_branch_switch_changes_state(test_db: Path) -> None:
    """FD-30 Criterion 1: Switching branches alters the retrieved freshness state."""
    main_sections = load_freshness_data(test_db, repo="owner/repo", branch="main")
    feature_sections = load_freshness_data(test_db, repo="owner/repo", branch="feature")

    # Main has mod_a and mod_b
    main_ids = {s["section_id"] for s in main_sections}
    assert main_ids == {
        "pkg/mod_a.py::func_a#docstring",
        "pkg/mod_b.py::func_b#docstring",
    }

    # Feature has mod_a and mod_c
    feat_ids = {s["section_id"] for s in feature_sections}
    assert feat_ids == {
        "pkg/mod_a.py::func_a#docstring",
        "pkg/mod_c.py::func_c#docstring",
    }

    # func_a has risk 0.85 on main, but 0.0 on feature
    main_func_a = next(s for s in main_sections if s["symbol"] == "func_a")
    feat_func_a = next(s for s in feature_sections if s["symbol"] == "func_a")
    assert main_func_a["risk"] == 0.85
    assert feat_func_a["risk"] == 0.0


def test_verify_resets_risk(test_db: Path) -> None:
    """FD-30 Criterion 2: Verification resets section risk to 0.0."""
    # Before verify: func_a on main has risk 0.85
    initial_main = load_freshness_data(test_db, repo="owner/repo", branch="main")
    item = next(s for s in initial_main if s["symbol"] == "func_a")
    assert item["risk"] == 0.85

    # Execute verify
    result = execute_verify(
        repo="owner/repo",
        branch="main",
        section_id=item["section_id"],
        commit="c101",
        api_url=None,  # test direct DB path
        db_path=test_db,
        verified_by="test_reviewer",
    )
    assert result["status"] == "verified"
    assert result["record"]["risk"] == 0.0
    assert result["record"]["verified_commit"] == "c101"

    # After verify: load_freshness_data returns risk 0.0
    updated_main = load_freshness_data(test_db, repo="owner/repo", branch="main")
    updated_item = next(s for s in updated_main if s["symbol"] == "func_a")
    assert updated_item["risk"] == 0.0
    assert updated_item["urgency"] == 0.0
    assert updated_item["event"] == "verify"
    assert updated_item["verified_commit"] == "c101"


def test_verify_via_api(test_db: Path, monkeypatch) -> None:
    """Test execute_verify calling live FastAPI client."""
    app = create_app(db_path=test_db)
    client = TestClient(app)

    # Monkeypatch requests.post to redirect to TestClient
    def mock_post(url, json=None, timeout=None):
        endpoint = "/" + url.split("/")[-1]
        response = client.post(endpoint, json=json)
        # Wrap TestClient response into a minimal requests-like object
        class MockResponse:
            status_code = response.status_code
            def json(self):
                return response.json()
        return MockResponse()

    import requests
    monkeypatch.setattr(requests, "post", mock_post)

    result = execute_verify(
        repo="owner/repo",
        branch="main",
        section_id="pkg/mod_b.py::func_b#docstring",
        commit="c101",
        api_url="http://mock-api",
        db_path=test_db,
    )
    assert result["source"] == "api"
    assert result["status"] == "verified"
    assert result["record"]["risk"] == 0.0


def test_filters_and_sorting(test_db: Path) -> None:
    """Test min_risk filter, search term, and urgency sorting."""
    # min_risk filter
    filtered = load_freshness_data(test_db, repo="owner/repo", branch="main", min_risk=0.5)
    assert len(filtered) == 1
    assert filtered[0]["symbol"] == "func_a"

    # search query filter
    searched = load_freshness_data(test_db, repo="owner/repo", branch="main", search="func_b")
    assert len(searched) == 1
    assert searched[0]["symbol"] == "func_b"

    # urgency sorting on branch feature: mod_c (urgency 1.8) before mod_a (urgency 0.0)
    feat_sections = load_freshness_data(test_db, repo="owner/repo", branch="feature")
    assert len(feat_sections) == 2
    assert feat_sections[0]["symbol"] == "func_c"
    assert feat_sections[1]["symbol"] == "func_a"


def test_section_history(test_db: Path) -> None:
    """Test retrieving chronological audit trail for a section."""
    history = load_section_history(
        test_db,
        section_id="pkg/mod_a.py::func_a#docstring",
        repo="owner/repo",
        branch="main",
    )
    assert len(history) == 1
    assert history[0]["commit_sha"] == "c101"
    assert history[0]["event"] == "risk_flagged"


def test_get_risk_level_info() -> None:
    """Check risk badge label and color mappings."""
    lbl, col = get_risk_level_info(0.85)
    assert lbl == "ВЫСОКИЙ РИСК"
    assert col == "#FF4B4B"

    lbl, col = get_risk_level_info(0.45)
    assert lbl == "СРЕДНИЙ РИСК"
    assert col == "#FFAA00"

    lbl, col = get_risk_level_info(0.1)
    assert lbl == "НИЗКИЙ РИСК"
    assert col == "#21C354"

    lbl, col = get_risk_level_info(None)
    assert lbl == "НЕ ОЦЕНЁН"
