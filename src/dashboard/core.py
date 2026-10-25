from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import requests

from src.registry import storage

DEFAULT_DB_PATH = Path("data/registry.db")
DEFAULT_API_URL = "http://127.0.0.1:8000"


def resolve_db_path(db_path: str | Path | None = None) -> Path:
    """Resolve registry SQLite DB path from argument, env, or default."""
    if db_path is not None:
        return Path(db_path)
    env_path = os.environ.get("FRESHDOC_DB_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_DB_PATH


def get_available_projects(db_path: Path | str) -> list[str]:
    """Return all repositories with records in the registry."""
    return storage.list_repos(db_path)


def get_available_branches(db_path: Path | str, repo: str) -> list[str]:
    """Return all branches for the selected repository."""
    return storage.list_branches(db_path, repo)


def get_available_commits(
    db_path: Path | str, repo: str, branch: str
) -> list[dict[str, Any]]:
    """Return recorded commits for a repo and branch ordered by sequence."""
    return storage.list_commits(db_path, repo, branch)


def load_freshness_data(
    db_path: Path | str,
    repo: str,
    branch: str,
    commit: str | None = None,
    min_risk: float | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    """Load latest documentation state sorted by urgency and risk."""
    rows = storage.latest(
        db_path=db_path,
        repo=repo,
        branch=branch,
        commit=commit,
        min_risk=min_risk,
    )

    if search:
        s_term = search.strip().lower()
        rows = [
            r
            for r in rows
            if s_term in (r.get("section_id") or "").lower()
            or s_term in (r.get("symbol") or "").lower()
            or s_term in (r.get("doc_path") or "").lower()
        ]

    # Sort descending by urgency, then risk, with None safe fallbacks
    def sort_key(item: dict[str, Any]) -> tuple[float, float]:
        urg = item.get("urgency")
        rsk = item.get("risk")
        return (
            float(urg) if urg is not None else -1.0,
            float(rsk) if rsk is not None else -1.0,
        )

    rows.sort(key=sort_key, reverse=True)
    return rows


def load_section_history(
    db_path: Path | str,
    section_id: str,
    repo: str,
    branch: str,
) -> list[dict[str, Any]]:
    """Load audit event history for a specific section."""
    return storage.history(
        db_path=db_path,
        section_id=section_id,
        repo=repo,
        branch=branch,
    )


def execute_verify(
    repo: str,
    branch: str,
    section_id: str,
    commit: str,
    api_url: str | None = None,
    db_path: Path | str | None = None,
    verified_by: str = "dashboard_user",
) -> dict[str, Any]:
    """Execute section verification either via FastAPI POST /verify or direct DB storage."""
    # First attempt: if api_url is provided, call FastAPI endpoint
    if api_url:
        endpoint = f"{api_url.rstrip('/')}/verify"
        payload = {
            "repo": repo,
            "branch": branch,
            "section_id": section_id,
            "commit": commit,
            "verified_by": verified_by,
        }
        try:
            resp = requests.post(endpoint, json=payload, timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "source": "api",
                    "status": "verified",
                    "record": data.get("record", {}),
                }
        except Exception:
            # Fall back to direct storage when API server is not running
            pass

    # Direct database fallback
    target_db = resolve_db_path(db_path)
    rec = storage.verify(
        db_path=target_db,
        repo=repo,
        branch=branch,
        section_id=section_id,
        commit=commit,
        verified_by=verified_by,
    )
    return {
        "source": "direct_storage",
        "status": "verified",
        "record": rec,
    }


def get_risk_level_info(risk: float | None) -> tuple[str, str]:
    """Return risk label and hex color for badges."""
    if risk is None:
        return ("НЕ ОЦЕНЁН", "#808495")
    if risk >= 0.7:
        return ("ВЫСОКИЙ РИСК", "#FF4B4B")
    if risk >= 0.3:
        return ("СРЕДНИЙ РИСК", "#FFAA00")
    return ("НИЗКИЙ РИСК", "#21C354")
