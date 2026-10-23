"""FD-29: deterministic demo seed produces a non-empty registry (no GPU)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest


def test_seed_demo_registry_no_gpu(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git binary required")
    from scripts.seed_demo_registry import build_demo_repo, seed_demo_db
    from src.registry.storage import latest

    repo = build_demo_repo(tmp_path / "demo_repo")
    assert (repo / ".git").exists()
    db = tmp_path / "registry.db"
    summary = seed_demo_db(db, repo, branch="demo")
    # Signature change (C2) flagged by M0, doc fix (C3) closes it with lead.
    assert summary["fixes"] == 1
    assert summary["preceded"] == 1
    assert summary["sections"] == 2
    rows = latest(db, "freshdoc/demo", "demo")
    assert {r["section_id"] for r in rows} == {
        "demo.py::fetch#docstring",
        "demo.py::parse#docstring",
    }
    # Deterministic: rebuild into a second DB gives the same shape.
    db2 = tmp_path / "registry2.db"
    summary2 = seed_demo_db(db2, repo, branch="demo")
    assert summary2 == summary


def test_seed_demo_registry_with_gpgsign_global(tmp_path: Path, monkeypatch) -> None:
    """Point 8: a global commit.gpgsign=true must not break the seed."""
    import shutil

    import pytest

    if shutil.which("git") is None:
        pytest.skip("git binary required")
    from scripts.seed_demo_registry import build_demo_repo, seed_demo_db

    fake_global = tmp_path / "gitconfig"
    fake_global.write_text("[commit]\n\tgpgsign = true\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(fake_global))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    repo = build_demo_repo(tmp_path / "demo_repo_gpg")
    summary = seed_demo_db(tmp_path / "gpg.db", repo, branch="demo")
    assert summary["fixes"] == 1
