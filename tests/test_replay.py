"""Tests for FD-29: Replay engine, caching, M0/M2 scorers and metrics (R12)."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.registry.replay import (
    M0Scorer,
    ReplayCache,
    SectionTrackingState,
    make_cache_key,
    replay_project_history,
)
from src.registry.storage import init_db, latest
from src.risk.config import RiskConfig


def test_make_cache_key_deterministic() -> None:
    diff1 = "-def foo(a):\n+def foo(a, b):"
    doc1 = "Does something important with a."
    k1 = make_cache_key(diff1, doc1)
    k2 = make_cache_key(diff1, doc1)
    assert k1 == k2
    assert len(k1) == 40  # SHA1 hexdigest


def test_replay_cache_set_get_and_persistence(tmp_path: Path) -> None:
    cache_file = tmp_path / "cache.json"
    cache = ReplayCache(cache_file)
    assert cache.hits == 0
    assert cache.misses == 0

    diff = "-x = 1\n+x = 2"
    doc = "A valid docstring with plenty of characters."
    assert cache.get("m0", diff, doc) is None
    assert cache.misses == 1

    cache.set("m0", diff, doc, p_stale=0.75, drift_type="param_added")
    val = cache.get("m0", diff, doc)
    assert val is not None
    assert val["p_stale"] == 0.75
    assert val["drift_type"] == "param_added"
    assert cache.hits == 1

    cache.save()
    assert cache_file.exists()

    # Load in new cache instance
    cache2 = ReplayCache(cache_file)
    val2 = cache2.get("m0", diff, doc)
    assert val2 is not None
    assert val2["p_stale"] == 0.75
    assert cache2.hits == 1


def test_m0_scorer_signature_changed_vs_same() -> None:
    scorer = M0Scorer()
    fn_old = ast.parse("def calculate(x, y):\n    '''Docstring.'''\n    return x + y").body[0]
    fn_sig_chg = ast.parse("def calculate(x, y, z=0):\n    '''Docstring.'''\n    return x + y + z").body[0]
    fn_same_sig = ast.parse("def calculate(x, y):\n    '''Docstring.'''\n    return x * y").body[0]

    p1, dt1 = scorer.score("def calculate(x, y):", "diff", "Docstring.", fn_old, fn_sig_chg)
    assert p1 == 1.0
    assert dt1 == "param_added"

    p2, dt2 = scorer.score("def calculate(x, y):", "diff", "Docstring.", fn_old, fn_same_sig)
    assert p2 == 0.0
    assert dt2 == "none"


def test_replay_end_to_end_mocked_git(tmp_path: Path) -> None:
    db_file = tmp_path / "registry.db"
    cache_file = tmp_path / "cache.json"
    cache = ReplayCache(cache_file)

    # 3 commits:
    # C1: baseline function (doc length >= 40 chars, 5 lines)
    # C2: signature change (x -> x, multiplier=2) without doc change -> risk accumulation & flag
    # C3: docstring updated to reflect multiplier -> risk reset, lead time calculated
    c1 = (
        "def compute_total(x):\n"
        "    '''Compute the grand total using standard formula for all items.'''\n"
        "    a = x * 10\n"
        "    b = a + 5\n"
        "    return b\n"
    )
    c2 = (
        "def compute_total(x, multiplier=2):\n"
        "    '''Compute the grand total using standard formula for all items.'''\n"
        "    a = x * multiplier\n"
        "    b = a + 5\n"
        "    return b\n"
    )
    c3 = (
        "def compute_total(x, multiplier=2):\n"
        "    '''Compute the grand total with optional custom multiplier.'''\n"
        "    a = x * multiplier\n"
        "    b = a + 5\n"
        "    return b\n"
    )

    commits = [
        ("sha_c1", "2026-01-01T10:00:00+00:00"),
        ("sha_c2", "2026-01-10T10:00:00+00:00"),
        ("sha_c3", "2026-01-20T10:00:00+00:00"),
    ]

    def mock_get_commits(repo_path, max_commits=300):
        return list(reversed(commits))

    def mock_get_py_files(repo_path, sha):
        if sha == "sha_c1":
            return []  # Initial commit has no changes
        return ["calc/math.py"]

    def mock_show_blob(repo_path, sha, path, before=False):
        if sha == "sha_c2":
            return c1 if before else c2
        if sha == "sha_c3":
            return c2 if before else c3
        return ""

    risk_cfg = RiskConfig(threshold=0.5)

    with patch("src.registry.replay.get_first_parent_commits", side_effect=mock_get_commits), \
         patch("src.registry.replay.get_modified_py_files", side_effect=mock_get_py_files), \
         patch("src.registry.replay.show_blob", side_effect=mock_show_blob):

        metrics, records = replay_project_history(
            repo_path="/fake/repo",
            repo_name="org/calc_repo",
            branch="main",
            max_commits=10,
            scorer=M0Scorer(),
            cache=cache,
            db_path=db_file,
            risk_config=risk_cfg,
        )

    assert metrics.total_fixes == 1
    assert metrics.preceded_fixes == 1
    assert metrics.preceded_fraction == 1.0
    # 10 days between 2026-01-10 and 2026-01-20
    assert metrics.median_lead_time_days == 10.0
    assert metrics.open_flags == 0
    assert metrics.open_flags_older_90d == 0

    # Verify SQLite database
    latest_recs = latest(db_file, "org/calc_repo", "main")
    assert len(latest_recs) == 1
    assert latest_recs[0]["section_id"] == "calc/math.py::compute_total#docstring"
    assert latest_recs[0]["risk"] == 0.0
    assert latest_recs[0]["verified_commit"] == "sha_c3"


def test_replay_cache_prevents_duplicate_calls(tmp_path: Path) -> None:
    cache = ReplayCache(tmp_path / "cache.json")
    spy_scorer = MagicMock()
    spy_scorer.name = "spy"
    spy_scorer.score.return_value = (0.9, "param_added")

    code_before = (
        "def process(item):\n"
        "    '''Process data item and format output properly for user.'''\n"
        "    x = str(item)\n"
        "    y = x.strip()\n"
        "    return y\n"
    )
    code_after = (
        "def process(item, mode='fast'):\n"
        "    '''Process data item and format output properly for user.'''\n"
        "    x = str(item)\n"
        "    y = x.strip()\n"
        "    return y\n"
    )

    commits = [
        ("sha1", "2026-02-01T12:00:00+00:00"),
        ("sha2", "2026-02-02T12:00:00+00:00"),
    ]

    def mock_get_commits(repo_path, max_commits=300):
        return list(reversed(commits))

    def mock_get_py_files(repo_path, sha):
        return ["worker.py"] if sha == "sha2" else []

    def mock_show_blob(repo_path, sha, path, before=False):
        return code_before if before else code_after

    with patch("src.registry.replay.get_first_parent_commits", side_effect=mock_get_commits), \
         patch("src.registry.replay.get_modified_py_files", side_effect=mock_get_py_files), \
         patch("src.registry.replay.show_blob", side_effect=mock_show_blob):

        # First run: miss, scorer invoked once
        replay_project_history(
            repo_path="/fake/repo",
            repo_name="test/app",
            scorer=spy_scorer,
            cache=cache,
        )
        assert spy_scorer.score.call_count == 1
        assert cache.hits == 0
        assert cache.misses == 1

        # Second run: hit, scorer invoked 0 additional times
        replay_project_history(
            repo_path="/fake/repo",
            repo_name="test/app",
            scorer=spy_scorer,
            cache=cache,
        )
        assert spy_scorer.score.call_count == 1  # Still 1, 0 new calls
        assert cache.hits == 1


def test_replay_filters_private_and_short_docs() -> None:
    spy_scorer = MagicMock()
    spy_scorer.name = "spy"
    spy_scorer.score.return_value = (0.9, "param_added")

    # Private function (_inner) and short docstring function (short_doc)
    code_before = (
        "def _inner_helper(x):\n"
        "    '''Helper docstring that is long enough to pass length filter.'''\n"
        "    a = x + 1\n"
        "    b = a * 2\n"
        "    return b\n"
        "def short_doc(x):\n"
        "    '''Too short.'''\n"
        "    a = x + 1\n"
        "    b = a * 2\n"
        "    return b\n"
    )
    code_after = (
        "def _inner_helper(x, y=1):\n"
        "    '''Helper docstring that is long enough to pass length filter.'''\n"
        "    a = x + y\n"
        "    b = a * 2\n"
        "    return b\n"
        "def short_doc(x, y=1):\n"
        "    '''Too short.'''\n"
        "    a = x + y\n"
        "    b = a * 2\n"
        "    return b\n"
    )

    commits = [
        ("sha1", "2026-03-01T12:00:00+00:00"),
        ("sha2", "2026-03-02T12:00:00+00:00"),
    ]

    def mock_get_commits(repo_path, max_commits=300):
        return list(reversed(commits))

    def mock_get_py_files(repo_path, sha):
        return ["utils.py"] if sha == "sha2" else []

    def mock_show_blob(repo_path, sha, path, before=False):
        return code_before if before else code_after

    with patch("src.registry.replay.get_first_parent_commits", side_effect=mock_get_commits), \
         patch("src.registry.replay.get_modified_py_files", side_effect=mock_get_py_files), \
         patch("src.registry.replay.show_blob", side_effect=mock_show_blob):

        metrics, records = replay_project_history(
            repo_path="/fake/repo",
            repo_name="test/filters",
            scorer=spy_scorer,
        )

        assert spy_scorer.score.call_count == 0
        assert len(records) == 0

