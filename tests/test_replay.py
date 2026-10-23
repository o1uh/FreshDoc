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
    EVIDENCE_DIFF_LIMIT,
    EVIDENCE_DOC_LIMIT,
    M0Scorer,
    ReplayCache,
    SectionTrackingState,
    _load_fan_in_map,
    _nested_qualnames,
    gate_drift,
    make_cache_key,
    replay_project_history,
)
from src.registry.storage import init_db, latest, replace_branch_records
from src.risk.config import RiskConfig


def test_make_cache_key_deterministic() -> None:
    sig1 = "def foo(a):"
    diff1 = "-def foo(a):\n+def foo(a, b):"
    doc1 = "Does something important with a."
    k1 = make_cache_key(sig1, diff1, doc1)
    k2 = make_cache_key(sig1, diff1, doc1)
    assert k1 == k2
    assert len(k1) == 40  # SHA1 hexdigest


def test_make_cache_key_includes_signature() -> None:
    """Point 8: same diff+doc under different signatures must not collide."""
    diff = "-x = 1\n+x = 2"
    doc = "A valid docstring with plenty of characters here."
    assert make_cache_key("def get(a):", diff, doc) != make_cache_key(
        "def post(a):", diff, doc
    )


def test_gate_drift_threshold() -> None:
    classes = ["param_added", "param_removed"]
    assert gate_drift(classes, 0, 0.9, 0.5) == "param_added"
    assert gate_drift(classes, 1, 0.9, 0.5) == "param_removed"
    assert gate_drift(classes, 0, 0.4, 0.5) == "none"
    assert gate_drift(classes, 0, 0.5, 0.5) == "none"  # strict >
    assert gate_drift(classes, 5, 0.9, 0.5) == "none"  # bad id


def test_replay_cache_corrupted_file_is_quarantined(tmp_path: Path) -> None:
    """Point 1: a truncated cache file is backed up, not silently dropped."""
    cache_file = tmp_path / "cache.json"
    cache_file.write_text('{"m0:abc": {"p_stale": 0.5, ', encoding="utf-8")
    cache = ReplayCache(cache_file)
    assert cache._data == {}
    backups = list(tmp_path.glob("cache.json.corrupted.bak"))
    assert len(backups) == 1


def test_replace_branch_records_is_atomic(tmp_path: Path) -> None:
    """Point 2: failed insert keeps the previous branch state (rollback)."""
    import sqlite3

    db = tmp_path / "registry.db"
    init_db(db)
    good = {
        "repo": "org/r",
        "branch": "main",
        "commit_sha": "aaa",
        "section_id": "f.py::f#docstring",
        "computed_at": "2026-01-01T00:00:00+00:00",
    }
    replace_branch_records(db, "org/r", "main", [good])
    assert len(latest(db, "org/r", "main")) == 1
    # Second record violates NOT NULL (commit_sha=None) -> IntegrityError.
    bad = dict(good, commit_sha=None, section_id="f.py::g#docstring")
    with pytest.raises(Exception):
        replace_branch_records(db, "org/r", "main", [bad])
    rows = latest(db, "org/r", "main")
    assert len(rows) == 1
    assert rows[0]["section_id"] == "f.py::f#docstring"


def test_nested_qualnames_detection() -> None:
    tree = ast.parse(
        "def outer(x):\n"
        "    '''Outer docstring that is long enough yes.'''\n"
        "    def inner(y):\n"
        "        '''Inner helper docstring long enough too.'''\n"
        "        return y\n"
        "    return inner(x)\n"
        "class C:\n"
        "    def method(self):\n"
        "        '''Method docstring that is long enough yes.'''\n"
        "        return 1\n"
    )
    nested = _nested_qualnames(tree)
    assert "outer.inner" in nested
    assert "outer" not in nested
    assert "C.method" not in nested


def test_m2_scorer_score_gating_fake_backbone() -> None:
    """Point 4: M2Scorer.score path (encode + calibrate + gate) on CPU stub."""
    torch = pytest.importorskip("torch")
    from src.models.m2_data import DRIFT_CLASSES
    from src.models.m2_model import DualHeadCodeClassifier
    from src.registry.replay import M2Scorer
    from tests.test_m2 import _FakeTok, _TinyBackbone

    torch.manual_seed(0)
    scorer = object.__new__(M2Scorer)
    scorer.tokenizer = _FakeTok()
    scorer.model = DualHeadCodeClassifier(_TinyBackbone(hidden=16), hidden_size=16)
    scorer.model.eval()
    scorer.device = torch.device("cpu")
    scorer.temperature = 2.384710179084571
    scorer.max_len = 64
    scorer.doc_budget = 32
    scorer.drift_classes = DRIFT_CLASSES

    fn_old = ast.parse(
        "def f(a):\n    '''Docstring long enough for the test yes.'''\n    return a\n"
    ).body[0]
    fn_new = ast.parse(
        "def f(a, b=1):\n    '''Docstring long enough for the test yes.'''\n    return a\n"
    ).body[0]

    scorer.threshold = 0.0  # everything gated to argmax class
    p_hi, dt_hi = scorer.score("def f(a):", "-x\n+y", "Docstring long enough.", fn_old, fn_new)
    assert 0.0 <= p_hi <= 1.0
    assert dt_hi in DRIFT_CLASSES

    scorer.threshold = 2.0  # nothing passes (stub backbone is random per call,
    # so only the gate outcome is asserted, not the probability value)
    p_lo, dt_lo = scorer.score("def f(a):", "-x\n+y", "Docstring long enough.", fn_old, fn_new)
    assert 0.0 <= p_lo <= 1.0
    assert dt_lo == "none"


def test_replay_cache_set_get_and_persistence(tmp_path: Path) -> None:
    cache_file = tmp_path / "cache.json"
    cache = ReplayCache(cache_file)
    assert cache.hits == 0
    assert cache.misses == 0

    sig = "def foo(a):"
    diff = "-x = 1\n+x = 2"
    doc = "A valid docstring with plenty of characters."
    assert cache.get("m0", diff, doc, sig) is None
    assert cache.misses == 1

    cache.set("m0", diff, doc, p_stale=0.75, drift_type="param_added", sig_old=sig)
    val = cache.get("m0", diff, doc, sig)
    assert val is not None
    assert val["p_stale"] == 0.75
    assert val["drift_type"] == "param_added"
    assert cache.hits == 1
    # Different signature -> different entry (point 8).
    assert cache.get("m0", diff, doc, "def bar(a):") is None

    cache.save()
    assert cache_file.exists()
    # Atomic save leaves no temp files behind.
    assert list(tmp_path.glob("cache.json.*.tmp")) == []

    # Load in new cache instance
    cache2 = ReplayCache(cache_file)
    val2 = cache2.get("m0", diff, doc, sig)
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


def test_load_fan_in_map_missing_repo_returns_empty(tmp_path: Path) -> None:
    assert _load_fan_in_map(tmp_path / "does_not_exist") == {}
    assert _load_fan_in_map("/fake/repo") == {}


def test_replay_impact_uses_real_fan_in(tmp_path: Path) -> None:
    """R12/FD-28: impact must reflect call counts on the checkout, not a constant."""
    # HEAD checkout contains two calls to process() -> fan_in=2.
    (tmp_path / "caller.py").write_text(
        "from worker import process\nprocess(1)\nprocess(2)\n", encoding="utf-8"
    )
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
        ("sha1", "2026-04-01T12:00:00+00:00"),
        ("sha2", "2026-04-02T12:00:00+00:00"),
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
        _, records = replay_project_history(
            repo_path=tmp_path,
            repo_name="test/fanin",
            scorer=spy_scorer,
        )

    assert len(records) == 1
    # fan_in=2 -> log1p(2)/log1p(20) ~= 0.3609 (public, default caps).
    import math

    expected = math.log1p(2.0) / math.log1p(20.0)
    assert records[0]["impact"] == round(expected, 4)
    assert records[0]["impact"] > 0.0


def test_replay_excludes_dunder_init() -> None:
    """R1: names starting with '_' (including __init__) are excluded from replay."""
    spy_scorer = MagicMock()
    spy_scorer.name = "spy"
    spy_scorer.score.return_value = (0.9, "param_added")

    code_before = (
        "class Widget:\n"
        "    def __init__(self, x):\n"
        "        '''Create a widget with the given size parameter for rendering.'''\n"
        "        self.x = x\n"
        "        self.y = x + 1\n"
        "        return None\n"
    )
    code_after = (
        "class Widget:\n"
        "    def __init__(self, x, y=1):\n"
        "        '''Create a widget with the given size parameter for rendering.'''\n"
        "        self.x = x\n"
        "        self.y = x + y\n"
        "        return None\n"
    )
    commits = [
        ("sha1", "2026-05-01T12:00:00+00:00"),
        ("sha2", "2026-05-02T12:00:00+00:00"),
    ]

    def mock_get_commits(repo_path, max_commits=300):
        return list(reversed(commits))

    def mock_get_py_files(repo_path, sha):
        return ["widget.py"] if sha == "sha2" else []

    def mock_show_blob(repo_path, sha, path, before=False):
        return code_before if before else code_after

    with patch("src.registry.replay.get_first_parent_commits", side_effect=mock_get_commits), \
         patch("src.registry.replay.get_modified_py_files", side_effect=mock_get_py_files), \
         patch("src.registry.replay.show_blob", side_effect=mock_show_blob):
        _, records = replay_project_history(
            repo_path="/fake/repo",
            repo_name="test/dunder",
            scorer=spy_scorer,
        )

    assert spy_scorer.score.call_count == 0
    assert len(records) == 0


def test_replay_evidence_limits() -> None:
    """FD-30 needs the cause: evidence keeps long diffs/docs up to raised limits."""
    assert EVIDENCE_DIFF_LIMIT >= 2000
    assert EVIDENCE_DOC_LIMIT >= 500

    spy_scorer = MagicMock()
    spy_scorer.name = "spy"
    spy_scorer.score.return_value = (0.9, "param_added")

    long_doc = "Long docstring sentence about behavior. " * 30  # >500 chars
    body_before = "\n".join(f"    v{i} = {i} + x" for i in range(60))
    body_after = "\n".join(f"    v{i} = {i} + x + 1" for i in range(60))
    code_before = f"def bigfunc(x):\n    '''{long_doc}'''\n{body_before}\n    return v0\n"
    code_after = f"def bigfunc(x, extra=1):\n    '''{long_doc}'''\n{body_after}\n    return v0\n"
    commits = [
        ("sha1", "2026-06-01T12:00:00+00:00"),
        ("sha2", "2026-06-02T12:00:00+00:00"),
    ]

    def mock_get_commits(repo_path, max_commits=300):
        return list(reversed(commits))

    def mock_get_py_files(repo_path, sha):
        return ["big.py"] if sha == "sha2" else []

    def mock_show_blob(repo_path, sha, path, before=False):
        return code_before if before else code_after

    with patch("src.registry.replay.get_first_parent_commits", side_effect=mock_get_commits), \
         patch("src.registry.replay.get_modified_py_files", side_effect=mock_get_py_files), \
         patch("src.registry.replay.show_blob", side_effect=mock_show_blob):
        _, records = replay_project_history(
            repo_path="/fake/repo",
            repo_name="test/evidence",
            scorer=spy_scorer,
        )

    assert len(records) == 1
    assert len(records[0]["evidence"]["diff"]) <= EVIDENCE_DIFF_LIMIT
    # Long diff must not be cut to the old 200-char stub.
    assert len(records[0]["evidence"]["diff"]) > 200


def _run_replay_with_blobs(code_before: str, code_after: str, **kwargs):
    """Helper: two-commit replay (C1 baseline, C2 change) with mocked git."""
    commits = [
        ("sha1", kwargs.pop("dt1", "2026-01-01T12:00:00+00:00")),
        ("sha2", kwargs.pop("dt2", "2026-02-01T12:00:00+00:00")),
    ]

    def mock_get_commits(repo_path, max_commits=300):
        return list(reversed(commits))

    def mock_get_py_files(repo_path, sha):
        return ["mod.py"] if sha == "sha2" else []

    def mock_show_blob(repo_path, sha, path, before=False):
        return code_before if before else code_after

    spy_scorer = MagicMock()
    spy_scorer.name = "spy"
    spy_scorer.score.return_value = (0.9, "param_added")
    with patch("src.registry.replay.get_first_parent_commits", side_effect=mock_get_commits), \
         patch("src.registry.replay.get_modified_py_files", side_effect=mock_get_py_files), \
         patch("src.registry.replay.show_blob", side_effect=mock_show_blob):
        return replay_project_history(
            repo_path=kwargs.pop("repo_path", "/fake/repo"),
            repo_name="test/edge",
            scorer=spy_scorer,
            risk_config=RiskConfig(threshold=0.5),
            **kwargs,
        )


def test_replay_doc_removed_flags_without_reset() -> None:
    """Point 7: deleted docs are doc_removed (risk accumulates), not a fix."""
    code_before = (
        "def worker(x):\n"
        "    '''Worker docstring describing the size parameter in detail here.'''\n"
        "    a = x + 1\n"
        "    b = a * 2\n"
        "    return b\n"
    )
    code_after = (
        "def worker(x):\n"
        "    a = x + 1\n"
        "    b = a * 2\n"
        "    return b\n"
    )
    metrics, records = _run_replay_with_blobs(code_before, code_after)
    assert metrics.total_fixes == 0
    assert len(records) == 1
    assert records[0]["event"] == "doc_removed"
    assert records[0]["risk"] == 1.0
    assert records[0]["verified_commit"] is None


def test_replay_private_class_method_skipped() -> None:
    """Point 15: public method of a private class is internal, skip it."""
    code_before = (
        "class _Worker:\n"
        "    def run(self, x):\n"
        "        '''Run the worker with the given size parameter in detail now.'''\n"
        "        a = x + 1\n"
        "        b = a * 2\n"
        "        return b\n"
    )
    code_after = (
        "class _Worker:\n"
        "    def run(self, x, y=1):\n"
        "        '''Run the worker with the given size parameter in detail now.'''\n"
        "        a = x + y\n"
        "        b = a * 2\n"
        "        return b\n"
    )
    _, records = _run_replay_with_blobs(code_before, code_after)
    assert len(records) == 0


def test_replay_closure_skipped_but_method_kept() -> None:
    """Point 16: nested closures are skipped, class methods are tracked."""
    code_before = (
        "def outer(x):\n"
        "    '''Outer function documented with the size parameter in detail.'''\n"
        "    def inner(y):\n"
        "        '''Inner helper documented sufficiently for the test case.'''\n"
        "        z = y + 1\n"
        "        w = z * 2\n"
        "        return w\n"
        "    a = inner(x)\n"
        "    b = a * 2\n"
        "    return b\n"
        "class Widget:\n"
        "    def render(self, x):\n"
        "        '''Render the widget with the given size parameter in detail.'''\n"
        "        a = x + 1\n"
        "        b = a * 2\n"
        "        return b\n"
    )
    code_after = (
        "def outer(x):\n"
        "    '''Outer function documented with the size parameter in detail.'''\n"
        "    def inner(y, z=0):\n"
        "        '''Inner helper documented sufficiently for the test case.'''\n"
        "        q = y + z\n"
        "        w = q * 2\n"
        "        return w\n"
        "    a = inner(x)\n"
        "    b = a * 2\n"
        "    return b\n"
        "class Widget:\n"
        "    def render(self, x, mode=1):\n"
        "        '''Render the widget with the given size parameter in detail.'''\n"
        "        a = x + mode\n"
        "        b = a * 2\n"
        "        return b\n"
    )
    metrics, records = _run_replay_with_blobs(code_before, code_after)
    symbols = {r["symbol"] for r in records}
    assert "outer.inner" not in symbols
    assert "Widget.render" in symbols


def test_replay_negative_lead_time_clamped() -> None:
    """Point 14: fix dated before the flag yields lead 0.0, not negative."""
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
        "    '''Compute the grand total with optional custom multiplier here.'''\n"
        "    a = x * multiplier\n"
        "    b = a + 5\n"
        "    return b\n"
    )
    # Fix commit (C3) is dated BEFORE the flag commit (C2): broken clock.
    commits = [
        ("sha_c1", "2026-01-01T10:00:00+00:00"),
        ("sha_c2", "2026-03-01T10:00:00+00:00"),
        ("sha_c3", "2026-02-01T10:00:00+00:00"),
    ]

    def mock_get_commits(repo_path, max_commits=300):
        return list(reversed(commits))

    def mock_get_py_files(repo_path, sha):
        return [] if sha == "sha_c1" else ["calc/math.py"]

    def mock_show_blob(repo_path, sha, path, before=False):
        if sha == "sha_c2":
            return c1 if before else c2
        if sha == "sha_c3":
            return c2 if before else c3
        return ""

    with patch("src.registry.replay.get_first_parent_commits", side_effect=mock_get_commits), \
         patch("src.registry.replay.get_modified_py_files", side_effect=mock_get_py_files), \
         patch("src.registry.replay.show_blob", side_effect=mock_show_blob):
        metrics, _ = replay_project_history(
            repo_path="/fake/repo",
            repo_name="test/clock",
            scorer=M0Scorer(),
            risk_config=RiskConfig(threshold=0.5),
        )
    assert metrics.total_fixes == 1
    assert metrics.preceded_fixes == 1
    assert metrics.median_lead_time_days == 0.0
    assert metrics.mean_lead_time_days == 0.0


def test_replay_urgency_grows_for_never_verified(tmp_path: Path) -> None:
    """Point 9: sections seen long ago gain urgency even without verification."""
    code_before = (
        "def process(item):\n"
        "    '''Process data item and format output properly for the user now.'''\n"
        "    x = str(item)\n"
        "    y = x.strip()\n"
        "    return y\n"
    )
    code_mid = (
        "def process(item, mode='a'):\n"
        "    '''Process data item and format output properly for the user now.'''\n"
        "    x = str(item)\n"
        "    y = x.strip()\n"
        "    return y\n"
    )
    code_after = (
        "def process(item, mode='a', extra=0):\n"
        "    '''Process data item and format output properly for the user now.'''\n"
        "    x = str(item)\n"
        "    y = x.strip()\n"
        "    return y\n"
    )
    commits = [
        ("sha1", "2026-01-01T12:00:00+00:00"),
        ("sha2", "2026-01-02T12:00:00+00:00"),
        ("sha3", "2026-06-01T12:00:00+00:00"),
    ]

    def mock_get_commits(repo_path, max_commits=300):
        return list(reversed(commits))

    def mock_get_py_files(repo_path, sha):
        return [] if sha == "sha1" else ["worker.py"]

    def mock_show_blob(repo_path, sha, path, before=False):
        if sha == "sha2":
            return code_before if before else code_mid
        return code_mid if before else code_after

    spy_scorer = MagicMock()
    spy_scorer.name = "spy"
    spy_scorer.score.return_value = (0.3, "param_added")
    with patch("src.registry.replay.get_first_parent_commits", side_effect=mock_get_commits), \
         patch("src.registry.replay.get_modified_py_files", side_effect=mock_get_py_files), \
         patch("src.registry.replay.show_blob", side_effect=mock_show_blob):
        _, records = replay_project_history(
            repo_path=str(tmp_path),
            repo_name="test/age",
            scorer=spy_scorer,
            risk_config=RiskConfig(threshold=0.99),
        )
    assert len(records) == 2
    # ~150 days of section age push urgency above raw risk on the later event.
    assert records[1]["urgency"] > records[1]["risk"]


def test_replay_section_histories_keyed_by_section_id() -> None:
    """Point 5: same qualname in different files must not overwrite history."""
    before_a = (
        "def setup(x):\n"
        "    '''Setup routine documented with the size parameter in detail.'''\n"
        "    a = x + 1\n"
        "    b = a * 2\n"
        "    return b\n"
    )
    after_a = (
        "def setup(x, y=1):\n"
        "    '''Setup routine documented with the size parameter in detail.'''\n"
        "    a = x + y\n"
        "    b = a * 2\n"
        "    return b\n"
    )
    before_b = (
        "def setup(x):\n"
        "    '''Setup helper documented with the size parameter in detail!'''\n"
        "    a = x + 3\n"
        "    b = a * 4\n"
        "    return b\n"
    )
    after_b = (
        "def setup(x, y=2):\n"
        "    '''Setup helper documented with the size parameter in detail!'''\n"
        "    a = x + y\n"
        "    b = a * 4\n"
        "    return b\n"
    )
    blobs = {("sha2", "a.py"): (before_a, after_a), ("sha2", "b.py"): (before_b, after_b)}
    commits = [
        ("sha1", "2026-01-01T12:00:00+00:00"),
        ("sha2", "2026-01-02T12:00:00+00:00"),
    ]

    def mock_get_commits(repo_path, max_commits=300):
        return list(reversed(commits))

    def mock_get_py_files(repo_path, sha):
        return ["a.py", "b.py"] if sha == "sha2" else []

    def mock_show_blob(repo_path, sha, path, before=False):
        pair = blobs.get((sha, path))
        if pair is None:
            return ""
        return pair[0] if before else pair[1]

    spy_scorer = MagicMock()
    spy_scorer.name = "spy"
    spy_scorer.score.return_value = (0.9, "param_added")
    with patch("src.registry.replay.get_first_parent_commits", side_effect=mock_get_commits), \
         patch("src.registry.replay.get_modified_py_files", side_effect=mock_get_py_files), \
         patch("src.registry.replay.show_blob", side_effect=mock_show_blob):
        metrics, _ = replay_project_history(
            repo_path="/fake/repo",
            repo_name="test/collide",
            scorer=spy_scorer,
            risk_config=RiskConfig(threshold=0.5),
        )
    assert "a.py::setup#docstring" in metrics.section_histories
    assert "b.py::setup#docstring" in metrics.section_histories


def test_replay_doc_degraded_below_threshold_is_not_fix() -> None:
    """Point 2: docs shrunk to a stub must not count as a fix/reset."""
    code_before = (
        "def worker(x):\n"
        "    '''Worker docstring describing the size parameter in detail here.'''\n"
        "    a = x + 1\n"
        "    b = a * 2\n"
        "    return b\n"
    )
    code_after = (
        "def worker(x):\n"
        "    '''TODO: fix'''\n"
        "    a = x + 1\n"
        "    b = a * 2\n"
        "    return b\n"
    )
    metrics, records = _run_replay_with_blobs(code_before, code_after)
    assert metrics.total_fixes == 0
    assert len(records) == 1
    assert records[0]["event"] == "doc_degraded"
    assert records[0]["risk"] == 1.0
    assert records[0]["verified_commit"] is None


def test_replay_doc_removed_recorded_in_samples() -> None:
    """Point 7: doc_removed events reach sample_timelines for the chart."""
    code_before = (
        "def worker(x):\n"
        "    '''Worker docstring describing the size parameter in detail here.'''\n"
        "    a = x + 1\n"
        "    b = a * 2\n"
        "    return b\n"
    )
    code_after = (
        "def worker(x):\n"
        "    a = x + 1\n"
        "    b = a * 2\n"
        "    return b\n"
    )
    commits = [
        ("sha1", "2026-01-01T12:00:00+00:00"),
        ("sha2", "2026-02-01T12:00:00+00:00"),
    ]

    def mock_get_commits(repo_path, max_commits=300):
        return list(reversed(commits))

    def mock_get_py_files(repo_path, sha):
        return ["mod.py"] if sha == "sha2" else []

    def mock_show_blob(repo_path, sha, path, before=False):
        return code_before if before else code_after

    spy_scorer = MagicMock()
    spy_scorer.name = "spy"
    spy_scorer.score.return_value = (0.9, "param_added")
    with patch("src.registry.replay.get_first_parent_commits", side_effect=mock_get_commits), \
         patch("src.registry.replay.get_modified_py_files", side_effect=mock_get_py_files), \
         patch("src.registry.replay.show_blob", side_effect=mock_show_blob):
        metrics, _ = replay_project_history(
            repo_path="/fake/repo",
            repo_name="test/edge",
            scorer=spy_scorer,
            risk_config=RiskConfig(threshold=0.5),
            sample_symbols=["worker"],
        )
    assert metrics.timeline_samples["worker"][0]["event"] == "doc_removed"


def test_replay_empty_repo_raises_value_error() -> None:
    """Point 9: a repo without commits yields a clear error, not git noise."""
    import subprocess

    def boom(repo_path, *args, **kwargs):
        raise subprocess.CalledProcessError(128, ["git", "log"])

    with patch("src.registry.replay._git_cmd", side_effect=boom):
        from src.registry.replay import get_first_parent_commits

        assert get_first_parent_commits("/fake/empty") == []
        with pytest.raises(ValueError, match="No commits"):
            replay_project_history(repo_path="/fake/empty", repo_name="test/empty")


def test_replay_null_bytes_blob_skipped() -> None:
    """Point 10: binary-looking .py content is skipped, not fatal."""
    code_before = (
        "def worker(x):\n"
        "    '''Worker docstring describing the size parameter in detail here.'''\n"
        "    a = x + 1\n"
        "    b = a * 2\n"
        "    return b\n"
    )
    commits = [
        ("sha1", "2026-01-01T12:00:00+00:00"),
        ("sha2", "2026-02-01T12:00:00+00:00"),
    ]

    def mock_get_commits(repo_path, max_commits=300):
        return list(reversed(commits))

    def mock_get_py_files(repo_path, sha):
        return ["mod.py"] if sha == "sha2" else []

    def mock_show_blob(repo_path, sha, path, before=False):
        return code_before if before else "def wo\x00rker(:\n binary \x00 blob"

    spy_scorer = MagicMock()
    spy_scorer.name = "spy"
    spy_scorer.score.return_value = (0.9, "param_added")
    with patch("src.registry.replay.get_first_parent_commits", side_effect=mock_get_commits), \
         patch("src.registry.replay.get_modified_py_files", side_effect=mock_get_py_files), \
         patch("src.registry.replay.show_blob", side_effect=mock_show_blob):
        metrics, records = replay_project_history(
            repo_path="/fake/repo",
            repo_name="test/edge",
            scorer=spy_scorer,
            risk_config=RiskConfig(threshold=0.5),
        )
    assert records == []
    assert metrics.total_fixes == 0

