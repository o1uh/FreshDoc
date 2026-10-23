from __future__ import annotations

from pathlib import Path
import sqlite3
import pytest

from src.registry.storage import (
    DDL_DOC_STATE,
    commit_known,
    get_connection,
    history,
    init_db,
    insert,
    insert_batch,
    latest,
    merge_branch,
    replace_branch_records,
    verify,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Fixture providing a fresh SQLite database path for each test."""
    p = tmp_path / "test_doc_state.db"
    init_db(p)
    return p


def test_init_db_creates_table_and_indices(db_path: Path) -> None:
    """Check that doc_state table and indices exist after init_db."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='doc_state'"
        )
        assert cursor.fetchone() is not None

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_doc_state_%'"
        )
        indices = {row[0] for row in cursor.fetchall()}
        assert "idx_doc_state_lookup" in indices
        assert "idx_doc_state_section" in indices
        assert "idx_doc_state_commit" in indices


def test_insert_and_auto_sequence(db_path: Path) -> None:
    """Check that inserting records auto-increments seq per (repo, branch)."""
    rec1 = {
        "repo": "owner/repo",
        "branch": "main",
        "commit_sha": "c1",
        "section_id": "mod.py::func_a#docstring",
        "doc_path": "mod.py",
        "symbol": "func_a",
        "risk": 0.3,
    }
    rec2 = {
        "repo": "owner/repo",
        "branch": "main",
        "commit_sha": "c2",
        "section_id": "mod.py::func_a#docstring",
        "doc_path": "mod.py",
        "symbol": "func_a",
        "risk": 0.8,
    }
    id1 = insert(db_path, rec1)
    id2 = insert(db_path, rec2)
    assert id1 > 0
    assert id2 > id1

    hist = history(db_path, "mod.py::func_a#docstring")
    assert len(hist) == 2
    assert hist[0]["seq"] == 1
    assert hist[1]["seq"] == 2


def test_insert_batch(db_path: Path) -> None:
    """Check batch insert into registry."""
    records = [
        {
            "repo": "owner/repo",
            "branch": "main",
            "commit_sha": "c1",
            "section_id": f"mod.py::func_{i}#docstring",
            "doc_path": "mod.py",
            "symbol": f"func_{i}",
            "risk": 0.1 * i,
        }
        for i in range(5)
    ]
    ids = insert_batch(db_path, records)
    assert len(ids) == 5

    res = latest(db_path, repo="owner/repo", branch="main")
    assert len(res) == 5


def test_insert_batch_returns_real_ids() -> None:
    """Point 1: executemany does not set lastrowid — ids must be real (>0)."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "ids.db"
        init_db(db)
        recs = [
            {
                "repo": "o/r",
                "branch": "main",
                "commit_sha": f"c{i}",
                "section_id": f"m.py::f{i}#docstring",
            }
            for i in range(3)
        ]
        ids = insert_batch(db, recs)
        assert ids == [1, 2, 3]
        recs2 = [dict(r, branch="other") for r in recs]
        ids2 = replace_branch_records(db, "o/r", "other", recs2)
        assert ids2 == [4, 5, 6]


def test_replace_branch_ids_start_at_fresh_seq(tmp_path: Path) -> None:
    """S1: after DELETE the next seq is 1 without an extra SELECT."""
    db = tmp_path / "seq.db"
    init_db(db)
    recs = [
        {"repo": "o/r", "branch": "b", "commit_sha": f"c{i}",
         "section_id": f"m.py::f{i}#docstring"}
        for i in range(2)
    ]
    replace_branch_records(db, "o/r", "b", recs)
    rows = latest(db, "o/r", "b")
    assert sorted(r["seq"] for r in rows) == [1, 2]


def test_latest_normalizes_windows_path(db_path: Path) -> None:
    """Point 6: backslash path filters must match git-style doc_path."""
    insert(
        db_path,
        {"repo": "o/r", "branch": "main", "commit_sha": "c1",
         "section_id": "pkg/mod.py::f#docstring", "doc_path": "pkg/mod.py",
         "symbol": "f", "risk": 0.5},
    )
    assert len(latest(db_path, "o/r", "main", path="pkg/mod.py")) == 1
    assert len(latest(db_path, "o/r", "main", path="pkg\\mod.py")) == 1
    assert len(latest(db_path, "o/r", "main", path="pkg\\")) == 1
    assert len(history(db_path, "pkg\\mod.py::f#docstring")) == 1


def test_commit_known_distinguishes_unknown_commit(db_path: Path) -> None:
    """Point 4: unknown commit vs empty state helper for API/dashboard."""
    insert(
        db_path,
        {"repo": "o/r", "branch": "main", "commit_sha": "c1",
         "section_id": "m.py::f#docstring", "risk": 0.5},
    )
    assert commit_known(db_path, "o/r", "main", "c1") is True
    assert commit_known(db_path, "o/r", "main", "nope") is False
    assert latest(db_path, "o/r", "main", commit="nope") == []


def test_connections_released_db_file_removable(tmp_path: Path) -> None:
    """Point 3: no leaked descriptors — the DB file can be moved/deleted."""
    import os

    db = tmp_path / "lock.db"
    init_db(db)
    insert_batch(db, [
        {"repo": "o/r", "branch": "main", "commit_sha": "c1",
         "section_id": "m.py::f#docstring"}
    ])
    latest(db, "o/r", "main")
    history(db, "m.py::f#docstring")
    moved = tmp_path / "moved.db"
    os.replace(db, moved)  # raises on Windows while handles are open
    assert moved.exists()
    os.remove(moved)


def test_latest_on_empty_db(db_path: Path) -> None:
    """Check that latest on empty db returns empty list."""
    res = latest(db_path, repo="owner/repo", branch="main")
    assert res == []


def test_latest_returns_highest_seq_per_section(db_path: Path) -> None:
    """Ensure latest returns only the most recent entry for each section."""
    insert(
        db_path,
        {
            "repo": "owner/repo",
            "branch": "main",
            "commit_sha": "c1",
            "section_id": "mod.py::func#docstring",
            "doc_path": "mod.py",
            "symbol": "func",
            "risk": 0.2,
        },
    )
    insert(
        db_path,
        {
            "repo": "owner/repo",
            "branch": "main",
            "commit_sha": "c2",
            "section_id": "mod.py::func#docstring",
            "doc_path": "mod.py",
            "symbol": "func",
            "risk": 0.75,
            "evidence": {"diff": "+line"},
        },
    )
    res = latest(db_path, repo="owner/repo", branch="main")
    assert len(res) == 1
    assert res[0]["commit_sha"] == "c2"
    assert res[0]["risk"] == 0.75
    assert res[0]["evidence"] == {"diff": "+line"}


def test_latest_with_commit_cutoff(db_path: Path) -> None:
    """Ensure latest respects commit cutoff by seq."""
    insert(
        db_path,
        {
            "repo": "owner/repo",
            "branch": "main",
            "commit_sha": "c1",
            "section_id": "mod.py::func#docstring",
            "risk": 0.1,
        },
    )
    insert(
        db_path,
        {
            "repo": "owner/repo",
            "branch": "main",
            "commit_sha": "c2",
            "section_id": "mod.py::func#docstring",
            "risk": 0.5,
        },
    )
    insert(
        db_path,
        {
            "repo": "owner/repo",
            "branch": "main",
            "commit_sha": "c3",
            "section_id": "mod.py::func#docstring",
            "risk": 0.9,
        },
    )

    res_c1 = latest(db_path, repo="owner/repo", branch="main", commit="c1")
    assert len(res_c1) == 1
    assert res_c1[0]["commit_sha"] == "c1"
    assert res_c1[0]["risk"] == 0.1

    res_c2 = latest(db_path, repo="owner/repo", branch="main", commit="c2")
    assert len(res_c2) == 1
    assert res_c2[0]["commit_sha"] == "c2"
    assert res_c2[0]["risk"] == 0.5

    # Unknown commit returns empty list
    assert latest(db_path, repo="owner/repo", branch="main", commit="unknown") == []


def test_latest_path_and_min_risk_filters(db_path: Path) -> None:
    """Test filtering by doc_path prefix and min_risk."""
    records = [
        {
            "repo": "owner/repo",
            "branch": "main",
            "commit_sha": "c1",
            "section_id": "pkg/a.py::f1#docstring",
            "doc_path": "pkg/a.py",
            "risk": 0.2,
        },
        {
            "repo": "owner/repo",
            "branch": "main",
            "commit_sha": "c1",
            "section_id": "pkg/b.py::f2#docstring",
            "doc_path": "pkg/b.py",
            "risk": 0.8,
        },
        {
            "repo": "owner/repo",
            "branch": "main",
            "commit_sha": "c1",
            "section_id": "other/c.py::f3#docstring",
            "doc_path": "other/c.py",
            "risk": 0.9,
        },
    ]
    insert_batch(db_path, records)

    pkg_res = latest(db_path, repo="owner/repo", branch="main", path="pkg/")
    assert len(pkg_res) == 2
    assert {r["section_id"] for r in pkg_res} == {
        "pkg/a.py::f1#docstring",
        "pkg/b.py::f2#docstring",
    }

    risk_res = latest(db_path, repo="owner/repo", branch="main", min_risk=0.5)
    assert len(risk_res) == 2
    assert {r["section_id"] for r in risk_res} == {
        "pkg/b.py::f2#docstring",
        "other/c.py::f3#docstring",
    }


def test_verify_resets_risk(db_path: Path) -> None:
    """Test that verify records verification event and resets risk to 0."""
    insert(
        db_path,
        {
            "repo": "owner/repo",
            "branch": "main",
            "commit_sha": "c1",
            "section_id": "mod.py::func#docstring",
            "doc_path": "mod.py",
            "symbol": "func",
            "risk": 0.95,
            "urgency": 0.90,
            "impact": 0.5,
        },
    )

    ver_rec = verify(
        db_path,
        repo="owner/repo",
        branch="main",
        section_id="mod.py::func#docstring",
        commit="c2",
        verified_by="test_user",
    )
    assert ver_rec["risk"] == 0.0
    assert ver_rec["urgency"] == 0.0
    assert ver_rec["p_stale"] == 0.0
    assert ver_rec["verified_commit"] == "c2"
    assert ver_rec["event"] == "verify"
    assert ver_rec["doc_path"] == "mod.py"
    assert ver_rec["symbol"] == "func"
    assert ver_rec["impact"] == 0.5

    cur_state = latest(db_path, repo="owner/repo", branch="main")
    assert len(cur_state) == 1
    assert cur_state[0]["risk"] == 0.0
    assert cur_state[0]["commit_sha"] == "c2"


def test_merge_branch_maximum_risk(db_path: Path) -> None:
    """Test merge_branch takes max risk for sections present on either/both branches."""
    # Setup target branch
    insert_batch(
        db_path,
        [
            {
                "repo": "owner/repo",
                "branch": "main",
                "commit_sha": "m1",
                "section_id": "sec_common",
                "doc_path": "common.py",
                "risk": 0.3,
            },
            {
                "repo": "owner/repo",
                "branch": "main",
                "commit_sha": "m1",
                "section_id": "sec_main_only",
                "doc_path": "main_only.py",
                "risk": 0.4,
            },
        ],
    )

    # Setup source branch (feature)
    insert_batch(
        db_path,
        [
            {
                "repo": "owner/repo",
                "branch": "feat",
                "commit_sha": "f1",
                "section_id": "sec_common",
                "doc_path": "common.py",
                "risk": 0.85,  # higher than main (0.3)
            },
            {
                "repo": "owner/repo",
                "branch": "feat",
                "commit_sha": "f1",
                "section_id": "sec_feat_only",
                "doc_path": "feat.py",
                "risk": 0.6,
            },
        ],
    )

    merged = merge_branch(
        db_path,
        repo="owner/repo",
        source_branch="feat",
        target_branch="main",
        merge_commit_sha="m_merge",
    )
    assert len(merged) == 3
    merged_map = {r["section_id"]: r for r in merged}

    # Common section got max risk 0.85
    assert merged_map["sec_common"]["risk"] == 0.85
    assert merged_map["sec_common"]["commit_sha"] == "m_merge"
    assert merged_map["sec_common"]["event"] == "merge"

    # Main-only retained risk 0.4
    assert merged_map["sec_main_only"]["risk"] == 0.4
    assert merged_map["sec_main_only"]["commit_sha"] == "m_merge"

    # Feat-only transferred to main with risk 0.6
    assert merged_map["sec_feat_only"]["risk"] == 0.6
    assert merged_map["sec_feat_only"]["branch"] == "main"
    assert merged_map["sec_feat_only"]["commit_sha"] == "m_merge"


def test_verify_same_commit_is_idempotent(db_path: Path) -> None:
    """Verify on an already-recorded commit updates the row in place."""
    insert(
        db_path,
        {
            "repo": "owner/repo",
            "branch": "main",
            "commit_sha": "c1",
            "section_id": "mod.py::func#docstring",
            "doc_path": "mod.py",
            "symbol": "func",
            "risk": 0.9,
            "p_stale": 0.8,
        },
    )
    first = verify(
        db_path,
        repo="owner/repo",
        branch="main",
        section_id="mod.py::func#docstring",
        commit="c1",
    )
    assert first["risk"] == 0.0
    assert first["verified_commit"] == "c1"

    second = verify(
        db_path,
        repo="owner/repo",
        branch="main",
        section_id="mod.py::func#docstring",
        commit="c1",
    )
    assert second["risk"] == 0.0
    assert second["verified_commit"] == "c1"

    cur = latest(db_path, repo="owner/repo", branch="main")
    assert len(cur) == 1
    assert cur[0]["risk"] == 0.0


def test_memory_db_shared_across_connections() -> None:
    """Check that ':memory:' is one shared DB across connections."""
    init_db(":memory:")
    insert(
        ":memory:",
        {
            "repo": "owner/repo",
            "branch": "main",
            "commit_sha": "c1",
            "section_id": "mod.py::mem#docstring",
            "risk": 0.5,
        },
    )
    res = latest(":memory:", repo="owner/repo", branch="main")
    assert len(res) == 1
    assert res[0]["section_id"] == "mod.py::mem#docstring"
