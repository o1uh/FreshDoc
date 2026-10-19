from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

DDL_DOC_STATE = """
CREATE TABLE IF NOT EXISTS doc_state (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  seq INTEGER NOT NULL,
  repo TEXT NOT NULL,
  branch TEXT NOT NULL,
  commit_sha TEXT NOT NULL,
  build_id TEXT,
  doc_path TEXT,
  section_id TEXT NOT NULL,
  symbol TEXT,
  p_stale REAL,
  drift_type TEXT,
  risk REAL,
  urgency REAL,
  impact REAL,
  evidence TEXT,
  verified_commit TEXT,
  event TEXT,
  computed_at TEXT,
  UNIQUE(repo, branch, commit_sha, section_id)
);
"""

INDEX_LOOKUP = """
CREATE INDEX IF NOT EXISTS idx_doc_state_lookup
ON doc_state (repo, branch, seq);
"""

INDEX_SECTION = """
CREATE INDEX IF NOT EXISTS idx_doc_state_section
ON doc_state (section_id);
"""

INDEX_COMMIT = """
CREATE INDEX IF NOT EXISTS idx_doc_state_commit
ON doc_state (repo, branch, commit_sha);
"""

# In-memory databases use a shared-cache URI so all connections see one DB
# (plain ":memory:" is private per connection). A process-level keep-alive
# connection keeps the shared DB alive between calls.
_MEMORY_URI = "file:fd27_shared_memory?mode=memory&cache=shared"
_memory_keepalive: sqlite3.Connection | None = None


def _is_memory(db_path: str | Path) -> bool:
    return str(db_path) == ":memory:"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with row factory enabled."""
    if _is_memory(db_path):
        global _memory_keepalive
        if _memory_keepalive is None:
            _memory_keepalive = sqlite3.connect(_MEMORY_URI, uri=True)
            _memory_keepalive.row_factory = sqlite3.Row
        conn = sqlite3.connect(_MEMORY_URI, uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    db_path_obj = Path(db_path)
    db_path_obj.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path_obj))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path) -> None:
    """Initialize SQLite database schema and indices."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(DDL_DOC_STATE)
        cursor.execute(INDEX_LOOKUP)
        cursor.execute(INDEX_SECTION)
        cursor.execute(INDEX_COMMIT)
        conn.commit()


def _serialize_evidence(evidence: Any) -> str | None:
    if evidence is None:
        return None
    if isinstance(evidence, (dict, list)):
        return json.dumps(evidence, ensure_ascii=False)
    return str(evidence)


def _deserialize_evidence(evidence: str | None) -> Any:
    if evidence is None:
        return None
    trimmed = evidence.strip()
    if (trimmed.startswith("{") and trimmed.endswith("}")) or (
        trimmed.startswith("[") and trimmed.endswith("]")
    ):
        try:
            return json.loads(evidence)
        except Exception:
            return evidence
    return evidence


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["evidence"] = _deserialize_evidence(data.get("evidence"))
    return data


def _get_next_seq(cursor: sqlite3.Cursor, repo: str, branch: str) -> int:
    cursor.execute(
        "SELECT COALESCE(MAX(seq), 0) + 1 FROM doc_state WHERE repo = ? AND branch = ?",
        (repo, branch),
    )
    row = cursor.fetchone()
    return int(row[0]) if row and row[0] is not None else 1


def insert(db_path: str | Path, record: dict[str, Any]) -> int:
    """Insert a single record into doc_state.

    If seq is not provided in record, it is automatically computed as MAX(seq) + 1
    for the (repo, branch). Duplicate (repo, branch, commit_sha, section_id)
    raises sqlite3.IntegrityError (UNIQUE in the DDL); upsert semantics live
    in verify()/merge_branch().
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        repo = record["repo"]
        branch = record["branch"]
        seq = record.get("seq")
        if seq is None:
            seq = _get_next_seq(cursor, repo, branch)

        evidence_str = _serialize_evidence(record.get("evidence"))
        computed_at = record.get("computed_at") or _iso_now()

        cursor.execute(
            """
            INSERT INTO doc_state (
                seq, repo, branch, commit_sha, build_id, doc_path,
                section_id, symbol, p_stale, drift_type, risk,
                urgency, impact, evidence, verified_commit, event, computed_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                seq,
                repo,
                branch,
                record["commit_sha"],
                record.get("build_id"),
                record.get("doc_path"),
                record["section_id"],
                record.get("symbol"),
                record.get("p_stale"),
                record.get("drift_type"),
                record.get("risk"),
                record.get("urgency"),
                record.get("impact"),
                evidence_str,
                record.get("verified_commit"),
                record.get("event"),
                computed_at,
            ),
        )
        row_id = cursor.lastrowid
        conn.commit()
        return row_id  # type: ignore[return-value]


def insert_batch(db_path: str | Path, records: list[dict[str, Any]]) -> list[int]:
    """Insert multiple records into doc_state within a single transaction."""
    if not records:
        return []
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        inserted_ids = []
        for record in records:
            repo = record["repo"]
            branch = record["branch"]
            seq = record.get("seq")
            if seq is None:
                seq = _get_next_seq(cursor, repo, branch)

            evidence_str = _serialize_evidence(record.get("evidence"))
            computed_at = record.get("computed_at") or _iso_now()

            cursor.execute(
                """
                INSERT INTO doc_state (
                    seq, repo, branch, commit_sha, build_id, doc_path,
                    section_id, symbol, p_stale, drift_type, risk,
                    urgency, impact, evidence, verified_commit, event, computed_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    seq,
                    repo,
                    branch,
                    record["commit_sha"],
                    record.get("build_id"),
                    record.get("doc_path"),
                    record["section_id"],
                    record.get("symbol"),
                    record.get("p_stale"),
                    record.get("drift_type"),
                    record.get("risk"),
                    record.get("urgency"),
                    record.get("impact"),
                    evidence_str,
                    record.get("verified_commit"),
                    record.get("event"),
                    computed_at,
                ),
            )
            inserted_ids.append(cursor.lastrowid)
        conn.commit()
        return inserted_ids  # type: ignore[return-value]


def latest(
    db_path: str | Path,
    repo: str,
    branch: str,
    commit: str | None = None,
    path: str | None = None,
    min_risk: float | None = None,
) -> list[dict[str, Any]]:
    """Return the latest state of each section on a branch up to an optional commit.

    If commit is provided, only states with seq <= max seq of that commit on this branch
    are considered. A commit with no rows in the registry yields [] (unknown cutoff);
    replay (FD-29) must pass either a recorded commit or None for a branch-tip snapshot.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        cutoff_seq: int | None = None
        if commit is not None:
            cursor.execute(
                "SELECT MAX(seq) FROM doc_state WHERE repo = ? AND branch = ? AND commit_sha = ?",
                (repo, branch, commit),
            )
            row = cursor.fetchone()
            if row is None or row[0] is None:
                return []
            cutoff_seq = int(row[0])

        query = """
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY section_id
                       ORDER BY seq DESC, id DESC
                   ) as rn
            FROM doc_state
            WHERE repo = ? AND branch = ?
              AND (? IS NULL OR seq <= ?)
              AND (? IS NULL OR doc_path = ? OR doc_path LIKE ?)
        )
        SELECT * FROM ranked
        WHERE rn = 1
          AND (? IS NULL OR risk >= ?)
        ORDER BY risk DESC, section_id ASC;
        """
        path_pattern = f"{path}%" if path else None
        cursor.execute(
            query,
            (
                repo,
                branch,
                cutoff_seq,
                cutoff_seq,
                path,
                path,
                path_pattern,
                min_risk,
                min_risk,
            ),
        )
        rows = cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


def history(
    db_path: str | Path,
    section_id: str,
    repo: str | None = None,
    branch: str | None = None,
) -> list[dict[str, Any]]:
    """Return the full history of a section, ordered chronologically."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        query = """
        SELECT * FROM doc_state
        WHERE section_id = ?
          AND (? IS NULL OR repo = ?)
          AND (? IS NULL OR branch = ?)
        ORDER BY seq ASC, id ASC;
        """
        cursor.execute(query, (section_id, repo, repo, branch, branch))
        rows = cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


def verify(
    db_path: str | Path,
    repo: str,
    branch: str,
    section_id: str,
    commit: str,
    verified_by: str | None = None,
    computed_at: str | None = None,
) -> dict[str, Any]:
    """Record verification of a documentation section, resetting its risk to 0.

    Idempotent on the natural key (repo, branch, commit_sha, section_id):
    a repeated verify of the same commit updates the existing row.
    The conflict branch resets risk/urgency/p_stale and refreshes
    verified_commit/event/computed_at; drift_type/evidence from a previous
    scorer row are kept as-is. Static fields (doc_path, symbol, impact) are
    inherited from the section's latest row; for a never-seen section_id
    they stay None (dashboard fallback parsing, if needed, belongs to FD-30).
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Find latest known state of this section to inherit static metadata
        cursor.execute(
            """
            SELECT * FROM doc_state
            WHERE repo = ? AND branch = ? AND section_id = ?
            ORDER BY seq DESC, id DESC
            LIMIT 1
            """,
            (repo, branch, section_id),
        )
        prev_row = cursor.fetchone()
        doc_path = prev_row["doc_path"] if prev_row else None
        symbol = prev_row["symbol"] if prev_row else None
        impact = prev_row["impact"] if prev_row else None

        seq = _get_next_seq(cursor, repo, branch)
        ts = computed_at or _iso_now()
        evidence_dict = {"verified_by": verified_by} if verified_by else None
        evidence_str = _serialize_evidence(evidence_dict)

        cursor.execute(
            """
            INSERT INTO doc_state (
                seq, repo, branch, commit_sha, doc_path,
                section_id, symbol, p_stale, drift_type, risk,
                urgency, impact, evidence, verified_commit, event, computed_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(repo, branch, commit_sha, section_id) DO UPDATE SET
                risk = 0.0,
                urgency = 0.0,
                p_stale = 0.0,
                verified_commit = excluded.verified_commit,
                event = excluded.event,
                computed_at = excluded.computed_at
            """,
            (
                seq,
                repo,
                branch,
                commit,
                doc_path,
                section_id,
                symbol,
                0.0,
                None,
                0.0,
                0.0,
                impact,
                evidence_str,
                commit,
                "verify",
                ts,
            ),
        )
        row_id = cursor.lastrowid
        conn.commit()

        # Read back by the natural key, which is stable for both
        # the plain-INSERT and the ON CONFLICT DO UPDATE paths.
        cursor.execute(
            "SELECT * FROM doc_state WHERE repo = ? AND branch = ? AND commit_sha = ? AND section_id = ?",
            (repo, branch, commit, section_id),
        )
        inserted = cursor.fetchone()
        assert inserted is not None
        return _row_to_dict(inserted)


def merge_branch(
    db_path: str | Path,
    repo: str,
    source_branch: str,
    target_branch: str,
    merge_commit_sha: str,
    computed_at: str | None = None,
) -> list[dict[str, Any]]:
    """Merge states from source_branch into target_branch.

    Per R12: for each section, creates a record on target_branch at merge_commit_sha
    with the maximum risk across the two branches. Not idempotent: repeating with
    the same merge_commit_sha raises sqlite3.IntegrityError; replay restarts
    (FD-29) own the conflict policy.
    """
    latest_target = {
        row["section_id"]: row
        for row in latest(db_path, repo=repo, branch=target_branch)
    }
    latest_source = {
        row["section_id"]: row
        for row in latest(db_path, repo=repo, branch=source_branch)
    }

    all_section_ids = sorted(set(latest_target.keys()) | set(latest_source.keys()))
    if not all_section_ids:
        return []

    records_to_insert: list[dict[str, Any]] = []
    ts = computed_at or _iso_now()

    for sec_id in all_section_ids:
        t_row = latest_target.get(sec_id)
        s_row = latest_source.get(sec_id)

        if t_row and s_row:
            t_risk = t_row.get("risk") if t_row.get("risk") is not None else -1.0
            s_risk = s_row.get("risk") if s_row.get("risk") is not None else -1.0
            chosen = t_row if t_risk >= s_risk else s_row
        elif t_row:
            chosen = t_row
        else:
            chosen = s_row  # type: ignore[assignment]

        record = {
            "repo": repo,
            "branch": target_branch,
            "commit_sha": merge_commit_sha,
            "build_id": chosen.get("build_id"),
            "doc_path": chosen.get("doc_path"),
            "section_id": sec_id,
            "symbol": chosen.get("symbol"),
            "p_stale": chosen.get("p_stale"),
            "drift_type": chosen.get("drift_type"),
            "risk": chosen.get("risk"),
            "urgency": chosen.get("urgency"),
            "impact": chosen.get("impact"),
            "evidence": chosen.get("evidence"),
            "verified_commit": chosen.get("verified_commit"),
            "event": "merge",
            "computed_at": ts,
        }
        records_to_insert.append(record)

    insert_batch(db_path, records_to_insert)
    return latest(
        db_path, repo=repo, branch=target_branch, commit=merge_commit_sha
    )
