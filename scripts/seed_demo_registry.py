"""FD-29: deterministic demo seed for the registry DB (no GPU, no network).

`data/registry.db` and `data/cache/` are git-ignored per R15, so a clean
checkout has an empty registry and FD-30 (dashboard) would start blank.
Instead of committing data files, this script rebuilds a small demo state
deterministically: it creates a synthetic git repo with a fixed 4-commit
history and replays it with the M0 rule scorer (no model weights needed).

Usage:
    python scripts/seed_demo_registry.py [--db data/registry.db] [--branch demo]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.registry.replay import M0Scorer, replay_project_history
from src.registry.storage import init_db, latest
from src.risk.config import RiskConfig

FETCH_V1 = '''def fetch(url, timeout=10):
    """Fetch a URL with a timeout in seconds. Defaults to 10 seconds."""
    body = _get(url, timeout)
    return body.decode("utf-8")


def parse(data):
    """Parse raw response data into a list of records for callers."""
    items = data.split(",")
    out = [i.strip() for i in items]
    return out
'''

FETCH_V2 = FETCH_V1.replace(
    "def fetch(url, timeout=10):",
    "def fetch(url, timeout=10, retries=3):",
).replace("body = _get(url, timeout)", "body = _get(url, timeout, retries)")

FETCH_V3 = FETCH_V2.replace(
    '"""Fetch a URL with a timeout in seconds. Defaults to 10 seconds."""',
    '"""Fetch a URL with retries. Timeout defaults to 10, retries to 3."""',
)

PARSE_V4 = FETCH_V3.replace(
    "out = [i.strip() for i in items]",
    "out = [i.strip() for i in items if i.strip()]",
)

# (message, date, file content): fixed dates keep the seed deterministic.
HISTORY = [
    ("Add demo module", "2026-01-05T12:00:00+00:00", FETCH_V1),
    ("Add retries to fetch", "2026-02-05T12:00:00+00:00", FETCH_V2),
    ("Document retries", "2026-03-05T12:00:00+00:00", FETCH_V3),
    ("Tidy parse loop", "2026-04-05T12:00:00+00:00", PARSE_V4),
]


def _git(repo: Path, *args: str, env: dict | None = None) -> None:
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    subprocess.run(
        # -c commit.gpgsign=false: a global `commit.gpgsign = true` without a
        # key would otherwise fail the seed commits (and its test).
        ["git", "-c", "commit.gpgsign=false", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env=full_env,
    )


def build_demo_repo(path: Path | str) -> Path:
    """Create the synthetic demo git repo with a fixed linear history."""
    repo = Path(path)
    if (repo / ".git").exists():
        return repo
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "FreshDoc")
    _git(repo, "config", "user.email", "freshdoc@example.com")
    for msg, date, content in HISTORY:
        (repo / "demo.py").write_text(content, encoding="utf-8")
        _git(repo, "add", "demo.py")
        stamp = {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
        _git(repo, "commit", "-q", "-m", msg, env=stamp)
    return repo


def seed_demo_db(
    db_path: Path | str,
    repo_path: Path | str,
    branch: str = "demo",
    repo_name: str = "freshdoc/demo",
) -> dict:
    """Replay the demo repo with M0 into `branch`; returns summary metrics."""
    db_path = Path(db_path)
    init_db(db_path)
    metrics, _ = replay_project_history(
        repo_path=repo_path,
        repo_name=repo_name,
        branch=branch,
        max_commits=50,
        scorer=M0Scorer(),
        cache=None,
        db_path=db_path,
        risk_config=RiskConfig(threshold=0.5),
    )
    return {
        "branch": branch,
        "fixes": metrics.total_fixes,
        "preceded": metrics.preceded_fixes,
        "sections": len(latest(db_path, repo_name, branch)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FD-29: seed demo registry DB.")
    parser.add_argument("--db", default="data/registry.db")
    parser.add_argument("--branch", default="demo")
    parser.add_argument("--repo-dir", default="data/demo_repo")
    args = parser.parse_args()

    db = PROJECT_ROOT / args.db if not Path(args.db).is_absolute() else Path(args.db)
    repo_dir = (
        PROJECT_ROOT / args.repo_dir
        if not Path(args.repo_dir).is_absolute()
        else Path(args.repo_dir)
    )
    repo = build_demo_repo(repo_dir)
    summary = seed_demo_db(db, repo, branch=args.branch)
    print(f"Demo repo: {repo}")
    print(f"Registry: {db} [{summary['branch']}]")
    print(
        f"fixes={summary['fixes']}, preceded={summary['preceded']}, "
        f"sections={summary['sections']}"
    )


if __name__ == "__main__":
    main()
