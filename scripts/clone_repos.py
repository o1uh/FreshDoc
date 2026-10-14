"""Clone mining projects shallowly for FD-05 (step 2).

Reads config/mining_projects.yaml, clones each URL with
`git clone --depth 1000` into data/repos/<owner>__<repo> using
4 parallel workers and one retry. Failures go to
reports/clone_failures.txt; existing checkouts are skipped.

Usage:
    python scripts/clone_repos.py [--depth 1000] [--workers 4]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import subprocess
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "mining_projects.yaml"
REPOS_DIR = PROJECT_ROOT / "data" / "repos"
FAILURES_PATH = PROJECT_ROOT / "reports" / "clone_failures.txt"


def target_dir(url: str) -> Path:
    m = re.search(r"github\.com/([^/]+)/([^/]+?)(?:\.git)?$", url.strip())
    if not m:
        raise ValueError(f"Cannot parse GitHub URL: {url}")
    return REPOS_DIR / f"{m.group(1)}__{m.group(2)}"


def clone_one(url: str, depth: int) -> str | None:
    dest = target_dir(url)
    if (dest / ".git").exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in (1, 2):
        proc = subprocess.run(
            ["git", "clone", "--depth", str(depth), url, str(dest)],
            capture_output=True,
            text=True,
            timeout=1200,
        )
        if proc.returncode == 0:
            return None
        err = (proc.stderr or "")[-300:]
        if attempt == 2:
            return f"{url} :: {err}"
    return f"{url} :: unreachable"


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone FD-05 mining projects (step 2).")
    parser.add_argument("--depth", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    with open(CONFIG_PATH, encoding="utf-8") as fh:
        projects = yaml.safe_load(fh)["projects"]

    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_url = {
            pool.submit(clone_one, p["url"], args.depth): p for p in projects
        }
        for future in concurrent.futures.as_completed(future_to_url):
            proj = future_to_url[future]
            try:
                error = future.result()
            except Exception as exc:  # noqa: BLE001 - recorded, not raised
                error = f"{proj['url']} :: {exc!r}"
            status = "FAIL" if error else "ok"
            print(f"[{status}] {proj['project']} {proj['url']}", flush=True)
            if error:
                failures.append(error)

    FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILURES_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(failures) + ("\n" if failures else ""))
    print(f"Done: {len(projects) - len(failures)}/{len(projects)} cloned, failures -> {FAILURES_PATH}")


if __name__ == "__main__":
    main()
