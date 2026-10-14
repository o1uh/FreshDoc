"""Pilot and full git-history mining for FD-05 (steps 4-6, R3).

Pilot (before the full run):
    python scripts/mine_negatives.py --mode pilot [--workers 4]
  Mines ``pilot_commits`` (150) per cloned project into
  ``data/interim/mining_pilot/<project>.parquet`` and writes
  ``reports/mining_pilot.json`` with sec/commit, rows/commit and the
  R3 project selection within ``budget_minutes``.

Full run (overnight-friendly, resumable):
    python scripts/mine_negatives.py --mode full [--workers 2]
  Mines ``max_commits`` per chosen project into
  ``data/interim/mining/<project>.parquet`` (existing files skipped),
  then concatenates everything into ``data/interim/negatives_raw.parquet``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "mining_projects.yaml"
REPOS_DIR = PROJECT_ROOT / "data" / "repos"
PILOT_DIR = PROJECT_ROOT / "data" / "interim" / "mining_pilot"
MINING_DIR = PROJECT_ROOT / "data" / "interim" / "mining"
PILOT_JSON = PROJECT_ROOT / "reports" / "mining_pilot.json"
NEGATIVES_RAW = PROJECT_ROOT / "data" / "interim" / "negatives_raw.parquet"

from src.mining.miner import mine_project  # noqa: E402


def repo_dir(url: str) -> Path:
    m = re.search(r"github\.com/([^/]+)/([^/]+?)(?:\.git)?$", url.strip())
    return REPOS_DIR / f"{m.group(1)}__{m.group(2)}"


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_pilot(workers: int) -> dict:
    cfg = load_config()
    pilot_commits = cfg.get("pilot_commits", 150)
    max_commits = cfg.get("max_commits", 1000)
    budget_s = cfg.get("budget_minutes", 240) * 60

    jobs = [(p["project"], repo_dir(p["url"])) for p in cfg["projects"]]
    stats_list: list[dict] = []

    def one(job):
        project, repo = job
        if not (repo / ".git").exists():
            return {"project": project, "cloned": False}
        st = mine_project(repo, project, max_commits=pilot_commits,
                          out_path=PILOT_DIR / f"{project}.parquet")
        st["cloned"] = True
        return st

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for st in pool.map(one, jobs):
            stats_list.append(st)
            if st.get("cloned"):
                n = st["commits_processed"]
                spc = st["elapsed_s"] / n if n else 0.0
                rpc = st["candidates"] / n if n else 0.0
                print(f"[{st['project']}] commits={n} candidates={st['candidates']} "
                      f"sec/commit={spc:.2f} rows/commit={rpc:.3f}", flush=True)
            else:
                print(f"[{st['project']}] NOT CLONED, skipped", flush=True)

    # R3 selection: rank by rows_per_commit / sec_per_commit, fill budget.
    ranked = []
    for st in stats_list:
        if not st.get("cloned"):
            continue
        n = st["commits_processed"]
        spc = st["elapsed_s"] / n if n else float("inf")
        rpc = st["candidates"] / n if n else 0.0
        ranked.append({**st, "sec_per_commit": round(spc, 3),
                       "rows_per_commit": round(rpc, 4)})
    ranked.sort(key=lambda s: (s["rows_per_commit"] / s["sec_per_commit"]
                               if s["sec_per_commit"] else 0.0), reverse=True)
    chosen, total = [], 0.0
    for st in ranked:
        est = st["sec_per_commit"] * max_commits
        if chosen and total + est > budget_s:
            continue
        chosen.append(st["project"])
        total += est
    report = {
        "pilot_commits": pilot_commits,
        "max_commits": max_commits,
        "budget_minutes": cfg.get("budget_minutes", 240),
        "estimated_full_s": round(total, 1),
        "chosen": chosen,
        "projects": ranked,
    }
    PILOT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PILOT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"Chosen {len(chosen)}/{len(ranked)} projects, "
          f"estimated full time {total / 60:.0f} min -> {PILOT_JSON}")
    return report


def run_full(workers: int) -> None:
    import pandas as pd

    cfg = load_config()
    max_commits = cfg.get("max_commits", 1000)
    try:
        with open(PILOT_JSON, encoding="utf-8") as fh:
            chosen = set(json.load(fh)["chosen"])
    except FileNotFoundError:
        chosen = None
    url_of = {p["project"]: p["url"] for p in cfg["projects"]}

    def one(project):
        out = MINING_DIR / f"{project}.parquet"
        if out.exists():
            return {"project": project, "resumed": True}
        repo = repo_dir(url_of[project])
        st = mine_project(repo, project, max_commits=max_commits, out_path=out)
        print(f"[{project}] commits={st['commits_processed']} "
              f"candidates={st['candidates']} errors={st['commit_errors']} "
              f"elapsed={st['elapsed_s']}s truncated={st['truncated']}", flush=True)
        return st

    targets = [p for p in url_of if chosen is None or p in chosen]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, targets))

    frames = [pd.read_parquet(f) for f in sorted(MINING_DIR.glob("*.parquet"))]
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    NEGATIVES_RAW.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(NEGATIVES_RAW, index=False)
    print(f"negatives_raw: {len(raw)} rows -> {NEGATIVES_RAW}")
    if len(raw):
        print(raw["category"].value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="FD-05 pilot / full mining (R3).")
    parser.add_argument("--mode", choices=["pilot", "full"], required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "pilot":
        run_pilot(args.workers)
    else:
        run_full(args.workers)


if __name__ == "__main__":
    main()
