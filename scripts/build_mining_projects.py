"""Build config/mining_projects.yaml for FD-05.

Top-60 projects by raw record count in CoDocBench, with clone URL
(from projects.csv) and pool (from project_pools.parquet, FD-04 logic).

Resolved ambiguities (see docs/data_notes.md, FD-05):
  - project 'manim' -> ManimCommunity/manim, pool test_pool
    (3b1b/manim shares old history; 2024 commits exist only in ManimCommunity).
  - project 'Python' -> TheAlgorithms/Python, pool train_pool
    (paths data_structures/... match; SHA confirmed via GitHub API).

Usage:
    python scripts/build_mining_projects.py [--top_n 60]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "external" / "codocbench" / "dataset"
POOLS_PATH = PROJECT_ROOT / "data" / "interim" / "project_pools.parquet"
PROJECTS_CSV = PROJECT_ROOT / "data" / "external" / "codocbench" / "projects.csv"
OUT_PATH = PROJECT_ROOT / "config" / "mining_projects.yaml"

# Manual overrides for duplicate repo names in projects.csv.
URL_OVERRIDE = {
    "manim": "https://github.com/ManimCommunity/manim.git",
    "python": "https://github.com/TheAlgorithms/Python.git",
}


def count_records() -> tuple[collections.Counter, dict[str, set[str]]]:
    counts: collections.Counter = collections.Counter()
    raw_pools: dict[str, set[str]] = {}
    for fname, pool in (("train.jsonl", "train"), ("test.jsonl", "test")):
        with open(DATASET_DIR / fname, encoding="utf-8") as fh:
            for line in fh:
                project = json.loads(line)["project"]
                counts[project] += 1
                raw_pools.setdefault(project, set()).add(pool)
    return counts, raw_pools


def load_pools() -> dict[str, str]:
    import pandas as pd

    df = pd.read_parquet(POOLS_PATH)
    pools = dict(zip(df["project"], df["pool"]))
    # 'train_pool' -> 'train', 'test_pool' -> 'test' (FD-05 short form).
    return {p: ("train" if v == "train_pool" else "test") for p, v in pools.items()}


def load_urls() -> dict[str, str]:
    urls: dict[str, list[str]] = collections.defaultdict(list)
    with open(PROJECTS_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            org = row["organization"].strip()
            repo = row["repo"].strip()
            urls[repo.lower()].append(f"https://github.com/{org}/{repo}.git")
    resolved: dict[str, str] = {}
    for key, candidates in urls.items():
        if key in URL_OVERRIDE:
            resolved[key] = URL_OVERRIDE[key]
        elif len(candidates) == 1:
            resolved[key] = candidates[0]
        else:
            raise ValueError(f"Ambiguous repo name without override: {key} -> {candidates}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mining_projects.yaml (FD-05, step 1).")
    parser.add_argument("--top_n", type=int, default=60)
    args = parser.parse_args()

    counts, raw_pools = count_records()
    pools = load_pools()
    urls = load_urls()

    entries = []
    missing_pool = []
    for project, n in counts.most_common(args.top_n):
        pool = pools.get(project)
        if pool is None:
            # Project fully filtered out in FD-04; fall back to raw membership.
            raw = raw_pools[project]
            pool = "test" if raw == {"test"} else "train" if raw == {"train"} else "test"
            if project == "manim":
                pool = "test"
            missing_pool.append(project)
        entries.append(
            {"project": project, "url": urls[project.lower()], "pool": pool, "records": n}
        )

    config = {
        "budget_minutes": 240,
        "max_commits": 1000,
        "pilot_commits": 150,
        "projects": entries,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, allow_unicode=True, sort_keys=False)

    n_train = sum(1 for e in entries if e["pool"] == "train")
    print(f"Wrote {len(entries)} projects to {OUT_PATH} (train={n_train}, test={len(entries) - n_train})")
    print(f"Records covered: {sum(e['records'] for e in entries)} of {sum(counts.values())}")
    if missing_pool:
        print(f"Pool fallback (absent in FD-04 pools): {missing_pool}")


if __name__ == "__main__":
    main()
