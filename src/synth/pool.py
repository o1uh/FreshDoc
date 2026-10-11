from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from src.data.normalizer import (
    extract_docstring_and_clean_node,
    parse_function_ast,
)

TRAIN_POOL = "train_pool"
TEST_POOL = "test_pool"

REQUIRED_PAIR_COLS = {"project", "split_pool", "function_hash", "_aux_src_old"}
REQUIRED_POOL_COLS = {"project", "pool"}


def _passes_r1(name: str, doc: str, n_lines: int) -> bool:
    if name.startswith("_"):
        return False
    if len(doc) < 40:
        return False
    if not (4 <= n_lines <= 300):
        return False
    return True


def build_functions_pool(
    df_pairs: pd.DataFrame,
    df_pools: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    missing = REQUIRED_PAIR_COLS - set(df_pairs.columns)
    if missing:
        raise ValueError(f"pairs_codocbench: нет колонок {sorted(missing)}")
    missing_pools = REQUIRED_POOL_COLS - set(df_pools.columns)
    if missing_pools:
        raise ValueError(f"project_pools: нет колонок {sorted(missing_pools)}")

    train_projects = set(df_pools[df_pools["pool"] == TRAIN_POOL]["project"])
    test_projects = set(df_pools[df_pools["pool"] == TEST_POOL]["project"])
    overlap = train_projects & test_projects
    if overlap:
        raise ValueError(f"Пулы проектов пересекаются: {sorted(overlap)}")

    stats: Dict[str, int] = {
        "вход_train_rows": 0,
        "утечка_test_project": 0,
        "дубликаты_hash": 0,
        "не_разобралось": 0,
        "не_прошёл_фильтры": 0,
        "успех": 0,
    }

    sub = df_pairs[df_pairs["split_pool"] == TRAIN_POOL].copy()
    stats["вход_train_rows"] = len(sub)

    # Изоляция: проект из test-пула не должен попасть в пул (защита, не фильтр умолчания).
    leak_mask = sub["project"].isin(test_projects)
    stats["утечка_test_project"] = int(leak_mask.sum())
    if stats["утечка_test_project"]:
        raise ValueError(
            f"Найдены train-строки из test-проектов: "
            f"{sorted(sub[leak_mask]['project'].unique())}"
        )

    before_dedup = len(sub)
    sub = sub.drop_duplicates(subset=["function_hash"], keep="first")
    stats["дубликаты_hash"] = before_dedup - len(sub)

    rows = []
    for _, r in sub.iterrows():
        src = r["_aux_src_old"]
        if not isinstance(src, str) or not src.strip():
            stats["не_разобралось"] += 1
            continue
        fn = parse_function_ast(src)
        if fn is None:
            stats["не_разобралось"] += 1
            continue
        doc, _ = extract_docstring_and_clean_node(fn)
        doc = (doc or "").strip()
        n_lines = len(src.strip().splitlines())
        if not _passes_r1(fn.name, doc, n_lines):
            stats["не_прошёл_фильтры"] += 1
            continue
        rows.append(
            {
                "project": str(r["project"]),
                "src": src.strip(),
                "function_hash": str(r["function_hash"]),
            }
        )

    stats["успех"] = len(rows)
    df_pool = pd.DataFrame(rows, columns=["project", "src", "function_hash"])
    return df_pool, stats
