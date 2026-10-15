from __future__ import annotations

import hashlib
import re
from typing import Dict, Set, Tuple

import pandas as pd

from src.data.normalizer import (
    extract_docstring_and_clean_node,
    normalize_code_from_node,
    parse_function_ast,
)

TRAIN_POOL = "train_pool"
TEST_POOL = "test_pool"

REQUIRED_PAIR_COLS = {"project", "split_pool", "function_hash", "_aux_src_old"}
REQUIRED_POOL_COLS = {"project", "pool"}
REQUIRED_NEG_COLS = {"project", "src_old", "doc_changed"}


def _passes_r1(name: str, doc: str, n_lines: int) -> bool:
    if name.startswith("_"):
        return False
    if len(doc) < 40:
        return False
    if not (4 <= n_lines <= 300):
        return False
    return True


def _is_test_path(path: str) -> bool:
    # Тестовые файлы и каталоги: test/, tests/, *_test.py, test_*.py,
    # conftest.py, каталоги вида llm_tests. Якорные правила, чтобы не
    # зацепить latest.py / contest.py (проверяется началом/концом, не подстрокой).
    parts = re.split(r"[\\/]", (path or "").lower())
    for i, p in enumerate(parts):
        if p in ("test", "tests", "testing"):
            return True
        if p.startswith(("test_", "test-")) or p.endswith(("_test", "_tests", "-test", "-tests")):
            return True
        if i == len(parts) - 1:
            stem = p[:-3] if p.endswith(".py") else p
            if stem.startswith("test_") or stem.endswith(("_test", "_tests")):
                return True
            if stem == "conftest":
                return True
    return False


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


def make_function_hash(code_old_norm: str) -> str:
    return hashlib.sha256(code_old_norm.encode("utf-8")).hexdigest()[:16]


def _train_test_projects(df_pools: pd.DataFrame) -> Tuple[Set[str], Set[str]]:
    missing_pools = REQUIRED_POOL_COLS - set(df_pools.columns)
    if missing_pools:
        raise ValueError(f"project_pools: нет колонок {sorted(missing_pools)}")
    train_projects = set(df_pools[df_pools["pool"] == TRAIN_POOL]["project"])
    test_projects = set(df_pools[df_pools["pool"] == TEST_POOL]["project"])
    overlap = train_projects & test_projects
    if overlap:
        raise ValueError(f"Пулы проектов пересекаются: {sorted(overlap)}")
    return train_projects, test_projects


def build_functions_pool_v2(
    df_pool_v1: pd.DataFrame,
    df_negatives: pd.DataFrame,
    df_pools: pd.DataFrame,
    exclude_tests: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    missing_v1 = {"project", "src", "function_hash"} - set(df_pool_v1.columns)
    if missing_v1:
        raise ValueError(f"functions_pool_v1: нет колонок {sorted(missing_v1)}")
    missing_neg = REQUIRED_NEG_COLS - set(df_negatives.columns)
    if missing_neg:
        raise ValueError(f"negatives_raw: нет колонок {sorted(missing_neg)}")

    train_projects, test_projects = _train_test_projects(df_pools)

    stats: Dict[str, int] = {
        "v1_вход": len(df_pool_v1),
        "neg_вход": len(df_negatives),
        "neg_не_train_pool": 0,
        "neg_doc_changed": 0,
        "утечка_test_project": 0,
        "дубликаты_hash_внутри_источника2": 0,
        "дубликаты_против_v1": 0,
        "не_разобралось": 0,
        "не_прошёл_фильтры": 0,
        "отсев_test_функции": 0,
        "отсев_test_файлы": 0,
        "источник2_успех": 0,
        "успех_v2": 0,
    }

    v1_hashes = set(df_pool_v1["function_hash"].astype(str))
    seen: Set[str] = set()
    has_file = "file" in df_negatives.columns

    new_rows = []
    for _, r in df_negatives.iterrows():
        project = str(r["project"])
        if project not in train_projects:
            stats["neg_не_train_pool"] += 1
            continue
        doc_changed = r["doc_changed"]
        if isinstance(doc_changed, str):
            changed = doc_changed.strip().lower() in ("true", "1", "yes")
        else:
            changed = bool(doc_changed)
        if changed:
            stats["neg_doc_changed"] += 1
            continue
        if project in test_projects:
            stats["утечка_test_project"] += 1
            continue
        if has_file and _is_test_path(r["file"] if isinstance(r["file"], str) else ""):
            stats["отсев_test_файлы"] += 1
            continue
        src = r["src_old"]
        if not isinstance(src, str) or not src.strip():
            stats["не_разобралось"] += 1
            continue
        fn = parse_function_ast(src)
        if fn is None:
            stats["не_разобралось"] += 1
            continue
        if exclude_tests and fn.name.startswith("test_"):
            stats["отсев_test_функции"] += 1
            continue
        doc, fn_no_doc = extract_docstring_and_clean_node(fn)
        doc = (doc or "").strip()
        n_lines = len(src.strip().splitlines())
        if not _passes_r1(fn.name, doc, n_lines):
            stats["не_прошёл_фильтры"] += 1
            continue
        code_norm = normalize_code_from_node(fn_no_doc)
        fhash = make_function_hash(code_norm)
        if fhash in v1_hashes:
            stats["дубликаты_против_v1"] += 1
            continue
        if fhash in seen:
            stats["дубликаты_hash_внутри_источника2"] += 1
            continue
        seen.add(fhash)
        new_rows.append(
            {"project": project, "src": src.strip(), "function_hash": fhash}
        )

    stats["источник2_успех"] = len(new_rows)

    df_new = pd.DataFrame(new_rows, columns=["project", "src", "function_hash"])
    df_v2 = pd.concat([df_pool_v1, df_new], ignore_index=True)

    leak = set(df_v2["project"].astype(str)) & test_projects
    stats["утечка_test_project"] += len(leak)
    if leak:
        raise ValueError(
            f"Найдены test-проекты в пуле v2: {sorted(leak)}"
        )
    if not df_v2["function_hash"].is_unique:
        raise ValueError("Дубликаты function_hash в пуле v2!")

    stats["успех_v2"] = len(df_v2)
    return df_v2, stats
