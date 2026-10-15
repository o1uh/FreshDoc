"""FD-12: чистая логика сборки dataset_v1 (R1, R8).

Функции без I/O: склейка и дедуп, пересчёт sig_type, сплиты с изоляцией
проектов, веса. Используются скриптом scripts/build_dataset.py и покрыты
тестами tests/test_build_dataset.py на фиктивных данных.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from src.data.drift import derive_drift_type, extract_signature
from src.data.normalizer import parse_function_ast

REAL_SOURCES = ("codocbench", "history_neg")
SYNTH_SOURCES = ("synth_ast", "synth_llm")

TRAIN_PROJECT_CUTOFF = 80
SYNTH_TRAIN_CUTOFF = 90

WEIGHTS = {
    "real_coupled": 1.0,
    "real_strong_neg": 1.0,
    "real_weak_neg": 0.5,
    "ast": 1.0,
    # P1 снижен решением команды: аудит показал 56-60% шума (тонкие docstring,
    # класс C по R9); файл-источник не меняется, перевес только в сборке.
    "llm_p1": 0.5,
    "llm_p2": 0.5,
}

# Порядок склейки = приоритет при дедупе pair_id (первый побеждает).
SOURCE_PRIORITY = ("codocbench", "history_neg", "synth_ast", "synth_llm")


def bucket100(key: str) -> int:
    """Детерминированная корзина 0-99 (sha256, без salt интерпретатора)."""
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 100


def compute_val_projects(pool_of: dict[str, str]) -> set[str]:
    """Val-проекты: train_pool с хешем >= 80. Только по пулам, без данных."""
    return {p for p, pl in pool_of.items() if pl == "train_pool" and bucket100(p) >= TRAIN_PROJECT_CUTOFF}


def assert_known_projects(projects, pool_of: dict[str, str]) -> None:
    """Строгий контроль: каждый проект обязан быть в project_pools."""
    unknown = sorted(set(map(str, projects)) - set(pool_of))
    if unknown:
        raise ValueError(f"Проекты без пула: {unknown}")


def dedup_by_pair_id(frames_in_priority_order) -> tuple[pd.DataFrame, int, int, int]:
    """Склейка + дедуп pair_id. Возвращает (таблица, было, снято, дублей-строк)."""
    df_all = pd.concat(frames_in_priority_order, ignore_index=True)
    before = len(df_all)
    n_dup_rows = int(df_all.duplicated(subset=["pair_id"], keep=False).sum())
    df_all = df_all.drop_duplicates(subset=["pair_id"], keep="first").reset_index(drop=True)
    return df_all, before, before - len(df_all), n_dup_rows


def drop_val_project_synthetic(
    df_all: pd.DataFrame, val_projects: set[str]
) -> tuple[pd.DataFrame, int]:
    """Полное исключение синтетики val-проектов (R8: проект не в нескольких
    сплитах). Val остаётся только реальным; test_synth — без val-проектов."""
    mask = (
        df_all["source"].isin(SYNTH_SOURCES)
        & df_all["project"].astype(str).isin(set(val_projects))
    )
    return df_all.loc[~mask].reset_index(drop=True), int(mask.sum())


def recompute_sig_type(code_old: str, code_new: str, fallback: str) -> tuple[str, bool]:
    """Тип по сигнатурам R2. (значение, удалось_ли_разобрать)."""
    try:
        fn_old = parse_function_ast(code_old)
        fn_new = parse_function_ast(code_new)
        if fn_old is None or fn_new is None:
            return fallback, False
        s_old = extract_signature(fn_old)
        s_new = extract_signature(fn_new)
        return derive_drift_type(s_old, s_new), True
    except Exception:
        return fallback, False


def recompute_sig_column(df_all: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Пересчёт sig_type для всех строк. Возвращает (таблица, ok, расхождений)."""
    col_old = df_all["code_old"].astype(str).tolist()
    col_new = df_all["code_new"].astype(str).tolist()
    col_sig = df_all["sig_type"].astype(str).tolist()
    out: list[str] = [""] * len(df_all)
    ok, mismatch = 0, 0
    for i, (co, cn, st) in enumerate(zip(col_old, col_new, col_sig)):
        v, good = recompute_sig_type(co, cn, st)
        out[i] = v
        ok += int(good)
        mismatch += int(v != st)
    df_all = df_all.copy()
    df_all["sig_type"] = out
    return df_all, ok, mismatch


def assign_splits(
    df_all: pd.DataFrame, pool_of: dict[str, str], val_projects: set[str]
) -> pd.Series:
    """Назначение сплитов R1: test_pool -> test_real; train_pool -> train/val
    по проекту; синтетика -> train/test_synth по function_hash."""
    splits: list[str] = []
    for _, r in df_all.iterrows():
        src, proj = str(r["source"]), str(r["project"])
        if src in REAL_SOURCES:
            if pool_of[proj] == "test_pool":
                splits.append("test_real")
            else:
                splits.append("val" if proj in val_projects else "train")
        else:
            b = bucket100(str(r["function_hash"]))
            splits.append("train" if b < SYNTH_TRAIN_CUTOFF else "test_synth")
    return pd.Series(splits, index=df_all.index)


def apply_weights(df_all: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Принудительные веса R1 (+llm_p1=0.5). Возвращает (таблица, исправлено)."""
    df_all = df_all.copy()
    fixed = 0
    for label_source, w in WEIGHTS.items():
        m = df_all["label_source"] == label_source
        if m.any():
            bad = m & (df_all["weight"] != w)
            fixed += int(bad.sum())
            df_all.loc[m, "weight"] = w
    return df_all, fixed


def drop_leaked_test_synth(df_all: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Исключение из test_synth функций, встречающихся в реальных строках.

    Иначе модель видит функцию в train (реальная строка), а оценивается на
    её синтетических мутациях в test_synth: sig_old/doc_old совпадают.
    """
    real_hashes = set(df_all.loc[df_all["source"].isin(REAL_SOURCES), "function_hash"].astype(str))
    mask = df_all["split"].eq("test_synth") & df_all["function_hash"].astype(str).isin(real_hashes)
    return df_all.loc[~mask].reset_index(drop=True), int(mask.sum())


def hash_overlap(df_all: pd.DataFrame, split_a: str, split_b: str) -> set[str]:
    """Общие function_hash двух сплитов (для assert изоляции)."""
    a = set(df_all.loc[df_all["split"] == split_a, "function_hash"])
    b = set(df_all.loc[df_all["split"] == split_b, "function_hash"])
    return a & b


def projects_of(df_all: pd.DataFrame, split: str, sources=None) -> set[str]:
    """Проекты сплита (опционально фильтр по источникам)."""
    sub = df_all.loc[df_all["split"] == split]
    if sources is not None:
        sub = sub.loc[sub["source"].isin(sources)]
    return set(sub["project"].astype(str).unique())
