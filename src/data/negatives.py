"""FD-06: отбор отрицательных пар из git-истории (R5).

Вход: negatives_raw (R3), положительные пары, таблица project → pool.
Выход: кандидаты в схеме R1 с ролями real_strong_neg / real_weak_neg
и unlabeled_sigchange (label_stale = -1, вес 0.0).

Примечания к схеме:
- split_pool — interim-формат. FD-12 перемаппит его в split по R1
  (реальные test-пула → test_real, train-пул по проектам → train/val).
- sig_type сохраняется из drift_type miner-а: он вычислен в момент
  майнинга теми же extract_signature / derive_drift_type (R2).
  FD-12 / R8 пересчитает sig_type единообразно для всех строк.
- label_source = real_unlabeled и вес 0.0 для unlabeled_sigchange —
  расширения R1, утверждены человеком 14.10.2026 (см. docs/data_notes.md).
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

import pandas as pd

from src.data.codocbench_adapter import make_function_hash

SOURCE = "history_neg"
STRONG = "real_strong_neg"
WEAK = "real_weak_neg"
UNLABELED = "real_unlabeled"

STRONG_FRAC = 0.3

BINS: List[Tuple[str, int, int]] = [
    ("4-8", 4, 8),
    ("9-15", 9, 15),
    ("16-40", 16, 40),
    ("41-300", 41, 300),
]

R1_COLS = [
    "pair_id",
    "source",
    "project",
    "commit_sha",
    "symbol",
    "function_hash",
    "sig_old",
    "code_old",
    "code_new",
    "diff_code",
    "doc_old",
    "label_stale",
    "drift_type",
    "sig_type",
    "label_source",
    "weight",
    "n_src_lines",
    "split_pool",
]


def assign_bin(n: int) -> str:
    for label, lo, hi in BINS:
        if lo <= int(n) <= hi:
            return label
    return "OUT"


def apply_r1_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    m_us = df["name"].astype(str).str.startswith("_")
    m_doc = df["doc_old"].astype(str).str.len() < 40
    m_len = ~df["n_src_lines"].between(4, 300)
    drop = m_us | m_doc | m_len
    stats = {
        "input": int(len(df)),
        "underscore": int(m_us.sum()),
        "doc_short": int(m_doc.sum()),
        "len_out": int(m_len.sum()),
        "dropped": int(drop.sum()),
        "kept": int((~drop).sum()),
    }
    return df.loc[~drop].copy(), stats


EXPECTED_CATEGORIES = {"cosmetic", "body_changed", "signature_changed"}


def make_neg_pair_id(
    project: str,
    symbol: str,
    commit_sha: str,
    code_old: str,
    code_new: str,
    doc_old: str,
) -> str:
    """pair_id для history_neg: включает code_new и doc_old.

    Отличие от FD-04 (только code_old): у вендоринг-дублей один commit
    правит ≥2 копий файла — одинаковый code_old при разном code_new
    (другой дифф) или разном doc_old (другая документация). По старому
    ключу такие варианты схлопывались в один pair_id и терялись (35 строк),
    а 2 id коллизили между cand и unlabeled. Полные дубликаты
    (всё одинаковое) по-прежнему схлопываются — это желательно.
    """
    key = (
        f"{SOURCE}:{project}:{symbol}:{commit_sha}:"
        f"{code_old}\x00{code_new}\x00{doc_old}"
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _enrich(sub: pd.DataFrame, role: str) -> pd.DataFrame:
    sub = sub.copy()
    if role == "strong":
        sub["label_stale"] = 0
        sub["label_source"] = STRONG
        sub["weight"] = 1.0
        sub["drift_type"] = "none"
        sub["sig_type"] = "none"
    elif role == "weak":
        sub["label_stale"] = 0
        sub["label_source"] = WEAK
        sub["weight"] = 0.5
        sub["drift_type"] = "none"
        sub["sig_type"] = "none"
    else:
        sub["label_stale"] = -1
        sub["label_source"] = UNLABELED
        sub["weight"] = 0.0
        sub["sig_type"] = sub["drift_type"].astype(str)
    sub["source"] = SOURCE
    sub["pair_id"] = [
        make_neg_pair_id(str(p), str(s), str(c), str(co), str(cn), str(d))
        for p, s, c, co, cn, d in zip(
            sub["project"],
            sub["symbol"],
            sub["commit_sha"],
            sub["code_old"],
            sub["code_new"],
            sub["doc_old"],
        )
    ]
    sub["function_hash"] = sub["code_old"].astype(str).map(make_function_hash)
    return sub


def _dedup(sub: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], int]:
    dup_mask = sub.duplicated(subset=["pair_id"], keep="first")
    dropped_ids = sorted(sub.loc[dup_mask, "pair_id"].astype(str).unique())
    return sub.loc[~dup_mask].copy(), dropped_ids, int(dup_mask.sum())


def build_r1_rows(
    df: pd.DataFrame, pool_of: Dict[str, str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], List[str]]:
    df = df.copy()
    unknown = sorted(set(df["project"].astype(str)) - set(pool_of))
    if unknown:
        raise ValueError(f"Проекты без пула: {unknown}")
    bad_cat = sorted(set(df["category"].astype(str)) - EXPECTED_CATEGORIES)
    if bad_cat:
        raise ValueError(f"Неизвестные category: {bad_cat}")
    df["split_pool"] = df["project"].astype(str).map(pool_of)
    n_doc_changed = int(df["doc_changed"].astype(bool).sum())
    df = df.loc[~df["doc_changed"].astype(bool)].copy()

    is_sig = df["category"] == "signature_changed"
    df_cand = df.loc[~is_sig].copy()
    df_un = df.loc[is_sig].copy()

    cand = pd.concat(
        [
            _enrich(df_cand[df_cand["category"] == "cosmetic"], "strong"),
            _enrich(df_cand[df_cand["category"] == "body_changed"], "weak"),
        ],
        ignore_index=True,
    )
    df_un = _enrich(df_un, "unlabeled")

    cand, cand_dup_ids, cand_dup_n = _dedup(cand)
    df_un, un_dup_ids, un_dup_n = _dedup(df_un)
    dropped_ids = sorted(set(cand_dup_ids) | set(un_dup_ids))

    # Кросс-уникальность cand/un: при расширенном pair_id (code_new+doc_old)
    # коллизий быть не должно — одинаковые code_old+code_new+doc_old дают
    # одинаковый drift, т.е. не могут лежать в разных категориях.
    # Fail fast для FD-13 (pair_id уникален на сборке FD-12).
    cross_ids = sorted(
        set(cand["pair_id"].astype(str)) & set(df_un["pair_id"].astype(str))
    )
    if cross_ids:
        raise ValueError(f"pair_id в cand и unlabeled одновременно: {cross_ids}")

    stats = {
        "doc_changed_dropped": n_doc_changed,
        "cand_strong": int((cand["label_source"] == STRONG).sum()),
        "cand_weak": int((cand["label_source"] == WEAK).sum()),
        "unlabeled": int(len(df_un)),
        "pair_id_dedup_dropped": int(cand_dup_n + un_dup_n),
        "pair_id_dedup_dropped_unique": int(len(dropped_ids)),
        "cross_pair_id_collisions": 0,
    }
    missing = set(R1_COLS) - set(cand.columns)
    missing_un = set(R1_COLS) - set(df_un.columns)
    if missing or missing_un:
        raise ValueError(
            f"Нет колонок схемы R1: cand={sorted(missing)} un={sorted(missing_un)}"
        )
    return cand[R1_COLS].copy(), df_un[R1_COLS].copy(), stats, dropped_ids


def select_balanced(
    df_pos: pd.DataFrame, df_cand: pd.DataFrame, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pos = df_pos.copy()
    pos["bin"] = pos["n_src_lines"].map(assign_bin)
    cand = df_cand.copy()
    cand["bin"] = cand["n_src_lines"].map(assign_bin)
    cand = cand[cand["bin"] != "OUT"].copy()

    cells_src = pos[pos["bin"] != "OUT"][["split_pool", "bin"]]
    cells = sorted(cells_src.drop_duplicates().values.tolist())
    picked_parts = []
    deficit_rows = []
    for i, (pool, b) in enumerate(cells):
        target = int(((pos["split_pool"] == pool) & (pos["bin"] == b)).sum())
        cell = cand[(cand["split_pool"] == pool) & (cand["bin"] == b)]
        strong = cell[cell["label_source"] == STRONG]
        weak = cell[cell["label_source"] == WEAK]
        want_strong = round(STRONG_FRAC * target)
        take_strong = strong.sample(
            n=min(want_strong, len(strong)), random_state=seed + i
        )
        want_weak = target - len(take_strong)
        take_weak = weak.sample(
            n=min(want_weak, len(weak)), random_state=seed + i + 1000
        )
        rest = want_weak - len(take_weak)
        extra = pd.DataFrame()
        if rest > 0:
            left = strong.drop(index=take_strong.index)
            extra = left.sample(
                n=min(rest, len(left)), random_state=seed + i + 2000
            )
        taken = pd.concat([take_strong, take_weak, extra], ignore_index=True)
        taken_strong = int((taken["label_source"] == STRONG).sum())
        taken_weak = int((taken["label_source"] == WEAK).sum())
        deficit = target - len(taken)
        strong_deficit = want_strong - taken_strong
        deficit_rows.append(
            {
                "pool": pool,
                "bin": b,
                "target": target,
                "taken_strong": taken_strong,
                "taken_weak": taken_weak,
                "taken": int(len(taken)),
                "deficit": int(deficit),
                "deficit_rate": round(deficit / target, 4) if target else 0.0,
                "strong_target": int(want_strong),
                "strong_taken": taken_strong,
                "strong_deficit": int(strong_deficit),
                "strong_deficit_rate": (
                    round(strong_deficit / want_strong, 4) if want_strong else 0.0
                ),
            }
        )
        picked_parts.append(taken)
    if picked_parts:
        selected = pd.concat(picked_parts, ignore_index=True)
        # Перемешать итог, чтобы порядок parquet не был сгруппирован
        # по (pool, bin) / strong→weak. Детерминировано тем же seed.
        selected = selected.sample(frac=1.0, random_state=seed + 9999).reset_index(
            drop=True
        )
    else:
        selected = df_cand.iloc[0:0].copy()
    deficit_df = pd.DataFrame(
        deficit_rows,
        columns=[
            "pool",
            "bin",
            "target",
            "taken_strong",
            "taken_weak",
            "taken",
            "deficit",
            "deficit_rate",
            "strong_target",
            "strong_taken",
            "strong_deficit",
            "strong_deficit_rate",
        ],
    )
    return selected, deficit_df


def ks_stat(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import ks_2samp

    return float(ks_2samp(a, b).statistic)
