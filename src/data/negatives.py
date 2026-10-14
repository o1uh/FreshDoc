from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from src.data.codocbench_adapter import make_function_hash, make_pair_id

SOURCE = "history_neg"
STRONG = "real_strong_neg"
WEAK = "real_weak_neg"
UNLABELED = "real_unlabeled"

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
        "вход": int(len(df)),
        "underscore": int(m_us.sum()),
        "doc_short": int(m_doc.sum()),
        "len_out": int(m_len.sum()),
        "отброшено_union": int(drop.sum()),
        "успех": int((~drop).sum()),
    }
    return df.loc[~drop].copy(), stats


def build_r1_rows(
    df: pd.DataFrame, pool_of: Dict[str, str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    df = df.copy()
    unknown = sorted(set(df["project"].astype(str)) - set(pool_of))
    if unknown:
        raise ValueError(f"Проекты без пула: {unknown}")
    df["split_pool"] = df["project"].astype(str).map(pool_of)
    n_doc_changed = int(df["doc_changed"].astype(bool).sum())
    df = df.loc[~df["doc_changed"].astype(bool)].copy()

    is_sig = df["category"] == "signature_changed"
    df_cand = df.loc[~is_sig].copy()
    df_un = df.loc[is_sig].copy()

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
            make_pair_id(SOURCE, str(p), str(s), str(c), str(co))
            for p, s, c, co in zip(
                sub["project"], sub["symbol"], sub["commit_sha"], sub["code_old"]
            )
        ]
        sub["function_hash"] = sub["code_old"].astype(str).map(make_function_hash)
        return sub

    cand_strong_weak = pd.concat(
        [
            _enrich(df_cand[df_cand["category"] == "cosmetic"], "strong"),
            _enrich(df_cand[df_cand["category"] == "body_changed"], "weak"),
        ],
        ignore_index=True,
    )
    df_un = _enrich(df_un, "unlabeled")

    for sub in (cand_strong_weak, df_un):
        dup = int(sub.duplicated(subset=["pair_id"]).sum())
        if dup:
            sub.drop_duplicates(subset=["pair_id"], keep="first", inplace=True)
    stats = {
        "doc_changed_отброшено": n_doc_changed,
        "cand_strong": int((cand_strong_weak["label_source"] == STRONG).sum()),
        "cand_weak": int((cand_strong_weak["label_source"] == WEAK).sum()),
        "unlabeled": int(len(df_un)),
    }
    cols = [c for c in R1_COLS if c in cand_strong_weak.columns]
    cols_un = [c for c in R1_COLS if c in df_un.columns]
    return cand_strong_weak[cols].copy(), df_un[cols_un].copy(), stats


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
        want_strong = round(0.3 * target)
        take_strong = strong.sample(
            n=min(want_strong, len(strong)), random_state=seed + i
        )
        want_weak = target - len(take_strong)
        take_weak = weak.sample(n=min(want_weak, len(weak)), random_state=seed + i)
        rest = want_weak - len(take_weak)
        extra = pd.DataFrame()
        if rest > 0:
            left = strong.drop(index=take_strong.index)
            extra = left.sample(n=min(rest, len(left)), random_state=seed + i)
        taken = pd.concat([take_strong, take_weak, extra], ignore_index=True)
        deficit = target - len(taken)
        deficit_rows.append(
            {
                "pool": pool,
                "bin": b,
                "target": target,
                "taken_strong": int((taken["label_source"] == STRONG).sum()),
                "taken_weak": int((taken["label_source"] == WEAK).sum()),
                "taken": int(len(taken)),
                "deficit": int(deficit),
                "deficit_rate": round(deficit / target, 4) if target else 0.0,
            }
        )
        picked_parts.append(taken)
    selected = (
        pd.concat(picked_parts, ignore_index=True) if picked_parts else df_cand.iloc[0:0].copy()
    )
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
        ],
    )
    return selected, deficit_df


def ks_stat(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import ks_2samp

    return float(ks_2samp(a, b).statistic)
