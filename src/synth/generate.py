from __future__ import annotations

import ast
import hashlib
import random
import zlib
from typing import Dict, List, Tuple

import pandas as pd

from src.data.drift import derive_drift_type, extract_signature
from src.data.normalizer import (
    compute_code_diff,
    extract_docstring_and_clean_node,
    extract_sig_line,
    normalize_code_from_node,
    parse_function_ast,
)
from src.synth.mutations import MUTATIONS, NEG_TYPES, POS_TYPES


def _pair_id(function_hash: str, drift_type: str, diff: str) -> str:
    key = f"{function_hash}:{drift_type}:{diff}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _attempt_seed(seed: int, idx, mtype: str, draw: int, att: int) -> int:
    # Детерминированный сид попытки. Встроенный hash() использовать нельзя:
    # он рандомизируется солью PYTHONHASHSEED при каждом запуске интерпретатора
    # (VS Code и Kaggle давали разные наборы при том же --seed).
    salt = zlib.crc32(f"{idx}:{mtype}:{draw}:{att}".encode("utf-8"))
    return seed + salt


# Редкие типы пробуются первыми в каждой функции (вариант 3, без ослабления
# условий применимости): default/return иначе не добирают до 3%.
POS_RARE_FIRST = ["default_changed", "return_changed"]


def generate_for_pool(
    df_pool: pd.DataFrame,
    seed: int = 42,
    max_pairs: int = 30000,
    pos_per_func: int = 3,
    neg_per_func: int = 3,
    attempts_per_type: int = 5,
) -> Tuple[pd.DataFrame, Dict]:
    rng_global = random.Random(seed)
    stats: Dict = {
        "функций_вход": len(df_pool),
        "функций_не_разобралось": 0,
        "попыток": 0,
        "принято": 0,
        "отклонено_none": 0,
        "отклонено_код_тот_же": 0,
        "отклонено_синтаксис": 0,
        "отклонено_страховка": 0,
        "по_типам": {},
    }
    pos_rows: List[dict] = []
    neg_rows: List[dict] = []

    for idx, r in df_pool.iterrows():
        src = r["src"]
        fn = parse_function_ast(src)
        if fn is None:
            stats["функций_не_разобралось"] += 1
            continue
        doc, fn_no_doc = extract_docstring_and_clean_node(fn)
        doc_old = (doc or "").strip()
        if not doc_old:
            stats["функций_не_разобралось"] += 1
            continue
        code_old = normalize_code_from_node(fn_no_doc)
        sig_old_line = extract_sig_line(code_old)
        sig_old_obj = extract_signature(fn_no_doc)
        n_src_lines = len(src.strip().splitlines())

        for types, want, bucket, rare_first, with_replacement in (
            (POS_TYPES, pos_per_func, pos_rows, True, False),
            (NEG_TYPES, neg_per_func, neg_rows, False, True),
        ):
            if with_replacement:
                # Типы с возвращением: функция может дать несколько пар
                # одного типа (разные сиды → разные диффы и pair_id).
                order_iter = (rng_global.choice(list(types))
                              for _ in range(want * len(types) * 2))
            else:
                if rare_first:
                    rare = [t for t in POS_RARE_FIRST if t in types]
                    rest = [t for t in types if t not in rare]
                    rng_global.shuffle(rest)
                    order_iter = iter(rare + rest)
                else:
                    order = list(types)
                    rng_global.shuffle(order)
                    order_iter = iter(order)
            got = 0
            for draw, mtype in enumerate(order_iter):
                if got >= want:
                    break
                func, label, expected = MUTATIONS[mtype]
                accepted = None
                for att in range(attempts_per_type):
                    stats["попыток"] += 1
                    rng = random.Random(_attempt_seed(seed, idx, mtype, draw, att))
                    try:
                        new_fn = func(fn_no_doc, doc_old, rng)
                    except Exception:  # noqa: BLE001 — мутация не должна ронять драйвер
                        new_fn = None
                    if new_fn is None:
                        continue
                    try:
                        code_new = normalize_code_from_node(new_fn)
                    except Exception:  # noqa: BLE001
                        continue
                    try:
                        ast.parse(code_new)
                    except SyntaxError:
                        stats["отклонено_синтаксис"] += 1
                        continue
                    if code_new == code_old:
                        stats["отклонено_код_тот_же"] += 1
                        continue
                    try:
                        sig_new_obj = extract_signature(new_fn)
                        drift = derive_drift_type(sig_old_obj, sig_new_obj)
                    except Exception:  # noqa: BLE001
                        continue
                    if drift != expected:
                        stats["отклонено_страховка"] += 1
                        continue
                    diff = compute_code_diff(code_old, code_new)
                    accepted = (code_new, drift, diff)
                    break
                if accepted is None:
                    stats["отклонено_none"] += 1
                    continue
                code_new, drift, diff = accepted
                row = {
                    "pair_id": _pair_id(str(r["function_hash"]), drift, diff),
                    "source": "synth_ast",
                    "project": str(r["project"]),
                    "commit_sha": "",
                    "symbol": fn.name,
                    "function_hash": str(r["function_hash"]),
                    "sig_old": sig_old_line,
                    "code_old": code_old,
                    "code_new": code_new,
                    "diff_code": diff,
                    "doc_old": doc_old,
                    "label_stale": label,
                    "drift_type": drift,
                    "sig_type": drift,
                    "label_source": "ast",
                    "weight": 1.0,
                    "n_src_lines": n_src_lines,
                }
                bucket.append(row)
                stats["по_типам"][mtype] = stats["по_типам"].get(mtype, 0) + 1
                stats["принято"] += 1
                got += 1

    # Баланс 1:1 + кап max_pairs (R6: общий объём ограничить).
    half = max_pairs // 2
    n = min(len(pos_rows), len(neg_rows), half)
    rng_global.shuffle(pos_rows)
    rng_global.shuffle(neg_rows)
    rows = pos_rows[:n] + neg_rows[:n]
    rng_global.shuffle(rows)
    stats["pos_всего"] = len(pos_rows)
    stats["neg_всего"] = len(neg_rows)
    stats["итого"] = len(rows)
    stats["баланс_n"] = n
    df = pd.DataFrame(rows)
    # Дедуп по pair_id (страховка от коллизий диффа).
    if len(df):
        before = len(df)
        df = df.drop_duplicates(subset=["pair_id"], keep="first")
        stats["дубликаты_pair_id"] = before - len(df)
    else:
        stats["дубликаты_pair_id"] = 0
    return df, stats
