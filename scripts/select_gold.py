"""FD-16: отбор кандидатов золотого набора и раздача (R9).

    python scripts/select_gold.py [--seed 42]

Только test-пул: 70 положительных (квоты пропорциональны пулу),
70 отрицательных (весь strong test-пула + weak 11 с проекта),
60 unlabeled (по 10 на тип дрейфа). Кап <= 3 на проект для
положительных и unlabeled. Отклонение от R9 (40 weak + 30 strong):
в test-пуле всего 4 strong — утверждено человеком 16.10.2026
(4 strong + 66 weak, решение в docs/data_notes.md).

Выходы: data/interim/gold_candidates_full.parquet (с метками,
разметчикам НЕ показывать), data/gold/gold_for_annotation.csv
и data/gold/annotator_1..4.csv (без меток). Раздача по R9:
первые 60 строк — по двое разметчиков по кругу, остальные 140 —
по одному.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.gold import (
    PROJECT_CAP,
    assign_annotators,
    largest_remainder,
    round_robin_pick,
)

IN_POS = PROJECT_ROOT / "data" / "interim" / "pairs_codocbench.parquet"
IN_NEG = PROJECT_ROOT / "data" / "interim" / "pairs_negatives.parquet"
IN_UNL = PROJECT_ROOT / "data" / "interim" / "unlabeled_sigchange.parquet"
IN_DATASET = PROJECT_ROOT / "data" / "processed" / "dataset_v1.parquet"
OUT_FULL = PROJECT_ROOT / "data" / "interim" / "gold_candidates_full.parquet"
OUT_TABLE = PROJECT_ROOT / "data" / "gold" / "gold_for_annotation.csv"
OUT_JSON = PROJECT_ROOT / "reports" / "tables" / "gold_select.json"
N_ANNOTATORS = 4

N_POS = 70
N_NEG_WEAK = 66
N_UNL_PER_TYPE = 10
DOUBLE_N = 60


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    pos = pd.read_parquet(IN_POS)
    neg = pd.read_parquet(IN_NEG)
    unl = pd.read_parquet(IN_UNL)
    pos_t = pos[pos["split_pool"] == "test_pool"].copy()
    neg_t = neg[neg["split_pool"] == "test_pool"].copy()
    unl_t = unl[unl["split_pool"] == "test_pool"].copy()

    # 1. Положительные: квоты пропорционально пулу, кап <= 3 на проект.
    pos_counts = pos_t.groupby("drift_type").size().to_dict()
    pos_quotas = largest_remainder(pos_counts, N_POS)
    cap: dict[str, int] = {}
    pos_parts = []
    for drift in sorted(pos_quotas):
        cand = pos_t[pos_t["drift_type"] == drift]
        assert len(cand) >= pos_quotas[drift], f"мало {drift}: {len(cand)}"
        pos_parts.append(round_robin_pick(cand, pos_quotas[drift], rng, cap))
    sel_pos = pd.concat(pos_parts, ignore_index=True)

    # 2. Отрицательные: весь strong test-пула + weak 11 с проекта.
    strong = neg_t[neg_t["label_source"] == "real_strong_neg"].copy()
    weak = neg_t[neg_t["label_source"] == "real_weak_neg"].copy()
    weak_projects = sorted(weak["project"].unique())
    per_project = N_NEG_WEAK // len(weak_projects)
    assert N_NEG_WEAK % len(weak_projects) == 0, "66 не делится на проекты"
    assert all((weak["project"] == p).sum() >= per_project
               for p in weak_projects), "мало weak в проекте"
    weak_parts = []
    for project in weak_projects:
        cand = weak[weak["project"] == project]
        idx = list(cand.index)
        rng.shuffle(idx)
        weak_parts.append(cand.loc[idx[:per_project]])
    sel_neg = pd.concat([strong] + weak_parts, ignore_index=True)

    # 3. Unlabeled: по 10 на тип, кап <= 3 на проект в пределах типа.
    # (Общий кап невозможен: в test-пуле всего 5-6 проектов на 60 строк.)
    # Редкие типы (мало проектов) — первыми, иначе кап внутри типа
    # заблокирует их квоту.
    unl_types = sorted(unl_t["drift_type"].unique(),
                       key=lambda d: (unl_t[unl_t["drift_type"] == d]
                                      ["project"].nunique(), d))
    unl_parts = []
    for drift in unl_types:
        cand = unl_t[unl_t["drift_type"] == drift]
        assert len(cand) >= N_UNL_PER_TYPE, f"мало unl {drift}: {len(cand)}"
        unl_parts.append(round_robin_pick(cand, N_UNL_PER_TYPE, rng, {}))
    sel_unl = pd.concat(unl_parts, ignore_index=True)

    # 4. Склейка, перемешивание, идентификаторы g000..g199.
    full = pd.concat([sel_pos, sel_neg, sel_unl], ignore_index=True)
    assert len(full) == 200, len(full)
    assert full["pair_id"].is_unique, "дубли pair_id в отборе"
    order = list(full.index)
    rng.shuffle(order)
    full = full.loc[order].reset_index(drop=True)
    full.insert(0, "gold_id", [f"g{i:03d}" for i in range(len(full))])

    # Изоляция: функции gold не из train (проверка FD-13 заранее).
    dataset = pd.read_parquet(IN_DATASET, columns=["function_hash", "split"])
    train_hashes = set(dataset[dataset["split"] == "train"]["function_hash"])
    overlap = set(full["function_hash"]) & train_hashes
    assert not overlap, f"пересечение с train: {len(overlap)}"

    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(OUT_FULL, index=False)

    # 5. Таблица для разметки (без меток) и личные файлы по R9.
    table = pd.DataFrame({
        "id": full["gold_id"],
        "project": full["project"],
        "symbol": full["symbol"],
        "sig_old": full["sig_old"],
        "diff_code": full["diff_code"],
        "doc_old": full["doc_old"],
        "label": "",
        "comment": "",
    })
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_TABLE, index=False)

    per_annotator = assign_annotators(len(table), N_ANNOTATORS, DOUBLE_N)
    for a in range(N_ANNOTATORS):
        path = OUT_TABLE.parent / f"annotator_{a + 1}.csv"
        table.iloc[sorted(per_annotator[a])].to_csv(path, index=False)

    summary = {
        "seed": args.seed,
        "n_total": len(full),
        "n_pos": len(sel_pos),
        "n_neg": len(sel_neg),
        "n_strong": len(strong),
        "n_weak": len(sel_neg) - len(strong),
        "n_unlabeled": len(sel_unl),
        "pos_quotas": pos_quotas,
        "pos_counts_pool": pos_counts,
        "weak_per_project": per_project,
        "weak_projects": weak_projects,
        "unl_per_type": N_UNL_PER_TYPE,
        "project_cap": PROJECT_CAP,
        "per_annotator_rows": {f"annotator_{a + 1}": len(per_annotator[a])
                               for a in range(N_ANNOTATORS)},
        "train_hash_overlap": 0,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False)
                        + "\n", encoding="utf-8")

    print(f"gold: {len(full)} строк "
          f"(pos {len(sel_pos)}, neg {len(sel_neg)} "
          f"[strong {len(strong)}], unl {len(sel_unl)})")
    print(f"квоты pos: {pos_quotas}")
    print(f"на разметчика: {summary['per_annotator_rows']}")
    print(f"выходы: {OUT_FULL.name}, {OUT_TABLE.name}, annotator_1..4.csv")


if __name__ == "__main__":
    main()
