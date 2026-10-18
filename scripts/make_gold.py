"""FD-18 (часть 1): сборка gold.csv и подмешивание в датасет.

    python scripts/make_gold.py [--apply]

Без --apply — только отчёт о том, что будет сделано (dry-run).
С --apply:
  1. data/gold/gold.csv — 200 строк по схеме R1: source='gold',
     label_source='gold_human', weight=0.0, split='gold',
     label_stale = (final in {A,B}) из annotations_final.csv,
     label_stale_strict = (final == A). Без _leak_* колонок.
  2. dataset_v1.parquet: строки gold из test_real получают split='gold';
     60 unlabeled (их не было в dataset_v1) добавляются строками.
  3. Прогон scripts/validate_dataset.py — проверки должны остаться зелёными.
Манифест dataset_v1.manifest.json обновляется на заморозке FD-31.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_dataset import main as validate_main  # noqa: E402

CAND = PROJECT_ROOT / "data" / "interim" / "gold_candidates_full.parquet"
ANN = PROJECT_ROOT / "reports" / "annotations_final.csv"
DATASET = PROJECT_ROOT / "data" / "processed" / "dataset_v1.parquet"
GOLD = PROJECT_ROOT / "data" / "gold" / "gold.csv"


def build() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cand = pd.read_parquet(CAND)
    ann = pd.read_csv(ANN)
    assert len(ann) == 200 and ann["id"].is_unique
    m = cand.merge(ann[["id", "final"]].rename(columns={"id": "gold_id"}),
                   on="gold_id", validate="1:1")
    m["label_stale"] = m["final"].isin(["A", "B"]).astype(int)
    m["label_stale_strict"] = (m["final"] == "A").astype(int)
    m["source"] = "gold"
    m["label_source"] = "gold_human"
    m["weight"] = 0.0
    m["split"] = "gold"
    keep_leak = [c for c in ["_leak_doc_new", "_leak_diff_docstring",
                             "_leak_commit_message", "_aux_src_old"]
                 if c in m.columns]
    gold = m.drop(columns=["split_pool"] + keep_leak)
    ds = pd.read_parquet(DATASET)
    in_ds = ds["pair_id"].isin(gold["pair_id"])
    moved = sorted(ds.loc[in_ds & (ds["split"] == "test_real"), "pair_id"].tolist())
    new_rows = m[~m["pair_id"].isin(set(ds["pair_id"]))]
    return gold, ds, moved, new_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    gold, ds, moved, new_rows = build()
    print(f"gold: {len(gold)} строк "
          f"(stale={int(gold['label_stale'].sum())}, "
          f"strict={int(gold['label_stale_strict'].sum())})")
    print(f"test_real -> gold: {len(moved)} строк")
    print(f"новых строк (unlabeled) в dataset: {len(new_rows)}")
    if not args.apply:
        print("dry-run: без --apply ничего не записано")
        return
    gold.to_csv(GOLD, index=False)
    in_gold = ds["pair_id"].isin(set(gold["pair_id"]))
    for col, val in (("split", "gold"), ("source", "gold"),
                     ("label_source", "gold_human"), ("weight", 0.0)):
        ds.loc[in_gold, col] = val
    gold_label = dict(zip(gold["pair_id"].astype(str),
                          gold["label_stale"].astype(int)))
    ds.loc[in_gold, "label_stale"] = ds.loc[in_gold, "pair_id"].astype(str).map(gold_label).astype(int)
    append_cols = [c for c in ds.columns if c in new_rows.columns]
    assert set(append_cols) == set(ds.columns), \
        sorted(set(ds.columns) - set(append_cols))
    ds = pd.concat([ds, new_rows[ds.columns]], ignore_index=True)
    assert (ds["pair_id"].isin(set(gold["pair_id"])) & (ds["split"] != "gold")).sum() == 0
    ds.to_parquet(DATASET, index=False)
    print(f"записано: {GOLD.name} + dataset_v1 ({len(ds)} строк)")
    import sys as _sys
    _argv, _sys.argv = _sys.argv, [str(PROJECT_ROOT / "scripts" / "validate_dataset.py")]
    try:
        rc = validate_main()
    finally:
        _sys.argv = _argv
    if rc != 0:
        raise SystemExit(f"validate_dataset FAILED (rc={rc})")
    print("проверки dataset_v1 зелёные")


if __name__ == "__main__":
    main()
