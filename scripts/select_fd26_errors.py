"""FD-26: отбор всех ошибок M2 на gold + test_real для LLM-разметки.

    python scripts/select_fd26_errors.py [--n_shards 6]

Отбор детерминирован:
- рабочий порог tau и T — из reports/calibration.json (FD-25, подобраны на val);
- ошибки = (p_cal >= tau) != label_stale, только split in (gold, test_real);
- сортировка по pair_id, err_id e001..eN;
- усечение — тем же кодом, что видел M2 (src.models.m2_data.encode_pair,
  microsoft/unixcoder-base @1024, doc_budget 384): флаги truncated/doc_truncated
  фиксируются в full-таблице, в blind-разметку НЕ идут (LLM видит полный текст,
  иначе повторит ошибку усечения).

Выходы (всё детерминировано, повторный прогон байт-идентичен по manifest-ключам):
- reports/fd26_errors_full.csv — с метками, скором M2 и флагами усечения (НЕ показывать LLM);
- reports/fd26_for_annotation.csv — слепая (err_id, project, symbol, sig_old, diff_code,
  doc_old, split, sig_type, source, n_src_lines);
- reports/fd26_shards/fd26_shard_00..05.csv — та же слепая, по 35 строк для параллельных агентов;
- reports/tables/fd26_manifest.json — tau/T, счётчики, ключи sha1(diff+doc), доли усечения.

Ожидаемо (калибровка FD-25, tau=0.874...): gold 62 (57 FP + 5 FN),
test_real 148 (44 FP + 104 FN), итого 210.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

IN_DATASET = PROJECT_ROOT / "data" / "processed" / "dataset_v1.parquet"
IN_PRED = PROJECT_ROOT / "reports" / "preds" / "m2_calibrated.csv"
IN_CAL = PROJECT_ROOT / "reports" / "calibration.json"
OUT_FULL = PROJECT_ROOT / "reports" / "fd26_errors_full.csv"
OUT_ANNO = PROJECT_ROOT / "reports" / "fd26_for_annotation.csv"
OUT_SHARDS = PROJECT_ROOT / "reports" / "fd26_shards"
OUT_MANIFEST = PROJECT_ROOT / "reports" / "tables" / "fd26_manifest.json"

MODEL_NAME = "microsoft/unixcoder-base"
MAX_LEN = 1024
DOC_BUDGET = 384
N_SHARDS_DEFAULT = 6


def key_of(diff_code: str, doc_old: str) -> str:
    return hashlib.sha1((str(diff_code or "") + str(doc_old or "")).encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="FD-26: отбор ошибок M2 для LLM-разметки.")
    ap.add_argument("--dataset", default=str(IN_DATASET))
    ap.add_argument("--pred", default=str(IN_PRED))
    ap.add_argument("--calibration", default=str(IN_CAL))
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--max_len", type=int, default=MAX_LEN)
    ap.add_argument("--doc_budget", type=int, default=DOC_BUDGET)
    ap.add_argument("--n_shards", type=int, default=N_SHARDS_DEFAULT)
    ap.add_argument("--out_full", default=str(OUT_FULL))
    ap.add_argument("--out_anno", default=str(OUT_ANNO))
    ap.add_argument("--shards_dir", default=str(OUT_SHARDS))
    ap.add_argument("--manifest", default=str(OUT_MANIFEST))
    args = ap.parse_args()

    cal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    tau = float(cal["working_threshold"])
    T = float(cal.get("T", 0.0))

    ds = pd.read_parquet(args.dataset)
    need = {"pair_id", "split", "label_stale", "project", "symbol", "sig_old",
            "diff_code", "doc_old", "sig_type", "drift_type", "source",
            "label_source", "n_src_lines"}
    missing = need - set(ds.columns)
    if missing:
        raise SystemExit(f"В датасете нет колонок: {sorted(missing)}")
    pred = pd.read_csv(args.pred)
    if "p_stale" not in pred.columns or "pair_id" not in pred.columns:
        raise SystemExit("В preds нет pair_id/p_stale")
    df = ds.merge(pred[["pair_id", "p_stale"]], on="pair_id", how="inner")
    if len(df) != len(ds):
        print(f"ВНИМАНИЕ: пересечение {len(df)} != датасет {len(ds)}")

    df["m2_pred"] = (df["p_stale"] >= tau).astype(int)
    err = df[(df["split"].isin(["gold", "test_real"]))
             & (df["m2_pred"] != df["label_stale"])].copy()
    err = err.sort_values("pair_id").reset_index(drop=True)
    err.insert(0, "err_id", [f"e{i + 1:03d}" for i in range(len(err))])
    err["error_type"] = err.apply(
        lambda r: "FP" if (r["m2_pred"] == 1 and r["label_stale"] == 0) else "FN", axis=1)

    n_gold = int((err["split"] == "gold").sum())
    n_test = int((err["split"] == "test_real").sum())
    n_fp_gold = int(((err["split"] == "gold") & (err["error_type"] == "FP")).sum())
    n_fn_gold = int(((err["split"] == "gold") & (err["error_type"] == "FN")).sum())
    n_fp_test = int(((err["split"] == "test_real") & (err["error_type"] == "FP")).sum())
    n_fn_test = int(((err["split"] == "test_real") & (err["error_type"] == "FN")).sum())
    print(f"Ошибки: всего {len(err)} (gold {n_gold}: FP {n_fp_gold} + FN {n_fn_gold}; "
          f"test_real {n_test}: FP {n_fp_test} + FN {n_fn_test}), tau={tau}")
    if (len(err), n_gold, n_test) != (210, 62, 148):
        print(f"ВНИМАНИЕ: ожидалось 210/62/148 (см. e4_e5_e6.md), факт "
              f"{len(err)}/{n_gold}/{n_test} — проверить tau и preds.")
    assert err["pair_id"].is_unique, "дубли pair_id в ошибках"
    assert err["err_id"].is_unique, "дубли err_id"

    # Детерминированное усечение тем же кодом, что видел M2.
    from transformers import AutoTokenizer

    from src.models.m2_data import encode_pair

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    trunc_flags = []
    for _, r in err.iterrows():
        enc = encode_pair(tok, str(r["sig_old"] or ""), str(r["diff_code"] or ""),
                          str(r["doc_old"] or ""), max_len=args.max_len,
                          doc_budget=args.doc_budget)
        trunc_flags.append((bool(enc["truncated"]), bool(enc["doc_truncated"]),
                            int(enc["n_a_tok"]), int(enc["n_b_tok"])))
    err["truncated"] = [t[0] for t in trunc_flags]
    err["doc_truncated"] = [t[1] for t in trunc_flags]
    err["n_a_tok"] = [t[2] for t in trunc_flags]
    err["n_b_tok"] = [t[3] for t in trunc_flags]
    print(f"Усечение A: {int(err['truncated'].sum())}/{len(err)}; "
          f"doc: {int(err['doc_truncated'].sum())}/{len(err)}")

    full_cols = ["err_id", "pair_id", "split", "source", "project", "symbol",
                 "sig_old", "diff_code", "doc_old", "label_stale", "drift_type",
                 "sig_type", "label_source", "n_src_lines", "p_stale", "m2_pred",
                 "error_type", "truncated", "doc_truncated", "n_a_tok", "n_b_tok"]
    full = err[full_cols].copy()
    full = full.rename(columns={"p_stale": "m2_p"})
    out_full = Path(args.out_full)
    out_full.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_full, index=False)

    anno_cols = ["err_id", "project", "symbol", "sig_old", "diff_code", "doc_old",
                 "split", "sig_type", "source", "n_src_lines"]
    anno = err[anno_cols].copy()
    out_anno = Path(args.out_anno)
    anno.to_csv(out_anno, index=False)

    # Шарды: стратифицированно по (split, error_type), внутри — по pair_id,
    # затем round-robin i % n_shards. Каждый шард — смесь gold/test и FP/FN.
    n_shards = int(args.n_shards)
    groups = err.groupby(["split", "error_type"], sort=True)
    ordered_idx: list[int] = []
    group_lists = [g.sort_values("pair_id").index.tolist() for _, g in groups]
    maxlen = max(len(lst) for lst in group_lists)
    for i in range(maxlen):
        for lst in group_lists:
            if i < len(lst):
                ordered_idx.append(lst[i])
    ordered = err.loc[ordered_idx].reset_index(drop=True)
    shards_dir = Path(args.shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)
    # Чистим старые шарды той же маски, чтобы не оставалось хвостов.
    for old in sorted(shards_dir.glob("fd26_shard_*.csv")):
        old.unlink()
    shard_sizes = []
    for s in range(n_shards):
        sub = ordered.iloc[s::n_shards].copy()
        sub_anno = sub[anno_cols].copy()
        path = shards_dir / f"fd26_shard_{s:02d}.csv"
        sub_anno.to_csv(path, index=False)
        shard_sizes.append(len(sub_anno))
    print(f"Шарды ({n_shards}): {shard_sizes} -> {shards_dir}")

    manifest = {
        "task": "FD-26",
        "model_m2": args.model,
        "max_len": args.max_len,
        "doc_budget": args.doc_budget,
        "tau": tau,
        "T": T,
        "pred_source": str(Path(args.pred).relative_to(PROJECT_ROOT).as_posix())
        if Path(args.pred).is_absolute() and
        str(Path(args.pred)).startswith(str(PROJECT_ROOT)) else str(args.pred),
        "dataset": str(Path(args.dataset).relative_to(PROJECT_ROOT).as_posix())
        if Path(args.dataset).is_absolute() and
        str(Path(args.dataset)).startswith(str(PROJECT_ROOT)) else str(args.dataset),
        "n_total": int(len(err)),
        "n_gold": int(n_gold),
        "n_test_real": int(n_test),
        "fp_gold": int(n_fp_gold),
        "fn_gold": int(n_fn_gold),
        "fp_test_real": int(n_fp_test),
        "fn_test_real": int(n_fn_test),
        "n_truncated_A": int(err["truncated"].sum()),
        "n_truncated_doc": int(err["doc_truncated"].sum()),
        "n_shards": int(n_shards),
        "shard_sizes": shard_sizes,
        "err_id_to_pair_id": dict(zip(full["err_id"].astype(str), full["pair_id"].astype(str))),
        "keys": {str(r["err_id"]): key_of(r["diff_code"], r["doc_old"])
                 for _, r in err.iterrows()},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": ("Слепая разметка LLM (Spark + Gemini) — только anno/shards, без "
                 "label_stale/m2_p/error_type/truncated. Full — для сведения споров и репорта."),
    }
    out_manifest = Path(args.manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(f"full: {out_full} ({len(full)} строк)")
    print(f"anno: {out_anno} ({len(anno)} строк)")
    print(f"manifest: {out_manifest}")


if __name__ == "__main__":
    main()
