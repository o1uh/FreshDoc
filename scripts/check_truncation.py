"""FD-21 шаг 4: доля усечённых входов на токенизаторе рабочей модели.

    python scripts/check_truncation.py [--dataset ...] [--model microsoft/unixcoder-base] [--max_len 1024]

Оценочные доли из workflow §3 (полный текст): 28.6% @512 / 4.4% @1024 / ~1% @2048.
Фактические доли на токенизаторе рабочей модели записываются в
reports/m2_truncation.json и в docs/data_notes.md (вручную).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.models.m2_data import encode_pair

IN_DATASET = PROJECT_ROOT / "data" / "processed" / "dataset_v1.parquet"
OUT_JSON = PROJECT_ROOT / "reports" / "m2_truncation.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="FD-21 шаг 4: усечение входов (R10).")
    ap.add_argument("--dataset", default=str(IN_DATASET))
    ap.add_argument("--model", default="microsoft/unixcoder-base")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--doc_budget", type=int, default=192)
    ap.add_argument("--out", default=None,
                    help="куда писать JSON. По умолчанию reports/m2_truncation.json, "
                         "но при --sample>0 --out обязателен (защита канона от затирания)")
    ap.add_argument("--sample", type=int, default=0,
                    help="0 = весь датасет; иначе первые N строк для быстрой прикидки")
    args = ap.parse_args()

    if args.sample and not args.out:
        print("ОШИБКА: --sample>0 требует явный --out "
              "(канон reports/m2_truncation.json затирать запрещено).",
              file=sys.stderr)
        return 2
    out_path = Path(args.out) if args.out else OUT_JSON

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    df = pd.read_parquet(args.dataset)
    if args.sample and len(df) > args.sample:
        df = df.head(args.sample).copy()
    print(f"Датасет: {args.dataset} ({len(df)} строк), модель: {args.model}, "
          f"max_len={args.max_len}, doc_budget={args.doc_budget}")
    print(f"Токенизатор: cls={tok.cls_token} sep={tok.sep_token} "
          f"model_max_length={getattr(tok, 'model_max_length', '?')}")
    # NaN-защита как в train_m2/predict_m2: сначала fillna, потом astype(str)
    # (str(np.nan) дал бы литерал "nan" в тексте входа).
    sigs = df["sig_old"].fillna("").astype(str)
    diffs = df["diff_code"].fillna("").astype(str)
    docs = df["doc_old"].fillna("").astype(str)

    rows = []
    for split, idx in df.groupby("split").groups.items():
        sub = df.loc[idx]
        n_trunc = n_doc_trunc = 0
        for j in sub.index:
            enc = encode_pair(tok, sigs.loc[j], diffs.loc[j], docs.loc[j],
                              max_len=args.max_len, doc_budget=args.doc_budget)
            n_trunc += int(enc["truncated"])
            n_doc_trunc += int(enc["doc_truncated"])
        n = len(sub)
        frac = n_trunc / max(1, n)
        doc_frac = n_doc_trunc / max(1, n)
        rows.append({"split": split, "n": int(n),
                     "n_truncated": int(n_trunc), "frac": float(frac),
                     "n_doc_truncated": int(n_doc_trunc),
                     "doc_frac": float(doc_frac)})
        print(f"  {split}: n={n} truncA={n_trunc} frac={frac:.4f} "
              f"truncDoc={n_doc_trunc} doc_frac={doc_frac:.4f}")
    out = {"model": args.model, "max_len": args.max_len,
           "doc_budget": args.doc_budget, "dataset_rows": int(len(df)),
           "sample": int(args.sample or 0),
           "by_split": rows,
           "note": "Оценки workflow §3 (полный текст): 28.6% @512 / 4.4% @1024 / ~1% @2048. "
                   "Фактические доли — на токенизаторе рабочей модели."}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"Отчёт: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
