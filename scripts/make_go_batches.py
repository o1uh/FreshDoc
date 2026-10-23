"""FD-39 фаза 2: батчи слепой R9-разметки полного Go-набора для изолированных судей.

    python scripts/make_go_batches.py --input data/processed/go_full381.csv \
        --out_dir reports/go_batches_full --per_batch 12

Читает data/processed/go_full381.csv (слепой: label/comment пустые;
sidecar с прокси-метками в батчи НЕ попадает),
пишет reports/go_batches_full/go_batchNN.prompt.txt (по 10-12 кейсов):
внутри — полный текст задания из prompts/go_annotate.txt + кейсы.
Субагенту передаётся ТОЛЬКО текст батча. Ответ — CSV id,label,stale_statement,why.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

PILOT = PROJECT_ROOT / "data" / "processed" / "go_full381.csv"
OUT_DIR = PROJECT_ROOT / "reports" / "go_batches_full"
HEADER_PATH = PROJECT_ROOT / "prompts" / "go_annotate.txt"


def case_block(r: pd.Series) -> str:
    return (
        f"=== CASE id={r['id']} ===\n"
        f"FILE: {r['file']} | SYMBOL: {r['symbol']}\n"
        f"--- SIG_OLD ---\n{str(r['sig_old'] or '').strip()}\n"
        f"--- DIFF ---\n{str(r['diff_code'] or '').strip()}\n"
        f"--- DOC_OLD ---\n{str(r['doc_old'] or '').strip()}\n"
        f"=== END CASE ===\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="FD-39: батчи полного Go-набора (full381).")
    ap.add_argument("--input", "--pilot", dest="input", default=str(PILOT),
                    help="входной CSV набора (слепой). --pilot — deprecated алиас.")
    ap.add_argument("--out_dir", "--shards_dir", dest="out_dir", default=str(OUT_DIR))
    ap.add_argument("--per_batch", type=int, default=12)
    args = ap.parse_args()
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    if len(df) == 0:
        raise SystemExit("набор пуст")
    if df["id"].duplicated().any():
        raise SystemExit("дубли id в наборе")
    if (df["doc_old"].str.strip() == "").any():
        raise SystemExit("в наборе есть пустые doc_old — судьи такое не размечают")
    if (df["diff_code"].str.strip() == "").any():
        raise SystemExit("в наборе есть пустые diff_code — судьи такое не размечают")
    forbidden = {"doc_changed", "doc_changed_proxy", "label_stale", "p_stale", "proxy",
                 "doc_new", "sig_changed", "code_new", "label_source"}
    # code_new/code_old — канонические исходные колонки silver-схемы (нужны
    # downstream для сборки parquet), в слепые батчи не попадают (case_block
    # берёт фиксированный набор колонок) — поэтому из запрета исключены.
    # Всё остальное метко-подобное — жёсткий отказ.
    source_ok = {"code_old", "code_new"}
    leak = (forbidden - source_ok) & set(df.columns)
    leak_prefix = sorted(c for c in df.columns if c.startswith("_leak_"))
    if leak or leak_prefix:
        raise SystemExit(f"в наборе утечка колонок: {sorted(leak) + leak_prefix}")
    header = Path(HEADER_PATH).read_text(encoding="utf-8")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in sorted(out_dir.glob("go_batch*.prompt.txt")):
        old.unlink()
    n = int(args.per_batch)
    batches = [df.iloc[i:i + n] for i in range(0, len(df), n)]
    for i, b in enumerate(batches):
        body = header + "\nКЕЙСЫ:\n" + "\n".join(case_block(r) for _, r in b.iterrows())
        out = out_dir / f"go_batch{i:02d}.prompt.txt"
        out.write_text(body, encoding="utf-8")
        print(f"{out.name}: {len(b)} кейсов, {out.stat().st_size} bytes")
    print(f"батчей: {len(batches)} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
