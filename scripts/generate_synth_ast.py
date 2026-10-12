from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.synth.generate import generate_for_pool


def main() -> None:
    ap = argparse.ArgumentParser(description="FD-08: генерация synth_ast (R6, 6 pos + 3 neg).")
    ap.add_argument("--in_pool", default="data/interim/functions_pool_v1.parquet")
    ap.add_argument("--out", default="data/interim/synth_ast.parquet")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_pairs", type=int, default=30000)
    args = ap.parse_args()

    df_pool = pd.read_parquet(args.in_pool)
    df, stats = generate_for_pool(df_pool, seed=args.seed, max_pairs=args.max_pairs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"Статистика FD-08: {stats}")
    print(f"Сохранено: {out} (строк: {len(df)})")
    if len(df):
        print("По типам:")
        print(df["drift_type"].value_counts().to_string())
        print(f"Баланс меток: {df['label_stale'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
