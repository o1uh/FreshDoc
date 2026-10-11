from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.synth.pool import TEST_POOL, TRAIN_POOL, build_functions_pool


def main() -> None:
    parser = argparse.ArgumentParser(description="FD-07: сборка пула функций v1 для синтетики.")
    parser.add_argument("--in_pairs", default="data/interim/pairs_codocbench.parquet")
    parser.add_argument("--in_pools", default="data/interim/project_pools.parquet")
    parser.add_argument("--out", default="data/interim/functions_pool_v1.parquet")
    args = parser.parse_args()

    df_pairs = pd.read_parquet(args.in_pairs)
    df_pools = pd.read_parquet(args.in_pools)
    df_pool, stats = build_functions_pool(df_pairs, df_pools)

    # Автопроверка FD-07: нет функций из test-пула.
    test_projects = set(df_pools[df_pools["pool"] == TEST_POOL]["project"])
    leak = set(df_pool["project"]) & test_projects
    assert not leak, f"Утечка test-проектов в пуле: {sorted(leak)}"
    assert df_pool["function_hash"].is_unique, "Дубликаты function_hash в пуле!"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_pool.to_parquet(out_path, index=False)

    print(f"Статистика FD-07: {stats}")
    print(f"Сохранено: {out_path} (строк: {len(df_pool)}, проектов: {df_pool['project'].nunique()})")
    print(f"Изоляция подтверждена: проектов test-пула в пуле 0, train-проектов: {df_pool['project'].nunique()}")
    print(f"Критерий >=1200: {'ДА' if len(df_pool) >= 1200 else 'НЕТ'}")


if __name__ == "__main__":
    main()
