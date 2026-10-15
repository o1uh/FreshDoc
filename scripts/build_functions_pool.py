from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.synth.pool import TEST_POOL, build_functions_pool, build_functions_pool_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="FD-07/FD-09: сборка пула функций.")
    parser.add_argument("--mode", choices=["v1", "v2"], default="v1")
    parser.add_argument("--in_pairs", default="data/interim/pairs_codocbench.parquet")
    parser.add_argument("--in_pools", default="data/interim/project_pools.parquet")
    parser.add_argument("--out", default="data/interim/functions_pool_v1.parquet")
    parser.add_argument("--in_pool_v1", default="data/interim/functions_pool_v1.parquet")
    parser.add_argument("--in_negatives", default="data/interim/negatives_raw.parquet")
    parser.add_argument("--out_v2", default="data/interim/functions_pool_v2.parquet")
    parser.add_argument("--include-tests", dest="include_tests", action="store_true",
                        help="Не отсекать test_-функции источника 2 (по умолчанию отсекаются, FD-09)")
    args = parser.parse_args()

    if args.mode == "v1":
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
        return

    df_pool_v1 = pd.read_parquet(args.in_pool_v1)
    df_negatives = pd.read_parquet(args.in_negatives)
    df_pools = pd.read_parquet(args.in_pools)
    df_v2, stats = build_functions_pool_v2(
        df_pool_v1, df_negatives, df_pools, exclude_tests=not args.include_tests
    )

    test_projects = set(df_pools[df_pools["pool"] == TEST_POOL]["project"])
    leak = set(df_v2["project"]) & test_projects
    assert not leak, f"Утечка test-проектов в пуле v2: {sorted(leak)}"
    assert df_v2["function_hash"].is_unique, "Дубликаты function_hash в пуле v2!"

    out_path = Path(args.out_v2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_v2.to_parquet(out_path, index=False)

    print(f"Статистика FD-09: {stats}")
    print(f"Сохранено: {out_path} (строк: {len(df_v2)}, проектов: {df_v2['project'].nunique()})")
    print(f"Источник2 новых: {stats['источник2_успех']}, v1 вход: {stats['v1_вход']}")
    print(f"Изоляция подтверждена: проектов test-пула в пуле 0")
    print(f"Критерий >=3000: {'ДА' if len(df_v2) >= 3000 else 'НЕТ'}")


if __name__ == "__main__":
    main()
