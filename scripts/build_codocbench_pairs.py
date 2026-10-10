from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.codocbench_adapter import CoDocBenchAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Адаптер CoDocBench во внутренний формат FreshDoc.")
    parser.add_argument(
        "--config",
        default="config/codocbench_map.yaml",
        help="Путь к YAML-маппингу полей CoDocBench",
    )
    parser.add_argument(
        "--train_file",
        default="data/external/codocbench/dataset/train.jsonl",
        help="Путь к train.jsonl",
    )
    parser.add_argument(
        "--test_file",
        default="data/external/codocbench/dataset/test.jsonl",
        help="Путь к test.jsonl",
    )
    parser.add_argument(
        "--out_pairs",
        default="data/interim/pairs_codocbench.parquet",
        help="Путь для сохранения pairs_codocbench.parquet",
    )
    parser.add_argument(
        "--out_pools",
        default="data/interim/project_pools.parquet",
        help="Путь для сохранения project_pools.parquet",
    )
    args = parser.parse_args()

    adapter = CoDocBenchAdapter(args.config)

    print(f"Обработка train: {args.train_file}")
    train_rows, train_stats = adapter.process_file(args.train_file, split_pool="train_pool")
    print(f"Статистика train: {train_stats}")

    print(f"Обработка test: {args.test_file}")
    test_rows, test_stats = adapter.process_file(args.test_file, split_pool="test_pool")
    print(f"Статистика test: {test_stats}")

    all_rows = train_rows + test_rows
    df_pairs = pd.DataFrame(all_rows)

    total_records = train_stats["всего"] + test_stats["всего"]
    total_parsed = len(df_pairs)
    print(f"\nИтого записей в CoDocBench: {total_records}")
    print(f"Итого отобрано положительных пар: {total_parsed} ({total_parsed / total_records * 100:.1f}%)")

    train_projects = set(r["project"] for r in train_rows)
    test_projects = set(r["project"] for r in test_rows)
    intersection = train_projects & test_projects

    if intersection:
        print(f"ВНИМАНИЕ: Найдена утечка проектов между train и test в CoDocBench: {intersection}")
        print(f"Разрешение утечки: перенос проектов {intersection} целиком в test_pool.")
        mask = df_pairs["project"].isin(intersection)
        df_pairs.loc[mask, "split_pool"] = "test_pool"
        train_projects = train_projects - intersection

    pool_records = [{"project": p, "pool": "train_pool"} for p in train_projects] + [
        {"project": p, "pool": "test_pool"} for p in test_projects
    ]
    df_pools = pd.DataFrame(pool_records).drop_duplicates(subset=["project"])

    out_pairs_path = Path(args.out_pairs)
    out_pools_path = Path(args.out_pools)
    out_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    out_pools_path.parent.mkdir(parents=True, exist_ok=True)

    df_pairs.to_parquet(out_pairs_path, index=False)
    df_pools.to_parquet(out_pools_path, index=False)

    print(f"\nСохранено пар: {out_pairs_path} (строк: {len(df_pairs)})")
    print(f"Сохранено пулов: {out_pools_path} (проектов: {len(df_pools)})")

    # Контрольная проверка изоляции
    final_train_projects = set(df_pairs[df_pairs["split_pool"] == "train_pool"]["project"])
    final_test_projects = set(df_pairs[df_pairs["split_pool"] == "test_pool"]["project"])
    assert len(final_train_projects & final_test_projects) == 0, "Критическая ошибка: пулы пересекаются!"
    print("Изоляция пулов подтверждена: 0 пересечений между train_pool и test_pool.")

    print("\n--- 30 случайных строк для аудита дрейфа ---")
    sample_df = df_pairs.sample(n=min(30, len(df_pairs)), random_state=42)
    for idx, (_, r) in enumerate(sample_df.iterrows(), 1):
        print(f"\n[{idx}/30] Project: {r['project']} | Symbol: {r['symbol']}")
        print(f"  sig_old:    {r['sig_old']}")
        print(f"  drift_type: {r['drift_type']} (sig_type: {r['sig_type']})")
        print(f"  doc_old:    {r['doc_old'][:80]}...")

if __name__ == "__main__":
    main()