from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.synth.llm_batch import PILOT_SEED, export_pilot_batches, import_pilot_responses


def main() -> None:
    parser = argparse.ArgumentParser(description="FD-10: экспорт и импорт пробного прогона LLM-синтетики.")
    parser.add_argument("--mode", required=True, choices=["export", "import"])
    parser.add_argument("--pool", default="data/interim/functions_pool_v1.parquet")
    parser.add_argument("--dir", default="data/llm_pilot")
    parser.add_argument("--prompts", default="prompts")
    parser.add_argument("--seed", type=int, default=PILOT_SEED)
    parser.add_argument("--report", default="reports/llm_pilot.md")
    parser.add_argument("--out", default="data/interim/synth_llm_pilot.parquet")
    args = parser.parse_args()

    if args.mode == "export":
        df_pool = pd.read_parquet(args.pool)
        info = export_pilot_batches(df_pool, Path(args.dir), Path(args.prompts), seed=args.seed)
        print(f"Выборка: {info['n_functions']} функций из {info['n_projects']} проектов (seed={args.seed})")
        print(f"Проекты: {', '.join(info['projects'])}")
        print(f"Манифест: {info['manifest']}")
        for name in info["prompts"]:
            print(f"Пакет: {Path(args.dir) / name}")
    else:
        stats = import_pilot_responses(Path(args.pool), Path(args.dir), Path(args.report), Path(args.out))
        print(f"Файлов: {stats['files_total']}, без ответов: {stats['files_missing_response']}")
        print(f"Принято P1/P2: {stats['accepted_p1']}/{stats['accepted_p2']}, доля: {stats['accept_rate']}")
        print(f"Причины отказов: {stats['reasons']}")
        print(f"Отчёт: {args.report}")
        print(f"Критерий >=0.40: {'ДА' if stats['accept_rate'] >= 0.40 else 'НЕТ'}")


if __name__ == "__main__":
    main()
