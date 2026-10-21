"""FD-24 шаг 5: выбор лучшей конфигурации по val PR-AUC и запись в config/train_best.yaml.

    python scripts/finalize_best_config.py --runs_csv reports/runs.csv \
        --base_config config/train.yaml --out_config config/train_best.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

DEFAULT_RUNS_CSV = PROJECT_ROOT / "reports" / "runs.csv"
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "config" / "train.yaml"
DEFAULT_OUT_CONFIG = PROJECT_ROOT / "config" / "train_best.yaml"


def has_checkpoint_artifacts(output_dir: str | None) -> bool:
    """Проверяет, существуют ли артефакты чекпоинта (директория best, preds.csv или run_meta.json)."""
    if not output_dir or pd.isna(output_dir):
        return False
    d = Path(str(output_dir))
    if (d / "best").exists() or (d / "preds.csv").exists() or (d / "run_meta.json").exists():
        return True
    import glob
    name = d.name
    if glob.glob(f"/kaggle/input/**/{name}/best", recursive=True) or glob.glob(f"/kaggle/input/**/{name}/preds.csv", recursive=True):
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="FD-24 шаг 5: финализация лучшей конфигурации M2 по val.")
    ap.add_argument("--runs_csv", default=str(DEFAULT_RUNS_CSV),
                    help="путь к reports/runs.csv")
    ap.add_argument("--base_config", default=str(DEFAULT_BASE_CONFIG),
                    help="путь к базовому config/train.yaml")
    ap.add_argument("--out_config", default=str(DEFAULT_OUT_CONFIG),
                    help="куда записать итоговый config/train_best.yaml")
    ap.add_argument("--include_stacking", action="store_true",
                    help="разрешить выбор stacking_m1_m2 как лучшего конфига (по умолчанию False, т.к. train_best.yaml конфигурирует обучение нейросетевого бэкбона)")
    ap.add_argument("--include_ablation", action="store_true",
                    help="разрешить выбор моделей с drop_diff=True (по умолчанию False, т.к. абляция без diff не является рабочей моделью детекции)")
    ap.add_argument("--require_existing_dir", action="store_true",
                    help="строго требовать физического существования чекпоинта на диске")
    args = ap.parse_args()

    runs_path = Path(args.runs_csv)
    if not runs_path.exists():
        print(f"ОШИБКА: файл {runs_path} не существует. Сначала выполните прогоны обучения.", file=sys.stderr)
        return 2

    df_runs = pd.read_csv(runs_path)
    if len(df_runs) == 0:
        print(f"ОШИБКА: {runs_path} пуст.", file=sys.stderr)
        return 2

    # Приведение eval_pr_auc к числу и фильтрация невалидных строк
    df_runs["eval_pr_auc"] = pd.to_numeric(df_runs["eval_pr_auc"], errors="coerce")
    valid_runs = df_runs.dropna(subset=["eval_pr_auc"]).copy()

    if not args.include_stacking:
        valid_runs = valid_runs.loc[~valid_runs["model_name"].astype(str).str.startswith("stacking")].copy()

    if not args.include_ablation and "drop_diff" in valid_runs.columns:
        is_ablation = valid_runs["drop_diff"].astype(str).str.strip().str.lower().isin(["true", "1"])
        valid_runs = valid_runs.loc[~is_ablation].copy()

    # Фильтрация legacy-запусков из репозитория, чьих чекпоинтов нет на диске (например, m2_run1 из FD-21)
    exists_mask = valid_runs["output_dir"].apply(has_checkpoint_artifacts)
    if args.require_existing_dir or exists_mask.any():
        if exists_mask.any():
            n_dropped = int((~exists_mask).sum())
            if n_dropped > 0:
                print(f"Фильтрация: исключено {n_dropped} прогонов без физических артефактов на диске (legacy/недоступные чекпоинты)")
                valid_runs = valid_runs.loc[exists_mask].copy()
        elif args.require_existing_dir:
            print("ОШИБКА: --require_existing_dir указан, но ни один чекпоинт не найден на диске.", file=sys.stderr)
            return 2

    if len(valid_runs) == 0:
        print("ОШИБКА: нет валидных прогонов нейросетевых моделей с eval_pr_auc в runs.csv.", file=sys.stderr)
        return 2

    # Сортировка по PR-AUC убыванию
    ranked = valid_runs.sort_values(by=["eval_pr_auc", "timestamp"], ascending=[False, False]).reset_index(drop=True)

    print("=== Ранжирование прогонов FD-24 по val_real PR-AUC ===")
    for idx, row in ranked.iterrows():
        print(f"  {idx+1}. PR-AUC={row['eval_pr_auc']:.5f} | model={row['model_name']} "
              f"| len={row.get('max_len', 1024)} | mix={row.get('train_mix', 'full')} "
              f"| drop_diff={row.get('drop_diff', False)} | dir={row['output_dir']}")

    winner = ranked.iloc[0].to_dict()
    print("\n>>> ПОБЕДИТЕЛЬ FD-24 (выбран строго по PR-AUC на реальных строках val):")
    print(f"    Модель: {winner['model_name']}")
    print(f"    Контекст: {winner.get('max_len', 1024)}")
    print(f"    Train mix: {winner.get('train_mix', 'full')}")
    print(f"    Drop diff: {winner.get('drop_diff', False)}")
    print(f"    val_real PR-AUC: {winner['eval_pr_auc']:.5f}")
    print(f"    Директория чекпоинта: {winner['output_dir']}")

    # Загрузка базового конфига
    base_cfg = {}
    base_path = Path(args.base_config)
    if base_path.exists():
        base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}

    # Чтение детального run_meta.json победителя, если доступен
    meta_path = Path(winner["output_dir"]) / "run_meta.json"
    if not meta_path.exists():
        import glob
        cands = glob.glob(f"/kaggle/input/**/{Path(winner['output_dir']).name}/run_meta.json", recursive=True)
        if cands:
            meta_path = Path(cands[0])
    winner_meta = {}
    if meta_path.exists():
        try:
            winner_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            print(f"    Детали прогона загружены из {meta_path}")
        except Exception as e:
            print(f"    Предупреждение: не удалось прочитать {meta_path}: {e}")

    # Формирование итогового конфига
    best_cfg = dict(base_cfg)
    best_cfg["model_name"] = str(winner["model_name"])
    best_cfg["max_len"] = int(winner.get("max_len", base_cfg.get("max_len", 1024)))
    best_cfg["train_mix"] = str(winner.get("train_mix", "full"))
    val_dd = winner.get("drop_diff", False)
    best_cfg["drop_diff"] = str(val_dd).strip().lower() in ("true", "1") if pd.notna(val_dd) else False

    if winner_meta:
        if "doc_budget" in winner_meta:
            best_cfg["doc_budget"] = int(winner_meta["doc_budget"])
        if "lr" in winner_meta:
            best_cfg["learning_rate"] = float(winner_meta["lr"])
        if "lambda_drift" in winner_meta:
            best_cfg["lambda_drift"] = float(winner_meta["lambda_drift"])
        if "gradient_checkpointing" in winner_meta:
            best_cfg["gradient_checkpointing"] = bool(winner_meta["gradient_checkpointing"])

    # Инвариант: для моделей с max_len >= 2048 gradient_checkpointing обязателен
    if best_cfg["max_len"] >= 2048:
        best_cfg["gradient_checkpointing"] = True

    # Формирование комментариев заголовка
    header_comment = f"""# ==============================================================================
# FD-24: Итоговая конфигурация лучшей модели M2
# Сформировано: {datetime.now(timezone.utc).isoformat()}
# Инвариант выбора: строго по max PR-AUC на реальных строках val (не по тестам)
# Правило ничьих: при равенстве PR-AUC побеждает более поздний запуск (по timestamp).
#
# Победитель:
#   Модель: {best_cfg['model_name']}
#   Длина контекста: {best_cfg['max_len']}
#   Смесь обучения: {best_cfg['train_mix']}
#   Абляция drop_diff: {best_cfg['drop_diff']}
#   PR-AUC на val_real: {winner['eval_pr_auc']:.5f}
#   Источник запуска: {winner['output_dir']}
# ==============================================================================
"""

    yaml_body = yaml.dump(best_cfg, sort_keys=False, allow_unicode=True, indent=2)
    out_path = Path(args.out_config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header_comment + "\n" + yaml_body, encoding="utf-8")
    print(f"\nИтоговый конфиг сохранен: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
