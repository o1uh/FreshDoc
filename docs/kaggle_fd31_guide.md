# FD-31: гайд финальных прогонов best-конфига на 3 сидах в Kaggle

> Блокнот: [`notebooks/kaggle_fd31_final3seeds.ipynb`](../notebooks/kaggle_fd31_final3seeds.ipynb).
> Конфиг заморожен: `config/train_best.yaml` (slot4_full: UniXcoder-base, 1024, full, lr 2e-5, 3 эпохи, eff_batch 32).
> Инвариант: отбор только по PR-AUC реальных строк `val`; `test_real`/`gold` — отчётность. Пороги — FD-25, здесь не трогаем.
> Подробности заморозки: `reports/fd31_freeze.md`.

## 0. Что залить в Kaggle

1. **Датасет `freshdoc-dataset`** (read-only, `+ Add Input`): файл `dataset_v1.parquet` — 27 203 строки
   (`train: 21997 / val: 559 / test_real: 2895 / test_synth: 1552 / gold: 200`).
   Блокнот находит его сам через `/kaggle/input/**/dataset_v1.parquet`, путь кладёт в `$DATA_PATH`.
2. **Код и `train_best.yaml` — из git**, заливать не нужно: блокнот клонирует репозиторий
   и пинится на коммит `c512dd7` (freeze-1: финал 3 сидов).
3. **Секреты (Add-ons → Secrets):** `GITHUB_TOKEN` (read-доступ к репозиторию).
4. **Настройки сессии:** Accelerator **GPU T4 x2**, Internet **Always ON**.

## 1. Порядок ячеек

| # | Ячейка | Что делает | Время |
|---|---|---|---|
| 0 | Заголовок (markdown) | Контекст, инвариант, бюджет | — |
| 1 | Env setup | `nvidia-smi`, clone + checkout `c512dd7`, `NCCL_P2P_DISABLE=1`, guard transformers≥4.48 **только с `--no-deps`** (рецепт FD-24), smoke-check `import torch, transformers; from transformers import Trainer` | ~1 мин |
| 2 | Данные | Поиск parquet в `/kaggle/input`, assert 27 203 + сплиты, `validate_dataset.py` (16 проверок) | ~1 мин |
| 3 | M0/M1 | `baselines_m0_m1.py` → `/kaggle/working/m0.csv`, `m1.csv` (вход для `evaluate.py`) | ~1 мин |
| 4 | Обучение ×3 | Цикл: `torchrun --nproc_per_node=2 scripts/train_m2.py --config config/train_best.yaml --seed 42/43/44 --output_dir /kaggle/working/checkpoints/fd31_seed$S`; `rm checkpoint-*` после каждого | ~27 мин × 3 |
| 5 | Predict+evaluate ×3 | `predict_m2.py` → `preds.csv`, `evaluate.py --n_boot 1000` → `metrics.json`, `table_e1.csv`, `figures_e1` | ~3–5 мин × 3 |
| 6 | Сводка | Таблица PR-AUC по сидам + **mean±std** по `val` / `test_real` / `gold` (group=all) → `/kaggle/working/fd31_seeds_summary.csv` | секунды |
| 7 | Упаковка | `tar.gz` артефактов для скачивания | ~1 мин |

Запуск: сверху вниз, **Save & Run All** или по ячейкам. Перезапуск ячейки 4 после падения безопасен:
`train_m2.py` дописывает строку в `reports/runs.csv` только по завершении прогона.

> **Окружение — рецепт FD-24 (как в `kaggle_fd24_*.ipynb` и FD-21):** ячейка env **не ставит**
> `requirements.txt` и не пинит `torchvision`/`torchaudio`. `requirements.txt` поднимает torch
> до 2.14.1 и разбивает согласованную связку torch/torchvision/torchaudio базового GPU-образа
> Kaggle (`transformers.audio_utils` падает на `import torchaudio`, за ним —
> `from transformers import Trainer` в `train_m2.py`). Поэтому torch/torchvision/torchaudio/
> transformers остаются из базового образа — на нём же обучался slot4_full (`train_best.yaml`),
> совместимость гарантирована. Единственная допустимая установка — guard
> `transformers>=4.48` строго с `--no-deps`, чтобы не задеть torch-стек. Доп. uninstall
> пакетов и переменные `TRANSFORMERS_NO_*` в FD-24 не использовались и здесь не нужны.
> Smoke-check ячейки (`import torch, transformers; from transformers import Trainer`) обязан проходить.

## 2. Ожидаемое время

- Эталон slot4 (тот же конфиг, тот же GPU): `train_time_sec=1627.5` ≈ **27 мин/прогон**.
- Итого: 3 × 27 мин + predict/evaluate ≈ **~1.5 ч** в одной интерактивной сессии (лимит Kaggle — 9 ч).
- Диск: 3 × ~500 МБ `best/` + предикты/метрики ≈ **~2 ГБ** из 20 ГБ.

## 3. Что скачать обратно (Output)

- `fd31_final3seeds_artifacts.tar.gz`: `checkpoints/fd31_seed{42,43,44}/` (`best/`, `preds.csv`, `metrics.json`, `table_e1.csv`),
  `reports/runs.csv` (3 новые строки), `fd31_seeds_summary.csv`, `m0.csv`, `m1.csv`.

## 4. Что коммитить локально (после скачивания)

1. Распаковать: `checkpoints/fd31_seed*/best/` → `checkpoints/` (в `.gitignore`, в коммит не входят),
   `preds.csv`/`metrics.json`/`table_e1.csv` — рядом с чекпоинтами.
2. Вручную дописать 3 строки сидов в `reports/runs.csv` (или влить из архива) — `runs.csv` в git входит.
3. Обновить `reports/fd31_freeze.md` (§6): фактические PR-AUC по 3 сидам + mean±std.
4. Коммит в ветку `feat/FD-31-freeze-final-runs`, **без весов и parquet** (проверить `git status`).
