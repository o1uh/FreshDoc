# Как воспроизвести результаты M2 (FD-32)

> Проверено на ветке `feat/FD-32-readme-note`, Python 3.11.9, 26.10.2026. Все команды ниже реально выполнены (`--help`) или прогнаны; ничего выдуманного.

## 0. Клонирование

```bash
git clone https://github.com/o1uh/FreshDoc
cd FreshDoc
```

URL — из [README.md](../README.md) §«Быстрый старт».

## 1. Окружение

Нужен Python 3.11 ([README.md](../README.md) §«Требования», проверено: `python --version` → 3.11.9):

```bash
python3.11 -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

[requirements.txt](../requirements.txt) существует. Важно: `train_m2.py` требует torch — запускать только через `.venv` (системный python даёт `ModuleNotFoundError: No module named 'torch'`, проверено).

## 2. Тесты

```bash
python -m pytest -q
```

Проверено: сбор — 487 тестов, пробный прогон `tests/test_smoke.py` — 2 passed. Конфиг: [pytest.ini](../pytest.ini) (`testpaths = tests`).

## 3. Откуда взять данные

Каталог [data/](../data) внесён в [.gitignore](../.gitignore) (`/data/`, `checkpoints/`, `reports/preds/`) и в git не входит. Два честных пути:

- **(a) Скачать готовый датасет.** Ссылка на Parquet — §4.6.3 [записки](../записка.md) (Kaggle `freshdoc-dataset`, `dataset_v1.parquet`, 27 203 строки). Положить как `data/processed/dataset_v1.parquet`. Промежуточные parquet для пересборки — Kaggle `freshdoc-interim` ([data_notes.md](./data_notes.md) FD-03). Целостность сверить с [fd31_freeze.md](../reports/fd31_freeze.md) §1: 49 066 226 байт, sha256 `cedc8e29…e314e1c`.
- **(b) Пересобрать.** [build_dataset.py](../scripts/build_dataset.py) (`--help` проверен; входы `data/interim/*.parquet`, выходы `--out/--out_stats/--out_manifest`).

Обучение с нуля требует GPU (см. §5).

## 4. Воспроизведение ключевых цифр (без GPU)

Предсказания `reports/preds/m2.csv` (slot4_full, в git не входит, локально лежал) + датасет из §3:

```bash
python scripts/validate_dataset.py --in data/processed/dataset_v1.parquet
python scripts/evaluate.py --dataset data/processed/dataset_v1.parquet --pred m2=reports/preds/m2.csv --out_metrics $env:TEMP\eval.json --out_table $env:TEMP\eval.csv --n_boot 100 --seed 42
```

Проверено: `--help` у [validate_dataset.py](../scripts/validate_dataset.py) и [evaluate.py](../scripts/evaluate.py) отрабатывают; прогон `evaluate.py` (n_boot 100 для скорости) дал test_real PR-AUC 0.984 / F1 0.947, gold PR-AUC 0.478 / F1 0.600 — совпадает с калиброванными цифрами [e4_e5_e6.md](../reports/e4_e5_e6.md) §E4 (0.9837 / 0.9477; gold 0.4776 / 0.600 при сыром пороге).

Важные нюансы:

- Точные CI из отчёта — при `--n_boot 1000 --seed 42` (дефолты скрипта).
- `evaluate.py` по умолчанию **перезаписывает картинки** `reports/figures/e1/*.png` — выводите артефакты во временные файлы (как выше) и откатывайте мусор через `git checkout -- reports/figures/e1/` (проверено, так же сделано в [fd31_freeze.md](../reports/fd31_freeze.md) §5).
- Калибровка: [calibrate_fd25.py](../scripts/calibrate_fd25.py) (`--help` проверен; команда из отчёта: `python scripts/calibrate_fd25.py --pred reports/preds/m2.csv`). Не запускать без флага `--no_update_risk`, если не хотите перезаписать порог в [risk.yaml](../config/risk.yaml).
- Финальные цифры 3 сидов ([fd31_seed42_metrics.json](../reports/fd31_seed42_metrics.json) и др.) получены в Kaggle на 2×T4 — локально на CPU воспроизводится только оценка по готовым предиктам, не обучение.

## 5. Полное обучение (GPU / Kaggle)

```bash
python scripts/train_m2.py --config config/train_best.yaml --data_path data/processed/dataset_v1.parquet --output_dir checkpoints/m2 --seed 42
```

Проверено: `--help` [train_m2.py](../scripts/train_m2.py) отрабатывает (только под `.venv`). Конфиг — [train_best.yaml](../config/train_best.yaml) (seed менять на 42/43/44 для трёх финальных сидов, ~30 мин на 2×T4 каждый — [fd31_freeze.md](../reports/fd31_freeze.md) §6.1). После обучения — оценка §4 и сверка с [fd31_seed42_metrics.json](../reports/fd31_seed42_metrics.json).
