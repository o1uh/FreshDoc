# FreshDoc

Мониторинг актуальности документации по изменениям кода методами машинного обучения.

> 📄 **Пояснительная записка:** [записка.md](./записка.md) — постановка задачи, данные, модели, эксперименты и результаты.
> 🤗 **Модель:** [o1uh/fresh-doc на Hugging Face](https://huggingface.co/o1uh/fresh-doc) — файнтюн `microsoft/unixcoder-base` (см. [записка.md](./записка.md), §5.3).
> 🏛️ Архитектура подробно — [§5 записки](./записка.md#5-архитектура-системы-обучение-моделей-и-программная-реализация).

## 1. Цель

Скоуп: Python 3.9+, уровень docstring функций и методов (Level 1).

- **Что предсказывает модель.** По паре «дифф изменения функции + её docstring до изменения» — вероятность устаревания документации `p_stale` и класс программного дрейфа `drift_type`: переименование / удаление / добавление параметра, смена дефолта, возврата, исключений, поведения (`behavior_changed`).
- **Зачем реестр.** Вердикты модели с порогом копятся в риск по веткам и коммитам и маршрутизируют ревью. Задача закрывает слепую зону ревью и CI: docstring не ломает сборку, поэтому устаревание накапливается молча.
- **Кому полезно.** Ревью (что именно устарело и почему), CI (гейт вместо молчаливого пропуска), IDE / RAG-контекст ИИ-агентов (чистые подсказки и справки вместо отравленных устаревшим docstring).

**Пример** (схематично: переименование параметра, docstring не обновлён):

- *Фрагмент Unified Diff:*
  ```diff
  -def f(path):
  +def f(file_path):
  ```
- *Docstring (до изменения):*
  ```
  """Обработать файл.

  Args:
      path: путь к файлу.
  """
  ```
- *Вердикт:* `p_stale ≈ 1`, `drift_type = param_renamed` — документированного имени больше нет, docstring устарел.

## 2. Архитектура

- **Уровень 1 — Fast-Path (M0).** Детерминированные правила по AST-сигнатуре: мгновенно закрывают грубый слом заголовка (`sig_changed`), физически слепы на семантике (`sig_same`, Recall = 0).
- **Уровень 2 — Deep-Path (M2).** Дообученный энкодер (`microsoft/unixcoder-base`, контекст 1024) с двумя головами; совместное внимание токенов диффа и docstring ловит дрейф при неизменной сигнатуре.
- **Реестр и API.** Вероятности копятся в риск `R = 1 − Π(1 − pᵢ)` до подтверждения человеком (`POST /verify` сбрасывает риск); срочность и влияние (fan-in вызовов) ранжируют очередь ревью.

> Подробно: [§5 «Архитектура системы, обучение моделей и программная реализация»](./записка.md#5-архитектура-системы-обучение-моделей-и-программная-реализация).

## 3. Быстрый старт

Требования: Python 3.11, Git.

### Структура репозитория

- [scripts/](./scripts/) — CLI конвейера (майнинг, сборка, обучение, оценка, калибровка, replay).
- [src/](./src/) — исходный код (`data`, `mining`, `synth`, `models`, `risk`, `registry`, `api`, `dashboard`).
- [config/](./config/) — конфиги (`train_best.yaml`, `risk.yaml`, маппинги).
- [docs/](./docs/) — гайды, карточки данных и модели.
- [reports/](./reports/) — метрики, таблицы, графики (артефакты заморозки).
- [notebooks/](./notebooks/) — Kaggle-блокноты обучения.
- `data/`, `checkpoints/`, `reports/preds/` — в git не входят (см. [.gitignore](./.gitignore)).

CI: [.github/workflows/ci.yml](./.github/workflows/ci.yml) — `pytest` на каждый push/PR в `main`.

### 3.0. Клон

```bash
git clone https://github.com/o1uh/FreshDoc
cd FreshDoc
```

### 3.1. Окружение

```bash
python3.11 -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

Зависимости — [requirements.txt](./requirements.txt), версии запинованы (`torch==2.14.1`, `transformers==5.19.0`). `train_m2.py` требует torch — запускать только через `.venv`. 

На Kaggle `requirements.txt` не ставить (ломает связку torch/torchvision базового образа) — см. [docs/kaggle_fd31_guide.md](./docs/kaggle_fd31_guide.md). Подробнее — [docs/how_to_reproduce.md](./docs/how_to_reproduce.md).

### 3.2. Проверка

```bash
python -m pytest -q
```

Конфиг — [pytest.ini](./pytest.ini) (`testpaths = tests`). Ожидается ~487 тестов, проверено 26.10.2026 (см. [docs/how_to_reproduce.md](./docs/how_to_reproduce.md)).

### 3.3. Данные

Каталог `data/` в git не входит (см. [.gitignore](./.gitignore)). Два пути:

- **(a) Скачать готовый корпус.** Kaggle `freshdoc-dataset`, файл `dataset_v1.parquet` (27 203 строки) — положить как `data/processed/dataset_v1.parquet`. Промежуточные parquet для пересборки — Kaggle `freshdoc-interim`. Ссылки — §4.6.3 [записки](./записка.md); состав и сплиты (train 21 997 / val 559 / test_real 2 895 / test_synth 1 552 / gold 200) — [docs/data_card.md](./docs/data_card.md). Целостность — [reports/fd31_freeze.md](./reports/fd31_freeze.md) §1 (49 066 226 байт, sha256 `cedc8e29…e314e1c`).
- **(b) Пересобрать.** [scripts/build_dataset.py](./scripts/build_dataset.py) (`--out` / `--out_stats` / `--out_manifest`).

### 3.4. Бейзлайны и оценка (без GPU)

```bash
python scripts/validate_dataset.py --in data/processed/dataset_v1.parquet
python scripts/baselines_m0_m1.py --dataset data/processed/dataset_v1.parquet
python scripts/evaluate.py --dataset data/processed/dataset_v1.parquet --pred m2=reports/preds/m2.csv --n_boot 100 --seed 42
```

- [scripts/validate_dataset.py](./scripts/validate_dataset.py) — 16 проверок датасета.
- [scripts/baselines_m0_m1.py](./scripts/baselines_m0_m1.py) — бейзлайны M0 (правила) и M1 (бустинг).
- [scripts/evaluate.py](./scripts/evaluate.py) — PR-AUC / F1 по сплитам. Точные CI — при `--n_boot 1000` (дефолт); `--n_boot 100` — быстрая проверка. Предикты `reports/preds/m2.csv` в git не входят.

### 3.5. Обучение (только GPU)

```bash
python scripts/train_m2.py --config config/train_best.yaml --data_path data/processed/dataset_v1.parquet --output_dir checkpoints/m2 --seed 42
```

- Скрипт — [scripts/train_m2.py](./scripts/train_m2.py); конфиг — [config/train_best.yaml](./config/train_best.yaml) (`microsoft/unixcoder-base`, контекст 1024, `lambda_drift=0.6`).
- Для трёх финальных сидов менять только `--seed 42/43/44` (~27 мин на 2×T4 каждый). Пошаговый гайд — [docs/kaggle_fd31_guide.md](./docs/kaggle_fd31_guide.md).

### 3.6. Калибровка

```bash
python scripts/calibrate_fd25.py --pred reports/preds/m2.csv
```

Скрипт — [scripts/calibrate_fd25.py](./scripts/calibrate_fd25.py). Результат — [config/risk.yaml](./config/risk.yaml): `T = 2.3847`, `threshold = 0.8743`. Источник цифр — [reports/calibration.json](./reports/calibration.json), [reports/e4_e5_e6.md](./reports/e4_e5_e6.md).

### 3.7. Работа с моделью

Готовая M2 — [o1uh/fresh-doc на Hugging Face](https://huggingface.co/o1uh/fresh-doc). Паспорт — [docs/model_card.md](./docs/model_card.md) (вход `diff_code` + `doc_old`, выход `p_stale` + `drift_type`).

Вариант A — напрямую через `transformers`:

```python
from transformers import AutoTokenizer, AutoModel
tok = AutoTokenizer.from_pretrained("o1uh/fresh-doc")
model = AutoModel.from_pretrained("o1uh/fresh-doc")
```

Вариант B — скриптом инференса [scripts/predict_m2.py](./scripts/predict_m2.py):

```bash
python scripts/predict_m2.py --checkpoint checkpoints/m2 --dataset data/processed/dataset_v1.parquet --out reports/preds/m2.csv
```

Выход CSV: `pair_id, p_stale, type_pred / type_pred_gated` (`type_pred` — сырой argmax головы дрейфа, `type_pred_gated` — `p_stale > tau`, иначе `none`). Рабочий порог `tau` — из [config/risk.yaml](./config/risk.yaml).

## 4. Результаты


**M0** — эвристики сигнатуры; 

**M1** — GBDT на 21 признаке; 

**M2** — `microsoft/unixcoder-base` @1024, mean±std по 3 сидам 42/43/44 (freeze-1, FD-31); 

**M3 / M3max** — Claude Haiku 5.5 (base / effort=max). Основной язык — Python 3.9+; кросс-язык — Java, Python 2, Go.

### 4.1. Главная таблица E1

| Сплит (n / pos) | M0 (эвристики) | M1 (GBDT) | M2 (UniXcoder, mean±std 42/43/44) | M3 (LLM-судья) |
|---|---|---|---|---|
| `val` (559 / 211) | 0.740 [0.696, 0.783] / 0.737 | 0.862 [0.825, 0.895] / 0.754 | **0.9677±0.0026** / 0.9170±0.0058 | — * |
| `test_real` (2 895 / 1 444) | 0.763 [0.746, 0.778] / 0.690 | 0.879 [0.867, 0.892] / 0.746 | **0.9782±0.0016** / 0.9306±0.0090 | — * |
| `test_synth` (1 552 / 746) | 1.000 / 1.000 | 1.000 / 0.998 | 1.000±0.000 / 1.000 | — * |
| `gold` (200 / 55) | 0.442 [0.346, 0.553] / 0.605 | 0.546 [0.427, 0.687] / 0.580 | 0.4894±0.0323 / 0.5923±0.0176 | 0.902 [0.833, 0.957] / 0.773 |

\* M3 оценён только на 500 строках (300 `test_real` + весь `gold`); вне `gold` прямое сравнение с полными сплитами некорректно.

Источник: [§6.1.1 записки](./записка.md#611-сводные-результаты-бейзлайнов-и-моделей-по-сплитам-val-test_real-test_synth-gold). Артефакты: [reports/fd31_seeds_summary.csv](./reports/fd31_seeds_summary.csv), [reports/e1_v0.md](./reports/e1_v0.md).

### 4.2. Семантический дрейф `sig_same`

| Сплит `sig_same` (n / pos, prev) | M0 | M1 | M2 (mean±std 42/43/44) |
|---|---|---|---|
| `val` (436 / 88, prev 0.202) | 0.202 [= prevalence] | 0.342 [0.266, 0.441] | **0.8816±0.0078** |
| `test_real` (2 134 / 683, prev 0.320) | 0.320 [= prevalence] | 0.478 [0.441, 0.518] | **0.9375±0.0021** |
| `gold` (103 / 9, prev 0.087) | 0.087 [= prevalence] | 0.265 [0.064, 0.556] | 0.2530±0.0603 (шум, n_pos = 9) |

Источник: [§6.1.2 записки](./записка.md#612-оценка-качества-на-изолированном-подмножестве-семантического-дрейфа-sig_same).

M0 на `sig_same` физически слеп (= prevalence); M2 держит 0.9375 на `test_real`, M1 — только 0.478. Оценки на `gold sig_same` (n_pos = 9) — шум.

### 4.3. Честное M3-подмножество

| Группа общего подмножества (n / pos) | M0 | M1 | M2-fd31 | M3 | M3max |
|---|---|---|---|---|---|
| `test_real_subset300 all` (300 / 145) | 0.779 [0.729, 0.826] / 0.728 | 0.890 [0.849, 0.922] / 0.784 | **0.9768 [0.9600, 0.9909]** / **0.934** | 0.886 [0.843, 0.923] / 0.791 | 0.878 [0.827, 0.915] / 0.784 |
| `test_real_subset300 sig_same` (217 / 62) | 0.286 / 0.000 | 0.466 [0.349, 0.584] / 0.431 | **0.9310 [0.8805, 0.9703]** / **0.864** | 0.667 [0.561, 0.766] / 0.600 | 0.649 [0.533, 0.766] / 0.606 |
| `gold all` (200 / 55) | 0.442 [0.346, 0.553] / 0.605 | 0.546 [0.427, 0.687] / 0.580 | 0.4776 [0.3878, 0.6111] / 0.617 | 0.902 [0.833, 0.957] / 0.773 | **0.920 [0.860, 0.961]** / **0.828** |

Источник: [§6.1.4 записки](./записка.md#614-честное-сопоставление-подходов-на-общем-m3-подмножестве). Артефакты: [reports/tables/e1_m3subset_fd31.csv](./reports/tables/e1_m3subset_fd31.csv), [reports/tables/m3_select.json](./reports/tables/m3_select.json).

### 4.4. Бэкбоны и контекст (слоты 4–8, односеедовый v0; актуальные числа — §4.1)

| Слот | Бэкбон | Окно | Время обучения | PR-AUC `val_real` [95% CI] | PR-AUC `test_real` | PR-AUC `gold` | Macro-F1 типирования |
|---|---|---|---|---|---|---|---|
| **4** | **UniXcoder-base** | **1024** | **27.1 мин** | **0.9724** [0.957, 0.985] | **0.9837** | 0.4776 | **0.8121** |
| 5 | ModernBERT-base | 1024 | 46.4 мин | 0.9695 [0.953, 0.983] | 0.9751 | 0.4815 | 0.6703 |
| 6 | GTE-ModernBERT | 2048 | 77.8 мин | 0.9604 [0.940, 0.976] | 0.9653 | **0.6289** | 0.6028 |
| 7 | ModernBERT-base | 2048 | 76.2 мин | 0.9669 [0.950, 0.980] | 0.9708 | 0.5504 | 0.6828 |
| 8 | Stacking (M1 + M2) | — | 0.5 с (CPU) | 0.9581 [0.938, 0.974] | 0.9713 | 0.6231 | 0.8121 |

Источник: [§6.4 записки](./записка.md#64-исследование-масштабирования-контекста-и-бэкбонов-слоты-48). Артефакты: [reports/tables_e1_all_slots.csv](./reports/tables_e1_all_slots.csv).

UniXcoder @1024 — лучший на `test_real` и по Macro-F1 типирования при минимальном времени; окно 2048 выигрыша на production-срезе не даёт.

### 4.5. Обобщение, калибровка и порог

| Сплит / Когорта | Объём (n / pos) | PR-AUC [95% CI] | F1 | Precision | Recall | Brier |
|---|---|---|---|---|---|---|
| `val` | 559 / 211 | 0.9724 [0.957, 0.985] | 0.9242 | 0.9242 | 0.9242 | 0.0905 |
| `val / sig_same` | 436 / 88 | 0.9043 [0.856, 0.948] | 0.8249 | 0.8202 | 0.8295 | 0.1155 |
| `test_real` | 2 895 / 1 444 | **0.9837** [0.980, 0.987] | **0.9477** | **0.9682** | **0.9280** | **0.0744** |
| `test_real / sig_same` | 2 134 / 683 | **0.9522** [0.941, 0.963] | **0.8968** | **0.9307** | **0.8653** | **0.1000** |

Источник: [§6.5 записки](./записка.md#65-эксперимент-e4-обобщение-калиброванной-модели-на-невиданные-ранее-проекты-val---test_real). Артефакт: [reports/e4_e5_e6.md](./reports/e4_e5_e6.md).

| Сплит | Brier (до → после) | NLL (до → после) | ECE (до → после) |
|---|---|---|---|
| `val real` | 0.0990 → **0.0905** | 0.4339 → **0.2998** | 0.1061 → 0.1262 |
| `test_real` | 0.0804 → **0.0744** | 0.3320 → **0.2484** | 0.0890 → 0.0949 |
| `gold` | 0.3674 → **0.3290** | 2.5097 → **1.1577** | 0.3752 → **0.3668** |

Источник: [§6.6 записки](./записка.md#66-эксперимент-e5-калибровка-вероятностных-оценок). Артефакт: [reports/calibration.json](./reports/calibration.json) (T = 2.3847).

| Порог τ | Precision | Recall | F1 | Статус |
|---|---|---|---|---|
| 0.5000 | 0.7660 | 0.9621 | 0.8529 | Избыток ложных срабатываний |
| 0.8000 | 0.8945 | 0.9242 | 0.9091 | Жёсткий режим CI |
| **0.8743** | **0.9242** | **0.9242** | **0.9242** | **Рабочая точка (середина плато)** |
| 0.9500 | 0.9883 | 0.8009 | 0.8848 | Экстремальное подавление алертов |

Источник: [§6.7 записки](./записка.md#67-эксперимент-e6-оптимизация-операционного-порога-принятия-решений). Артефакт: [reports/tables/e6_thresholds.csv](./reports/tables/e6_thresholds.csv) (полная сетка 0.05–0.95 — там).

Просадки при переносе val → `test_real` нет (PR-AUC 0.9724 → 0.9837). Рабочий порог τ\* = 0.8743 — середина плато максимума F1 при Precision ≥ 0.80.

### 4.6. Кросс-язык (zero-shot, τ = 0.8743, без дообучения)

| Бенчмарк | Язык | N | Позитивов | PR-AUC [95% CI] | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| **HebCup** | Java | 8 278 | 8 278 | — | — | 0.4346 | 1.0000 | 0.2776 |
| **Panthaplackel JIT** | Java | 3 944 | 1 972 | **0.6207** [0.598, 0.644] | 0.5924 | 0.3463 | 0.7285 | 0.2272 |
| **Python 2 CoCC** | Python 2 | 4 957 | 2 072 | **0.5790** [0.556, 0.602] | 0.6266 | 0.3712 | 0.6799 | 0.2553 |
| **Go Silver** | Go | 380 | 27 | **0.2787** [0.155, 0.471] | 0.7662 | 0.3529 | 0.3750 | 0.3333 |

Источник: [§6.10.3 записки](./записка.md#6103-экспериментальные-результаты-zero-shot-инференса). Артефакты: [reports/tables/external.csv](./reports/tables/external.csv), [reports/external_check.md](./reports/external_check.md).

Модель, обученная только на Python 3, частично переносится: Java-JIT 0.621, Python 2 0.579, Go 0.279 при ROC-AUC 0.766 (широкий CI, n_pos = 27).

### 4.7. Задержки и экономика

| Модуль / Конфигурация | Mean (мс) | p50 (мс) | p95 (мс) | Среда |
|---|---|---|---|---|
| **M0: AST-эвристика (CPU)** | **0.0215** | 0.0205 | 0.0262 | 1 поток CPU |
| **M1: GBDT (CPU, 21 признак)** | **5.227** | 4.863 | 6.716 | 6 потоков CPU |
| **M2: GPU Single** | **10.82** | 10.78 | 11.28 | GTX 1070 (FP16) |
| **M2: GPU Batch-16 (на элемент)** | **5.789** | 5.790 | 5.828 | GTX 1070 (FP16) |
| **M2: CPU fallback** | **77.54** | 76.74 | 83.53 | 6 потоков CPU |
| *Cloud API Claude Haiku 5.5 (M3)* | ~2 600 | ~2 400 | ~3 400 | Внешний REST API |

Источник: [§6.11.2 записки](./записка.md#6112-сравнительный-анализ-задержек-инференса-cpu-vs-gpu-vs-cloud-api). Артефакт: [reports/latency_profile.json](./reports/latency_profile.json).

| Решение | Режим | Стоимость на 10k проверок |
|---|---|---|
| **M2 self-hosted** | **GPU Batch-16** | **`$0.001`** |
| DeepSeek-V4 / Flash | Batch (ночной) | `$0.588` |
| Claude Haiku 5.5 | Batch API (−50%) | `$0.750` |
| Claude Haiku 5.5 | Прямой API (судья M3) | `$1.500` |
| Claude Haiku 5.5 (M3max) | Прямой API + CoT | `$19.350` |

Источник: [§6.11.3 записки](./записка.md#6113-экономическая-смета-стоимость-локального-инференса-против-облачных-llm) (полная смета из 20+ тарифов — в табл. 6.12 записки).

**Выводы.** На живых данных ранжирование строго M2 > M1 > M0 при непересекающихся CI; решающий вклад — срез `sig_same`, недоступный правилам. LLM-судьи ближе к консенсусу человека на `gold`, но проигрывают M2 на полном `test_real` и дороже на порядки (M2 — $0.001 против $0.75–1.50 у судей). Перенос на другие языки требует своих парсеров и дообучения.

## 5. Ограничения

- Шум разметки до 45% на `test_real` (скрытый дрейф в репозиториях; согласие экспертов κ = 0.72, аудит ошибок κ = 0.674).
- AST-синтетика в одиночку коллапсирует на `sig_same` (PR-AUC 0.19) — не заменяет реальные коммиты.
- Python-only, контекст 1024 (усечение ~4.4% длинных функций; doc-бюджет 384).
- Дельты метрик ≤ 0.005 — в зоне DDP-шума ±0.007, нужны сиды и бутстрэп-CI.

Подробно — [§7.3 записки](./записка.md#73-системные-ограничения-разработанного-решения).

## 6. Документация

**Пояснительная записка:**

- [записка.md](./записка.md) — полный текст (постановка, данные, модели, E1–E7, replay, кросс-язык, экономика, ограничения).
- [docs/note/содержание.md](./docs/note/содержание.md) — исходная спецификация структуры записки (рабочие материалы).

**Воспроизведение:**

- [docs/how_to_reproduce.md](./docs/how_to_reproduce.md) — воспроизведение от чистого клона.
- [docs/colab.md](./docs/colab.md) — запуск обучения на GPU (Kaggle/Colab).
- [docs/kaggle_fd31_guide.md](./docs/kaggle_fd31_guide.md) — финальные прогоны на 3 сидах (freeze-1).
- [docs/kaggle_fd24_guide.md](./docs/kaggle_fd24_guide.md) — операционный гайд экспериментов FD-24.
- [docs/data_notes.md](./docs/data_notes.md) — журнал чисел и решений по данным.

**Данные и модель:**

- [docs/data_card.md](./docs/data_card.md) — источники, лицензии, объёмы, правила меток, известные проблемы.
- [docs/model_card.md](./docs/model_card.md) — назначение, данные, метрики, ограничения модели M2.
- [docs/licenses.md](./docs/licenses.md) — лицензионная чистота весов и данных.

**Отчёты:**

- [reports/e1_v0.md](./reports/e1_v0.md) — сводный бенчмаркинг M0/M1/M2/M3.
- [reports/e4_e5_e6.md](./reports/e4_e5_e6.md) — калибровка, порог, обобщение.
- [reports/error_analysis.md](./reports/error_analysis.md) — аудит ошибок E7.
- [reports/external_check.md](./reports/external_check.md) — кросс-языковая проверка.
- [reports/gold_report.md](./reports/gold_report.md) — протокол Gold Standard.

## 7. Соглашения по разработке (R15)

- Ветки: `feat/FD-xx-описание`, `fix/FD-xx-описание`
- Коммиты: `<type>(<scope>): <описание>` с `Refs: FD-xx` в теле
- Слияние в `main` строго через PR с >= 1 approve и зеленым CI
