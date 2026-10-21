# FD-24: операционный гайд экспериментов M2 в Kaggle (2×T4)

> **Статус документа:** Рабочий драфт для ветки `feat/FD-24-model-research` (детализация шага 3 FD-23 — приоритеты экспериментов недели 3).
> **Бюджет GPU:** строго $\le 9$ обязательных прогонов (~4.5–5.0 ч суммарно, ~15–17% недельной квоты Kaggle).
> **Инвариант выбора лучшей модели:** только по PR-AUC на реальных строках `val` (не по тестам и не по синтетике).
> **Ограничение платформы:** 20 ГБ диска на `/kaggle/working`. Обязательно удалять промежуточные чекпоинты сразу после завершения каждого слота.

---

## 0. Архитектура исполнения: 3 независимые сессии (Commit / Run)

Запуск всех 9 слотов в одном интерактивном блокноте **запрещен** из-за риска переполнения диска, таймаута сессии и фрагментации VRAM. Процесс разделен на 3 изолированных сессии (ноутбука) в режиме **Save & Run All**:

```
[Сессия 1: E2 Данные] ────► 4 прогона UniXcoder @1024 (разные смеси) ────► reports/runs.csv (слоты 1-4)
                                                                                   │
[Сессия 2: Бэкбоны]   ────► 3 прогона на полной смеси (ModernBERT, GTE)   ────► reports/runs.csv (слоты 5-7) ──► Выбор лучшего бэкбона
                                                                                   │
[Сессия 3: Финал]     ────► Слот 8 (стэкинг на CPU) + Слот 9 (абляция)    ────► reports/runs.csv (слоты 8-9) ──► config/train_best.yaml
```

---

## 1. Подготовительные шаги в ветке FD-24

Перед стартом запусков в ветке `feat/FD-24-model-research`:
1. Расширить CLI `scripts/train_m2.py`:
   - Флаг `--train_mix` с поддержкой 4 режимов фильтрации обучающей выборки (`split == 'train'`):
     - `real_only`: только реальные источники (`codocbench`, `history_neg`).
     - `synth_only`: только синтетика AST (`synth_ast`).
     - `real_ast`: реальные + AST (`codocbench`, `history_neg`, `synth_ast` без `synth_llm`).
     - `full` (дефолт): полная смесь датасета v1 (включая LLM P1/P2).
   - Флаг `--drop_diff` (`action="store_true"`): подменяет поле `diff_code` на пустую строку `""` перед токенизацией (для слота 9).
   - Флаг `--gradient_checkpointing` (`action="store_true"` или парсер `str2bool`): проброс в `TrainingArguments` (критично для контекста 2048, вызов в CLI: `--gradient_checkpointing`).
2. Создать вспомогательные скрипты для Сессии 3:
   - `scripts/train_stacking.py`: single-split стэкинг M1 + M2 на предсказаниях `val` (слот 8).
   - `scripts/finalize_best_config.py`: авто-выбор лучшего запуска по max PR-AUC на `val` из `reports/runs.csv` и сохранение в `config/train_best.yaml`.

---

## 2. Сессия 1: Эксперимент E2 — Сравнение смесей данных (Слоты 1–4)

- **Модель:** строго `microsoft/unixcoder-base` @1024, `doc_budget: 384`.
- **Окружение:** Kaggle 2×T4, PyTorch DDP (`torchrun --nproc_per_node=2`), FP16, eff_batch 32 (`per_device=8`, `accum=2`).
- **Время сессии:** ~1.5–2.0 часа.

### Команды выполнения в блокноте

```bash
# Клетка 1: Подготовка окружения (аналогично kaggle_fd21_guide.md)
%cd /kaggle/working/FreshDoc
export NCCL_P2P_DISABLE=1

# Слот 1: E2(a) Только реальные данные
!torchrun --nproc_per_node=2 scripts/train_m2.py --config config/train.yaml \
  --data_path "$DATA_PATH" --train_mix real_only \
  --output_dir /kaggle/working/checkpoints/slot1_real --runs_csv reports/runs.csv
!rm -rf /kaggle/working/checkpoints/slot1_real/checkpoint-*

# Слот 2: E2(b) Только синтетика AST
!torchrun --nproc_per_node=2 scripts/train_m2.py --config config/train.yaml \
  --data_path "$DATA_PATH" --train_mix synth_only \
  --output_dir /kaggle/working/checkpoints/slot2_synth --runs_csv reports/runs.csv
!rm -rf /kaggle/working/checkpoints/slot2_synth/checkpoint-*

# Слот 3: E2(c) Реальные + AST
!torchrun --nproc_per_node=2 scripts/train_m2.py --config config/train.yaml \
  --data_path "$DATA_PATH" --train_mix real_ast \
  --output_dir /kaggle/working/checkpoints/slot3_real_ast --runs_csv reports/runs.csv
!rm -rf /kaggle/working/checkpoints/slot3_real_ast/checkpoint-*

# Слот 4: E2(d) Полная смесь (+ LLM-пары)
!torchrun --nproc_per_node=2 scripts/train_m2.py --config config/train.yaml \
  --data_path "$DATA_PATH" --train_mix full \
  --output_dir /kaggle/working/checkpoints/slot4_full --runs_csv reports/runs.csv
!rm -rf /kaggle/working/checkpoints/slot4_full/checkpoint-*
```

---

## 3. Сессия 2: Факториальное сравнение бэкбонов и контекста (Слоты 5–7)

- **Данные:** все модели обучаются **строго на полной смеси (`--train_mix full`)**.
- **Время сессии:** ~2.0–2.5 часа.

### Важные технические требования
1. **Пилот FP16 на ModernBERT:** перед запуском обязателен прогон `--pilot_steps 100`. Если `eval_loss` уходит в NaN — снизить `learning_rate` с `2e-5` до `1e-5`.
2. **Контекст 2048 токенов:** для слотов 6 и 7 обязательно включать `--gradient_checkpointing true`, иначе гарантирован CUDA OOM на 16 ГБ VRAM карты T4.

### Команды выполнения в блокноте

```bash
# Клетка 1: Пилот проверки FP16 на ModernBERT
!python scripts/train_m2.py --config config/train.yaml \
  --data_path "$DATA_PATH" --model_name answerdotai/ModernBERT-base \
  --max_len 1024 --pilot_steps 100 --output_dir /kaggle/working/checkpoints/pilot_modernbert

# Слот 5: ModernBERT-base @1024 (контроль домена: UniXcoder vs ModernBERT)
!torchrun --nproc_per_node=2 scripts/train_m2.py --config config/train.yaml \
  --data_path "$DATA_PATH" --model_name answerdotai/ModernBERT-base \
  --max_len 1024 --train_mix full \
  --output_dir /kaggle/working/checkpoints/slot5_modernbert1024 --runs_csv reports/runs.csv
!rm -rf /kaggle/working/checkpoints/slot5_modernbert1024/checkpoint-*

# Слот 6: GTE-ModernBERT-base @2048 (целевой: парный претрейн + снятие усечения)
!torchrun --nproc_per_node=2 scripts/train_m2.py --config config/train.yaml \
  --data_path "$DATA_PATH" --model_name Alibaba-NLP/gte-modernbert-base \
  --max_len 2048 --gradient_checkpointing --train_mix full \
  --output_dir /kaggle/working/checkpoints/slot6_gte2048 --runs_csv reports/runs.csv
!rm -rf /kaggle/working/checkpoints/slot6_gte2048/checkpoint-*

# Слот 7: ModernBERT-base @2048 (контроль контекста)
# ПРАВИЛО ПРОПУСКА: если слот 6 показал паритет со слотом 5 в пределах шума ДИ, слот 7 пропускается.
!torchrun --nproc_per_node=2 scripts/train_m2.py --config config/train.yaml \
  --data_path "$DATA_PATH" --model_name answerdotai/ModernBERT-base \
  --max_len 2048 --gradient_checkpointing --train_mix full \
  --output_dir /kaggle/working/checkpoints/slot7_modernbert2048 --runs_csv reports/runs.csv
!rm -rf /kaggle/working/checkpoints/slot7_modernbert2048/checkpoint-*
```

---

## 4. Сессия 3: Финализация — Стэкинг и Абляция (Слоты 8–9)

- **Время сессии:** ~45 минут.
- **Вход:** выбор модели-победителя по итогам Сессии 2 (максимальный `eval_pr_auc` на сплите `val`).

### Команды выполнения в блокноте

```bash
# Слот 8: E2b — single-split стэкинг M1 + M2 на val (CPU, ~5-10 мин)
!python scripts/train_stacking.py --dataset "$DATA_PATH" \
  --m1_model /kaggle/working/m1_hgb.joblib \
  --m2_checkpoint /kaggle/working/checkpoints/<ПОБЕДИТЕЛЬ>/best \
  --output_dir /kaggle/working/stacking --runs_csv reports/runs.csv

# Слот 9: E3 — абляция БЕЗ diff_code на лучшем бэкбоне (~25-35 мин GPU)
!torchrun --nproc_per_node=2 scripts/train_m2.py --config config/train.yaml \
  --data_path "$DATA_PATH" --model_name <ПОБЕДИТЕЛЬ_ИМЯ> \
  --max_len <ПОБЕДИТЕЛЬ_MAX_LEN> --drop_diff --train_mix full \
  --output_dir /kaggle/working/checkpoints/slot9_ablation_no_diff --runs_csv reports/runs.csv
!rm -rf /kaggle/working/checkpoints/slot9_ablation_no_diff/checkpoint-*

# Фиксация победителя в конфигурации репозитория
!python scripts/finalize_best_config.py --runs_csv reports/runs.csv \
  --out_config config/train_best.yaml
```

---

## 5. Сводная таблица бюджета слотов FD-24

| Слот | Код | Модель | Длина | Смесь train | Время GPU | Назначение |
|---|---|---|---|---|---|---|
| **1** | E2(a) | UniXcoder-base | 1024 | `real_only` | ~20–25 мин | Базовый вклад реальных данных |
| **2** | E2(b) | UniXcoder-base | 1024 | `synth_only` | ~20–25 мин | Оценка доменного сдвига синтетики |
| **3** | E2(c) | UniXcoder-base | 1024 | `real_ast` | ~20–25 мин | Базовая смесь реальные + AST |
| **4** | E2(d) | UniXcoder-base | 1024 | `full` | ~20–25 мин | Оценка вклада LLM-пар на sig_same |
| **5** | E2c | ModernBERT-base | 1024 | `full` | ~25–35 мин | Контроль претрейна (код vs общий) |
| **6** | E2c | GTE-ModernBERT | 2048 | `full` | ~45–55 мин | Снятие усечения + relevance претрейн |
| **7** | E2c | ModernBERT-base | 2048 | `full` | ~45–55 мин | Контроль контекста (пропуск при паритете 5-6) |
| **8** | E2b | Стэкинг M1+M2 | — | val preds | ~5–10 мин (CPU) | Проверка ансамблирования без утечки |
| **9** | E3 | Лучший бэкбон | 1024/2048 | `full` (no diff) | ~25–35 мин | Абляция признака diff_code |

---

## 6. Что забрать из Kaggle Output в репозиторий

1. Полный файл `reports/runs.csv` (содержит метрики всех 9 запусков).
2. Папку весов лучшей модели: `checkpoints/<ПОБЕДИТЕЛЬ>/best/` (`model.safetensors`, `config.json`, `dualhead_config.json`, токенизатор).
3. Метаданные запусков: `run_meta.json` для каждого слота.
4. Сформированный `config/train_best.yaml`.
