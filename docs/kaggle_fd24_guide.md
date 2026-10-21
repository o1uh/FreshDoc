# FD-24: операционный гайд экспериментов M2 в Kaggle (2×T4)

> **Статус:** Инструменты и блокноты готовы для ветки `feat/FD-24-kaggle-experiments`.
> **Бюджет GPU:** строго $\le 9$ обязательных прогонов (~4.5–5.0 ч суммарно, ~15–17% недельной квоты Kaggle).
> **Инвариант выбора лучшей модели:** только по PR-AUC на реальных строках `val` (не по тестам и не по синтетике). Абляции (`drop_diff=True`) и стэкинг исключаются автоматически.
> **Ограничение диска:** 20 ГБ на `/kaggle/working`. Промежуточные чекпоинты (`checkpoint-*`) удаляются автоматически после каждого слота; сохраняются только `best/` (~500 МБ) и артефакты оценки (`preds.csv`, `metrics.json`, `table_e1.csv`, `figures_e1`).

---

## 0. Архитектура исполнения и блокноты

Поддерживаются два полноценных сценария запуска:

### Сценарий А: Единый мастер-блокнот [`notebooks/kaggle_fd24_master.ipynb`](../notebooks/kaggle_fd24_master.ipynb) (в 1 сессии)
* **Потребление диска:** Диск **не переполнится** (лимит 20 ГБ не будет превышен).
  * Датасет лежит в `/kaggle/input/` (read-only, **0 байт** из квоты working).
  * В конце каждого слота промежуточные чекпоинты удаляются (`!rm -rf checkpoint-*`).
  * Сохраняется только папка `best/` (~500 МБ на модель).
  * Все 8 нейросетевых слотов с весами `best/` займут суммарно: $8 \times 500 \text{ МБ} \approx 4.0 \text{ ГБ}$.
  * Таблицы метрик и предсказания: $< 100 \text{ МБ}$.
  * **Пиковое заполнение диска:** ~4.5–5.0 ГБ из доступных 20 ГБ.
* Время прогона: ~4.5–5.0 часов (лимит интерактивной сессии Kaggle — 9 часов).
* Единое окружение исключает ручную передачу артефактов между сессиями.

### Сценарий Б: Раздельные сессионные блокноты (в 3 сессиях)
Если предпочтителен запуск отдельными шагами:

| Блокнот | Файл | Слоты | Входные данные | Время |
|---|---|---|---|---|
| **Сессия 1: Данные** | [`notebooks/kaggle_fd24_session1_data.ipynb`](../notebooks/kaggle_fd24_session1_data.ipynb) | 1–4 | `freshdoc-dataset` | ~1.5–2.0 ч |
| **Сессия 2: Бэкбоны** | [`notebooks/kaggle_fd24_session2_backbones.ipynb`](../notebooks/kaggle_fd24_session2_backbones.ipynb) | 5–7 | `freshdoc-dataset` + Output Сессии 1 | ~2.0–2.5 ч |
| **Сессия 3: Финал** | [`notebooks/kaggle_fd24_session3_final.ipynb`](../notebooks/kaggle_fd24_session3_final.ipynb) | 8–9 | `freshdoc-dataset` + Output Сессии 2 | ~45 мин |

> **Сквозная преемственность истории `runs.csv` при раздельном запуске:**
> В Сессии 2 и Сессии 3 встроен автоматический поиск и копирование `runs.csv` из подключенных inputs (`/kaggle/input/**/runs.csv`) в локальный `reports/runs.csv`. Это обеспечивает автоматическое агрегирование истории экспериментов в рабочем пространстве `reports/runs.csv`.

---

## 1. Конфигурация Kaggle окружения

1. **Датасет `freshdoc-dataset`** (read-only):
   - Файл: `dataset_v1.parquet` (ровно 27 203 строки; `train: 21997 / val: 559 / test_real: 2895 / test_synth: 1552 / gold: 200`).
2. **Секреты блокнота (Add-ons -> Secrets):**
   - Ключ: `GITHUB_TOKEN` (токен с read-правами к GitHub-репозиторию).
3. **Настройки сессии:**
   - **Accelerator:** GPU T4 x2.
   - **Internet:** Always ON.
4. **Контроль версии transformers в Клетке 1:**
   - В Клетке 1 всех блокнотов встроен безопасный однострочник:
     ```bash
     !python -c "import transformers, sys; from packaging import version; sys.exit(0 if version.parse(transformers.__version__) >= version.parse('4.48.0') else 1)" || pip install --quiet --no-deps "transformers>=4.48.0"
     ```
   - Если установлена версия $\ge 4.48.0$, pip не вызывается. При более старой версии устанавливается только пакет transformers без зависимостей (`--no-deps`), что предотвращает конфликты с PyTorch и CUDA.

---

## 2. Спецификация режимов `--train_mix` (Эксперимент E2)

* `real_only`: только реальные источники (`codocbench` + `history_neg`, 2 393 строки) — базовый вклад реальных данных.
* `synth_only`: строго `synth_ast` (19 008 строк, **без `synth_llm`**) — измерение доменного сдвига синтетических мутаций.
* `real_ast`: `codocbench` + `history_neg` + `synth_ast` (21 401 строка) — базовая смесь реальных данных и AST-синтетики.
* `full`: полная смесь датасета v1 (21 997 строк, включая LLM P1/P2) — оценка вклада LLM-пар на группе `sig_same`.

---

## 3. Сквозной операционный флоу

### Шаг 1: Предполётная проверка бэкбонов (workflow.md §6 FD-24)
1. **Smoke-тест весов на GPU:** Granite (48M), UniXcoder (125M), ModernBERT (149M), GTE-ModernBERT (149M).
2. **Overfit-test на ModernBERT:** 200 строк × 20 эпох (`--overfit_test`). Критерий: PR-AUC $\ge 0.99$.
3. **Пилот 100 шагов FP16 на ModernBERT через DDP:**
   - Запуск через `torchrun --nproc_per_node=2 scripts/train_m2.py ... --pilot_steps 100`.
   - Проверка NaN в loss. Unpadding в текущей реализации отключен (используется стандартный batch padding до max_len батча).
   - **Стоп-критерий workflow:** при NaN лосса снизить `learning_rate` с `2e-5` до `1e-5` (`--lr 1e-5`).
   - Время эпохи рассчитывается динамически по размеру `train_df` (для `full` 687 шагов, для `real_only` 74 шага при batch=32).
4. **Пилот 20 шагов FP16 на GTE-ModernBERT @2048 через DDP (`--gradient_checkpointing`):**
   - Ранняя валидация аллокации памяти и стабильности контекста 2048 до запуска основных слотов.

### Слоты 1–4: Эксперимент E2 (UniXcoder @1024, doc_budget 384)
Каждый слот выполняет:
1. Обучение: `torchrun --nproc_per_node=2 scripts/train_m2.py --config config/train.yaml --train_mix <РЕЖИМ> ...`
2. Очистку диска: `rm -rf <output_dir>/checkpoint-*`
3. Инференс: `python scripts/predict_m2.py --checkpoint <output_dir>/best --dataset "$DATA_PATH" --out <output_dir>/preds.csv`
4. Оценку: `python scripts/evaluate.py --dataset "$DATA_PATH" --pred M0=m0.csv --pred M1=m1.csv --pred M2=<output_dir>/preds.csv --out_metrics <output_dir>/metrics.json --out_table <output_dir>/table_e1.csv --figdir <output_dir>/figures_e1`

### Слоты 5–7: Факториал бэкбонов и контекста
* **Слот 5:** ModernBERT-base @1024 (`--model_name answerdotai/ModernBERT-base --max_len 1024 --train_mix full`).
* **Слот 6:** GTE-ModernBERT-base @2048 (`--model_name Alibaba-NLP/gte-modernbert-base --max_len 2048 --gradient_checkpointing --train_mix full`). Инференс: `--batch_size 16`.
* **Слот 7:** ModernBERT-base @2048 (`--model_name answerdotai/ModernBERT-base --max_len 2048 --gradient_checkpointing --train_mix full`). Инференс: `--batch_size 16`.
  * *Правило пропуска:* если слот 6 не превосходит слот 5 на $\ge 0.005$ PR-AUC, слот 7 можно пропустить для экономии ~50 мин GPU.

### Слоты 8–9: Финализация
* **Слот 8 (Стэкинг M1+M2):** single-split стэкинг на `val_real`. Вход: готовый `/kaggle/working/m1.csv` + готовые предсказания победителя `--m2_preds` (или `--m2_checkpoint`).
  * Порог классификации: читается через `resolve_threshold` (из `config/risk.yaml` или 0.5).
  * Метрика `val_pr_auc` в `runs.csv` для стэкинга является in-sample диагностикой; стэкинг автоматически исключается из отбора бэкбонов в `finalize_best_config.py`.
* **Слот 9 (Абляция E3 без `diff_code`):** обучение и инференс с флагами `--drop_diff --batch_size 16` на архитектуре победителя Сессии 2.
* **Финализация:** `python scripts/finalize_best_config.py --runs_csv reports/runs.csv --out_config config/train_best.yaml`.
  * Выбор строго по max PR-AUC на реальных строках `val`.
  * Абляции (`drop_diff=True`) и стэкинг исключаются автоматически.
  * При равенстве метрик побеждает более поздний запуск по timestamp.

---

## 4. Сводная таблица бюджета слотов FD-24

| Слот | Код | Модель | Длина | Смесь train | Время GPU | Назначение |
|---|---|---|---|---|---|---|
| **1** | E2(a) | UniXcoder-base | 1024 | `real_only` | ~20–25 мин | Базовый вклад реальных данных |
| **2** | E2(b) | UniXcoder-base | 1024 | `synth_only` | ~20–25 мин | Оценка доменного сдвига синтетики |
| **3** | E2(c) | UniXcoder-base | 1024 | `real_ast` | ~20–25 мин | Базовая смесь реальные + AST |
| **4** | E2(d) | UniXcoder-base | 1024 | `full` | ~20–25 мин | Оценка вклада LLM-пар на sig_same |
| **5** | E2c | ModernBERT-base | 1024 | `full` | ~25–35 мин | Контроль претрейна (код vs общий) |
| **6** | E2c | GTE-ModernBERT | 2048 | `full` | ~45–55 мин | Снятие усечения + relevance претрейн |
| **7** | E2c | ModernBERT-base | 2048 | `full` | ~45–55 мин | Контроль контекста (пропуск при паритете 5-6) |
| **8** | E2b | Стэкинг M1+M2 | — | val preds | ~1–5 сек (CPU) | Проверка ансамблирования без утечки |
| **9** | E3 | Лучший бэкбон | 1024/2048 | `full` (no diff) | ~25–35 мин | Абляция признака diff_code |

---

## 5. Что забрать из Kaggle Output в репозиторий

1. Полный `reports/runs.csv` со всеми 9 прогонами.
2. Сформированный `config/train_best.yaml` победителя.
3. Папка весов лучшей модели: `checkpoints/<ПОБЕДИТЕЛЬ>/best/` (`model.safetensors`, `config.json`, `dualhead_config.json`, токенизатор).
4. Таблицы `table_e1.csv`, графики `figures_e1/` и `metrics.json` для каждого слота (содержат полный срез метрик по `test_real` и `gold`).
