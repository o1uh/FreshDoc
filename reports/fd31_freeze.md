# FD-31 · Заморозка freeze-1 — черновик отчёта (шаг 1: код и данные)

> Статус: freeze-1 = `c512dd7` (тег поставлен; 3 сида 42/43/44 выполнены, см. §6.1).
> Полное обучение 3 сидов делает другой агент; здесь только CPU-подготовка, без переобучения.
>
> Ветка: `feat/FD-31-freeze-final-runs` (от `main`). Дата: 2026-10-25.

## 1. Проверки данных — ok:true

- Команда: `.\.venv\Scripts\python.exe scripts/validate_dataset.py --in data/processed/dataset_v1.parquet`
  (скрипт FD-13 в репозитории называется `validate_dataset.py`, не `check_dataset.py`).
- Результат: **все 16 проверок PASS** (`ok:true`), строк 27203, пулов 181.
  `pair_id` уникален, сплиты `{'train': 21997, 'test_real': 2895, 'test_synth': 1552, 'val': 559, 'gold': 200}`,
  изоляция `function_hash` между train/val/test/gold — 0 пересечений, веса соответствуют R1.
- Файл данных (заморожен, в git НЕ входит — см. `.gitignore:/data/`):
  `data/processed/dataset_v1.parquet`, 49 066 226 байт, дата 18.10.2026,
  `sha256: cedc8e294b2cb8ae902f4416aac4d252a95b5642fabe6bea358ce3e3ce314e1c`.

## 2. Сверка манифеста — расхождений нет, пересборка не нужна

`reports/dataset_v1.manifest.json` (включая `gold_patch` FD-18: 140 `test_real→gold` + 60 `unlabeled_sigchange`)
полностью совпадает с фактическим parquet. Агрегат `split × source × label_stale`:

| split | source | label_stale | manifest (Σ по drift_type) | parquet (факт) | Δ |
|---|---|---|---|---|---|
| gold | gold | 0 | 145 | 145 | 0 |
| gold | gold | 1 | 55 | 55 | 0 |
| test_real | codocbench | 1 | 1444 | 1444 | 0 |
| test_real | history_neg | 0 | 1451 | 1451 | 0 |
| test_synth | synth_ast | 0 | 806 | 806 | 0 |
| test_synth | synth_ast | 1 | 746 | 746 | 0 |
| train | codocbench | 1 | 1265 | 1265 | 0 |
| train | history_neg | 0 | 1128 | 1128 | 0 |
| train | synth_ast | 0 | 9558 | 9558 | 0 |
| train | synth_ast | 1 | 9450 | 9450 | 0 |
| train | synth_llm | 0 | 365 | 365 | 0 |
| train | synth_llm | 1 | 231 | 231 | 0 |
| val | codocbench | 1 | 211 | 211 | 0 |
| val | history_neg | 0 | 348 | 348 | 0 |
| **Итого** | | | **27203** | **27203** | **0** |

Отложенный патч gold (140 строк) уже учтён в манифесте — факт зафиксирован, **перегенерация не выполнялась**.
Команда пересборки (только для справки, тяжёлую сборку НЕ запускать без решения оператора):
`.\.venv\Scripts\python.exe scripts/build_dataset.py --help`
(входы `data/interim/*.parquet` + выходы `--out/--out_stats/--out_manifest`).

## 3. Best-конфиг (заморожен в `config/train_best.yaml`, FD-24)

- Модель: `microsoft/unixcoder-base`, `max_len=1024`, `doc_budget=384`, CLS-пулинг.
- Смесь обучения: `full` (real + AST + LLM P1/P2), `drop_diff=false`.
- Лосс: BCE + `λ_drift=0.6` (Masked Conditional Loss, `ignore_index=-100` для `y_stale=0`/`unknown`).
- Оптимизация: `lr=2e-5`, `weight_decay=0.01`, `warmup=6%`, `AdamW`, 3 эпохи,
  `per_device_batch_size=8 × accum=2 × 2×T4 DDP` (эфф. батч 32), `fp16`, `seed=42`.
- Инвариант выбора: **строго max PR-AUC на реальных строках val** (не по тестам).
- Эталон: `PR-AUC val_real = 0.97241`, источник — `/kaggle/working/checkpoints/slot4_full`.

## 4. Обоснование выбора slot4_full против ModernBERT-конкурента

(источник цифр: шапка `config/train_best.yaml` + `reports/tables_e1_all_slots.csv` + `run_meta.json` слотов)

- Топ-4 прогона лежат в окне **DDP-шума (Delta=0.007)**, 95% CI ва́ла перекрываются [0.950, 0.985]:
  slot4 (UniXcoder@1024) val PR-AUC **0.97244** vs slot5 (ModernBERT@1024) 0.96953
  vs slot7 (ModernBERT@2048) 0.96686 vs slot6 (GTE@2048) 0.96037.
  Разница slot4–slot5 (0.0029) — внутри шума, выбор по вторичным критериям.
- **Drift-F1 (test_real): 0.812 против 0.670** — slot4 `drift_macro_f1=0.812068`,
  ближайший конкурент (slot5 ModernBERT@1024) `0.670307`.
- **Время обучения: 27 мин против ~76 мин** — slot4 `train_time_sec=1627.5` (~27.1 мин),
  ModernBERT@2048 (slot7) `4574.5 c` (~76.2 мин), GTE@2048 (slot6) `4667.6 c` (~77.8 мин).
- Итог: победитель — **slot4_full (UniXcoder-base, seed 42)**. ModernBERT-ветка закрыта как проигравшая
  по вторичным критериям при паритете в пределах шума.

## 5. Пересчёт оценки best БЕЗ переобучения — воспроизведено

- Предикты: `checkpoints/slot4_full/preds.csv` (27203 строк; побайтово идентичны `reports/preds/m2.csv` —
  `reports/preds/m2.csv` был перезаписан предиктами slot4 при финализации FD-24,
  а `reports/metrics_m2.json` остался от старого прогона FD-21 и для slot4 нерелевантен).
- Команда: `.\.venv\Scripts\python.exe scripts/evaluate.py --dataset data/processed/dataset_v1.parquet
  --pred slot4_full=checkpoints/slot4_full/preds.csv
  --out_metrics $TEMP\fd31_eval_metrics.json --out_table $TEMP\fd31_eval_table.csv`
  (вывод — только во временные файлы вне `reports/`; побочный перезаписанный
  `reports/figures/e1/*.png` откачен через `git checkout`).
- Ключевые цифры (совпадают с `checkpoints/slot4_full/table_e1.csv`):
  - **val/all: PR-AUC 0.9724 [0.957, 0.985]**, F1=0.924 (n=559, pos=211) — метрика выбора;
  - test_real/all: PR-AUC 0.9837 [0.980, 0.987], F1=0.947;
  - test_synth/all: PR-AUC 1.000, F1=0.998;
  - gold/all: PR-AUC 0.4776 [0.388, 0.611], F1=0.600 (n=200, n_pos=55 — только внешний контроль).

## 6. План 3 финальных сидов (выполняет другой агент)

- Конфигурация — **та же, что `config/train_best.yaml`**, меняется **только `seed`: 42 / 43 / 44**.
- Метрика выбора лучшего сида: **val PR-AUC** (реальные строки val); отчёт — **mean±std** по 3 сидам.
- Инварианты (не нарушать):
  - `test_real` / `test_synth` / `gold` — только для отчётности, **не для подбора** модели/порога/сида;
  - сплиты заморожены: реальные строки — **по проектам** (val-проекты перечислены в манифесте),
    синтетика — **по `function_hash`** (единая карта ast+llm, 90/10);
  - порог и калибровка — отдельная задача (FD-25), не часть выбора сида.

## 6.1. Факт: 3 сида выполнены в Kaggle (25.10.2026, 2×T4)

- Прогоны seed 42/43/44, конфиг `train_best.yaml` без изменений (только seed); время ~28.5 мин/прогон (1710/1712/1699 с), итого ~1.5 ч. Строки дописаны в `reports/runs.csv` (eval_pr_auc = val PR-AUC из run_meta).
- PR-AUC (group=all, `evaluate.py --n_boot 1000`), источник `reports/fd31_seeds_summary.csv`:

| seed | val (95% CI) | test_real (95% CI) | gold (95% CI) |
|---|---|---|---|
| 42 | 0.9706 [0.9553, 0.9828] | 0.9783 [0.9737, 0.9827] | 0.5259 [0.4230, 0.6623] |
| 43 | 0.9669 [0.9494, 0.9813] | 0.9797 [0.9752, 0.9839] | 0.4780 [0.3896, 0.6193] |
| 44 | 0.9655 [0.9479, 0.9792] | 0.9766 [0.9710, 0.9815] | 0.4644 [0.3706, 0.6017] |
| **mean±std (n=3, sample)** | **0.9677 ± 0.0026** | **0.9782 ± 0.0016** | **0.4894 ± 0.0323** |

- F1 при пороге max-F1 на val real (тот же источник): val 0.9209 / 0.9198 / 0.9103 (mean 0.9170 ± 0.0058); test_real 0.9326 / 0.9384 / 0.9207 (mean 0.9306 ± 0.0090); gold 0.5926 / 0.6098 / 0.5746 (mean 0.5923 ± 0.0176).
- Примечание: eval_pr_auc в `runs.csv` (0.97058 / 0.96687 / 0.96547) отличается от pr_auc сводки на ~2e-5 — округление разных стадий (run_meta vs evaluate); отчётные цифры — из сводки.
- Артефакты — из `fd31_final3seeds_artifacts.tar.gz` (Kaggle): per-seed `reports/fd31_seed{42,43,44}_{metrics.json,table_e1.csv}` — копии `checkpoints/fd31_seed*/{metrics.json,table_e1.csv}` из архива (предикты/веса в репо не входят); PR-графики перестроены локально (`scripts/evaluate.py`, 5 моделей seed42/43/44+M0/M1, CPU) в `reports/figures/fd31/`.

## 7. `.gitignore` — веса/данные/предикты в коммит не попадут

`git check-ignore -v` подтверждает:
- `data/processed/dataset_v1.parquet` → `.gitignore:2:/data/`
- `checkpoints/slot4_full/best/model.safetensors` → `.gitignore:8:checkpoints/`
- `reports/preds/m2.csv` → `.gitignore:27:reports/preds/`

## 8. Осталось оператору / другому агенту

1. Ревью этого черновика (`reports/fd31_freeze.md`).
2. Тег **`freeze-1 = c512dd7`** — поставлен (финал M2 на 3 сидах, #37).
3. Полное обучение 3 сидов (42/43/44) по `config/train_best.yaml` — вне рамок данного шага.
