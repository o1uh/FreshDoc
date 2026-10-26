# Model Card: M2 (FreshDoc)

> Статус: freeze-1, финальный сид 42. Все цифры ниже — только из отчётных файлов, источник каждой цифры в скобках.

## 1. Назначение

M2 — основная модель FreshDoc. На вход — пара (изменение кода `diff_code` + старый docstring `doc_old`), на выходе:

- `p_stale` — вероятность, что docstring устарел после правки кода (бинарная голова);
- `drift_type` — тип дрейфа (вторая голова): `param_added` / `param_removed` / `param_renamed` / `default_changed` / `return_changed` / `exception_changed` / `behavior_changed` / `unknown`.

Архитектура: `microsoft/unixcoder-base`, контекст 1024, бюджет docstring 384 ([train_best.yaml](../config/train_best.yaml)); обучение — [train_m2.py](../scripts/train_m2.py), инференс — [predict_m2.py](../scripts/predict_m2.py).

## 2. Данные обучения

Train: **n = 21997** = codocbench 1265 (pos) + history_neg 1128 (neg) + synth_ast 19008 (9558 neg / 9450 pos) + synth_llm 596 (365 neg / 231 pos); всего строк 27203, сплиты train 21997 / val 559 / test_real 2895 / test_synth 1552 / gold 200 ([dataset_v1.manifest.json](../reports/dataset_v1.manifest.json), сводка — [fd31_freeze.md](../reports/fd31_freeze.md) §2).

Паспорт данных: [data_card.md](./data_card.md) (состав, веса, шумы); статистика сборки: [data_stats.md](../reports/data_stats.md); журнал сбора: [data_notes.md](./data_notes.md); состав датасета — §4.6.3 [записки](../записка.md).

## 3. Метрики (seed 42, сырой max-F1 порог с val)

Порог подобран на **val** (max F1): τ_raw = 0.9850 (точно 0.9849684834480286). Источник всех цифр раздела — [fd31_seed42_metrics.json](../reports/fd31_seed42_metrics.json), M2, `all`.

| Сплит | n / pos | PR-AUC [95% CI] | F1 (P / R) | drift_acc / drift_macro-F1 |
|---|---|---|---|---|
| test_real | 2895 / 1444 | 0.9783 [0.9737, 0.9827] | 0.9326 (0.9563 / 0.9100) | 0.8055 / 0.6723 |
| gold | 200 / 55 | 0.5259 [0.4230, 0.6623] | 0.5926 (0.4486 / 0.8727) | 0.7609 / 0.6483 |

Контекст: это лучший из 3 финальных сидов (val PR-AUC 0.9706 / 0.9669 / 0.9655, mean 0.9677 ± 0.0026 — [fd31_freeze.md](../reports/fd31_freeze.md) §6.1; побуквенные цифры сидов 43/44 — [fd31_seed43_metrics.json](../reports/fd31_seed43_metrics.json), [fd31_seed44_metrics.json](../reports/fd31_seed44_metrics.json)). Старый прогон FD-21 ([metrics.json](../reports/metrics.json)) не отчётный.

## 4. Калибровка и рабочий порог (калиброванная модель slot4_full)

Температурное масштабирование на реальных строках val: **T = 2.3847** (NLL val 0.4339 → 0.2998). Рабочий порог по критерию precision ≥ 0.80 на калиброванном val: **τ = 0.8743** (точно 0.8742609896291749, P = R = F1 = 0.9242 на val), записан в [risk.yaml](../config/risk.yaml) (`threshold`). Источник — [calibration.json](../reports/calibration.json), [e4_e5_e6.md](../reports/e4_e5_e6.md).

| Сплит | PR-AUC [95% CI] | F1 (P / R) | Brier | ECE |
|---|---|---|---|---|
| val (559 / 211) | 0.9724 [0.9569, 0.9850] | 0.9242 (0.9242 / 0.9242) | 0.0905 | 0.1262 |
| test_real (2895 / 1444) | 0.9837 [0.9796, 0.9873] | 0.9477 (0.9682 / 0.9280) | 0.0744 | 0.0949 |
| gold (200 / 55) | 0.4776 [0.3878, 0.6111] | 0.6173 (0.4673 / 0.9091) | 0.3290 | 0.3668 |

Оговорка: T минимизирует NLL, а не ECE — ECE на val/test_real после калибровки слегка вырос (0.106 → 0.126 / 0.089 → 0.095), Brier и NLL улучшились на всех реальных срезах ([e4_e5_e6.md](../reports/e4_e5_e6.md) §E5).

## 5. Ограничения

- **Только Python.** Обучена на Python-парах (R1-фильтры: приватные имена, docstring < 40 символов отсеяны); на Java/Go и других языках не валидирована ([external_sources.md](./external_sources.md)).
- **Контекст 1024 / doc 384.** Усечено: train 2.3%, val 5.5%, test_real 2.0% ([m2_truncation.json](../reports/m2_truncation.json)); вклад усечения в ошибки мал (4/137 истинных, [error_analysis.md](../reports/error_analysis.md)).
- **Шум weak-меток.** P1-синтетика шумна на 56–60% ([data_notes.md](./data_notes.md) FD-11); на `test_real` 45.3% ошибок — шум меток, а не модели ([error_analysis.md](../reports/error_analysis.md)). Главный режим отказа — переоценка сигнатурного диффа без проверки docstring (67.2% истинных ошибок — FP). Честная оценка — только `gold`.
- **Когда лучше M0.** M0 (сигнатурное правило, без GPU/весов) детерминирован и бесплатен, на `sig_changed` точен по построению; но в целом слабее: test_real PR-AUC 0.7629 / F1 0.6902, gold PR-AUC 0.4416 / F1 0.6053 ([fd31_seed42_metrics.json](../reports/fd31_seed42_metrics.json), M0). M0 — бейзлайн и фолбэк без GPU, не замена M2.
