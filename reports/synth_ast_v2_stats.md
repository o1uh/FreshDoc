# FD-09: статистика synth_ast_v2 (R6, драйвер FD-08 без изменений)

- Команда пула: `python scripts/build_functions_pool.py --mode v2`
- Вход пула: `functions_pool_v1.parquet` (1474) + `negatives_raw.parquet`
  train с `doc_changed=false` (все категории, 21271 кандидат train).
- Отсев источника 2: 5556 не-train, 0 doc_changed, 9955 фильтры R1,
  125 внутренние дубликаты hash, 1 против v1.
- Выход пула: `data/interim/functions_pool_v2.parquet` — **12664 функции**,
  91 проект, `function_hash` уникален, утечек test-пула 0.
- Команда генерации:
  `python scripts/generate_synth_ast.py --in_pool .../functions_pool_v2.parquet
  --out .../synth_ast_v2.parquet --seed 42 --max_pairs 30000`
- Выход: `data/interim/synth_ast_v2.parquet` — **29426 пар**
  (pos 15000 / neg 14426 после дедупа 574 `pair_id`).
- Сырые принятия до балансировки: pos 29538 / neg 36369; попыток 509417;
  отклонено: none(неприменимо) 88545, страховка 666, синтаксис 0,
  код тот же 0. Условия мутаций НЕ менялись (решение FD-09 п.5).
- Проектов в синтетике 89 (из 91 пула; 2 проекта не дали принятых пар),
  утечек test-пула 0. `v1` (`synth_ast.parquet`) не тронут.

## Число пар по типам

| drift_type | n | доля от всех |
|---|---|---|
| none | 14426 | 49.0% |
| param_added | 5847 | 19.9% |
| exception_changed | 4770 | 16.2% |
| param_renamed | 2300 | 7.8% |
| return_changed | 927 | 3.2% |
| param_removed | 785 | 2.7% |
| default_changed | 371 | 1.3% |

Баланс меток: stale=1 — 15000 (51.0%), stale=0 — 14426 (49.0%).
Редкие типы: return_changed 3.2% (выше 3%), default_changed 1.3% (ниже 3%,
принято сознательно без ослабления условий, как в FD-08).

## Новизна относительно v1

- Пар на новых хэшах (не из v1): 25898 (88.0%); на старых хэшах: 3528.
- Ручная проверка: стратифицированная выборка 20 пар только из новых
  источников — `reports/synth_ast_v2_manual20.csv` (по 3 на каждый тип,
  `none` — 2; колонки `verdict`/`comment` пустые, заполняет человек).

## Критерии приёмки FD-09

- [x] пул ≥ 3000 — 12664
- [x] синтетика ≥ 15000 — 29426
- [x] пересечений с test-пулом 0 (пул и синтетика)
- [ ] ручная проверка 20 — отдана на разметку (verdict/comment — человек)

Refs: FD-09
