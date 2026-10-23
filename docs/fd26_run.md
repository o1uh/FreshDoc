# FD-26: разметка ошибок M2 двумя независимыми LLM-прогонами

Порог и калибровка: `reports/calibration.json`
(tau = 0.8742609896291749, T = 2.3847, подобраны на реальных строках `val`,
тесты не использовались).

## 1. Отбор ошибок

```bash
.venv/Scripts/python scripts/select_fd26_errors.py --n_shards 6
.venv/Scripts/python scripts/combine_fd26.py --check
.venv/Scripts/python scripts/make_fd26_batches.py
```

- Всего **210 ошибок**: `gold` 62 (FP 57 + FN 5), `test_real` 148 (FP 44 + FN 104).
- По источникам: `test_real` FN = 104 `real_coupled`, FP = 44 `real_weak_neg`;
  `gold` — все 62 `gold_human`.
- По сигнатурам: `test_real` FN `none/unknown` 92/104; `test_real` FP `none/none` 44/44;
  `gold` FP: `none/unknown` 19, `renamed` 9, `default` 9, `removed` 7, прочие;
  `gold` FN: 5 (`none` 3, `exception` 1, `removed` 1).
- Усечение тем же кодом, что видел M2 (`src/models.m2_data.encode_pair`,
  `microsoft/unixcoder-base @1024`, `doc_budget=384`): A 2/210, doc 3/210
  (строки e106, e110, e113, e146, e160).

Отбор детерминирован: сортировка по `pair_id`, `err_id` e001…e210;
шарды — стратифицированно по `(split,error_type)` + round-robin (6×35).
Ключи манифеста: `sha1(diff + \x00 + doc)`.

## 2. Артефакты

| Файл | Назначение |
|---|---|
| `reports/fd26_errors_full.csv` (210) | метки + `m2_p/m2_pred/error_type` + `truncated/doc_truncated`; для сведения споров, не для разметки |
| `reports/fd26_for_annotation.csv` (210) | слепая таблица: `err_id,project,symbol,sig_old,diff_code,doc_old,sig_type,n_src_lines` (без split/source/меток) |
| `reports/fd26_shards/fd26_shard_00..05.csv` (6×35) | та же слепая, по одному шарду на агента |
| `reports/fd26_batches/fd26_batch00..05.prompt.txt` (6) | self-contained батчи: задание R9 + 35 кейсов; субагенту передаётся только текст батча |
| `reports/tables/fd26_manifest.json` | tau/T, счётчики, `err_id→pair_id`, ключи sha1 |
| `prompts/fd26_annotate.txt` | задание шард-агенту (R9, формат ответа) |
| `prompts/fd26_orchestrator_spark.txt` | задание оркестратору прогона Spark |
| `prompts/fd26_orchestrator_gemini.txt` | задание оркестратору прогона Gemini |
| `scripts/select_fd26_errors.py` | отбор + флаги усечения + шарды + манифест |
| `scripts/make_fd26_batches.py` | сборка self-contained батчей из шардов |
| `scripts/combine_fd26.py` | `--check`; сведение; `--export-blind`; `--human` (final+cause) |
| `scripts/fd26_cause_ci.py` | бутстрэп-CI долей причин → `reports/tables/fd26_cause_ci.json` |
| `src/fd26util.py` | общие helpers: `key_of`, `norm_text`, `contains_quote` |

Слепота проверена кодом (`combine_fd26.py --check`): в anno/shards/батчах нет
`split/source/label_stale/m2_p/m2_pred/error_type/truncated/doc_truncated`;
ключи sha1 сошлись; шарды покрывают anno без дублей (6×35).
Строгий blind споров генерируется из `disputes` командой `--export-blind`;
`disputes.csv` с метаданными — внутренний файл для анализа, не для разметчиков.

## 3. Прогоны (два, независимо, каждый в отдельной сессии)

Каждый прогон размечает все 210 строк по своему файлу-заданию.
Один шард/батч — один изолированный агент; вложенных субагентов нет.
Выход агента строго: `err_id,label,stale_statement,why`
(`label` в A/B/C/D; для A — точная подстрока `doc_old`, иначе пусто).

Прогон Spark: responses `reports/fd26_batches/fd26_batchNN.spark.response.csv`,
слияние в `reports/fd26_spark.csv` (210 строк, порядок по `err_id`).
Прогон Gemini: responses `reports/fd26_batches/fd26_batchNN.gemini.response.csv`,
слияние в `reports/fd26_gemini.csv`.

Сведение после обоих прогонов и арбитраж человека:

```bash
.venv/Scripts/python scripts/combine_fd26.py \
  --spark reports/fd26_spark.csv --gemini reports/fd26_gemini.csv \
  --human reports/fd26_disputes_labeled.csv
.venv/Scripts/python scripts/fd26_cause_ci.py
```

Скрипт проверяет покрытие 210/210, валидность меток, цитаты A как подстроки
`doc_old` (с нормализацией CRLF/пробелов); считает каппу между прогонами
(4-класс и бинарно A/B→1, C→0, явные labels); метки человека принимает только
для строк со статусом dispute, иначе — отказ; записывает
`reports/fd26_combined.csv` (консенсус, human, final, cause) и
`reports/fd26_disputes.csv`.

## 4. Ручной разбор

- Разбираются только споры (`consensus == dispute`); согласие не пересматривается.
- Причина на ошибку (шкала FD-26): `шум метки / особенность кода / сложный
  docstring / усечение входа / другое`; коды в `cause`, ручной override —
  колонкой `cause` в human-файле (доступна и `complex_docstring`).
- Усечение: первыми проверяются 5 строк с флагами (e106, e110, e113, e146, e160).
- Первые кандидаты для записки по уверенности M2: `gold` FP e141, e169;
  `test_real` FN e148, e197; итоговый выбор — после LLM и ручного разбора.

## 5. Отчёт

- `reports/error_analysis.md`: метод + таблица причин по сплитам с CI
  + 4 примера + выводы для записки (FD-33).
