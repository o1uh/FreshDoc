# FD-39: внешняя проверка

> Источники и решения: `docs/external_sources.md`. Воспроизведение: раздел 4.

## 1. Java-контроль: CUP/HebCup + Panthaplackel JIT

**Что.** Тот же язык-семейство, что M2 не видел (M2 обучен только на Python):
HebCup `test_clean` (8 278 позитивов после дедупа) + JIT `test` всех трёх типов
(Param 1 038 + Return 1 840 + Summary 1 066 = 3 944, баланс 50/50).
Адаптер: сырой текст, `drift_type/sig_type=unknown`, без фильтров R1, вес 0.0
(`src/data/external_adapter.py`, `scripts/build_external.py` → `data/processed/external_all.parquet`).

**Прогон.** M2 `slot4_full` (UnixCoder @1024, thr E1 0.9898 / рабочий 0.8743) —
предсказания `reports/preds/m2_ext_all.csv`.

**M3 судит ВЕСЬ external-пакет (17 560 строк: все языки, включая Go):**
отбор пересобран без семплирования (`--n_*` по умолчанию = все):
`data/m3_judge_ext/m3_manifest.csv`, список `reports/tables/m3_select_ext.json`,
очищенный parquet для прогона — `data/m3_judge_ext/m3_ext.parquet`
(только m3_id/pair_id/split/sig_old/diff_code/doc_old, без `_leak_*`/`_aux_*`).
Формула объёма: 17 603 = 17 560 + 43 пустых CoCC (строки без комментария,
судить нечего — исключены на экспорте, см. проверку ниже).
Оценка cost/time (Haiku, тир <=100K: $0.10/$0.50 за 1M; ~800 входных + ~60 выходных токенов/запрос):
~$2 вход + ~$0.5 выход; время при workers=4 примерно 2-4 ч.
Команды M3:
`python scripts/m3_judge.py export-external --dataset data/processed/external_all.parquet --dir data/m3_judge_ext --select_json reports/tables/m3_select_ext.json --out_parquet data/m3_judge_ext/m3_ext.parquet`
(создаёт `data/m3_judge_ext/m3_ext.parquet` — 17 560 строк, без `_leak_*`/`_aux_*` — и `data/m3_judge_ext/m3_manifest.csv`)
→ `python scripts/m3_judge.py run --dataset data/m3_judge_ext/m3_ext.parquet --dir data/m3_judge_ext`
→ `python scripts/m3_judge.py import-api --dir data/m3_judge_ext --out reports/preds/m3_ext.csv --report reports/m3_judge_ext.md`.

Проверка легитимности: в отбор попали только `diff_code`+`doc_old`
(run/import-api других колонок не читают — `_leak_doc_new` в суде не участвует,
экспорт пишет очищенный parquet без `_leak_*`/`_aux_*`);
43 строки CoCC без комментария исключены из отбора на экспорте (судить нечего);
расхождение объёмов: `m2_ext_all.csv` покрывает 17 603 строк (все, включая 43 пустые),
метрики M2 пересчитаны на 17 560 (`ext_python2`: 5 000 → 4 957; `dataset_rows: 17 560`,
`threshold_from: fixed` в `reports/external_metrics.json`); исходный
`data/processed/external_all.parquet` не тронут (пересчёт — флагом `--drop_empty`,
без временного parquet);
пересечений pair_id и `function_hash` с `dataset_v1` нет (0/0);
судья Haiku в проекте ранее не использовался, генератор синтетики — другой (R7);
единственная оговорка — промпт R10 говорит «Python function», а в отборе есть
Java-методы: текст оставлен байт-в-байт ради сопоставимости с основным M3.
Метрики тем же `scripts/evaluate.py` (`--threshold_fixed`, сплиты `ext_hebcup`/`ext_jit`/`ext_python2`),
смотреть только группу `all` (sig-группы для external не валидны, см. `docs/external_sources.md` §5).

**Результаты M2; колонки M3 — после прогона. Порог 0.9898 (E1), группа `all`.**

| Набор | N (pos/neg) | M2 PR-AUC [95% CI] | M2 F1 (P/R) | M3 PR-AUC [95% CI] | M3 F1 | Вывод |
|---|---|---|---|---|---|---|
| `ext_hebcup` (только pos) | 8278 (8278/0) | н/д (один класс) | R@thr = 0.278 | — | — | Java-позитивы: детектит ~28% на строгом пороге |
| `ext_jit` (Param+Return+Summary test) | 3944 (1972/1972) | 0.621 [0.598, 0.644] | 0.346 (0.73/0.23) | — | — | ≈ уровень gold (0.573): перенос на Java есть |
| `ext_python2` | 4957 (2072/2885; 43 пустые CoCC исключены) | 0.579 [0.556, 0.602] | 0.371 (0.68/0.26) | — | — | ≈ уровень gold: тот же язык, другой источник держится |

Для сравнения, M2 на своём: `test_real` PR-AUC 0.984 / F1 0.952; `gold` PR-AUC 0.573 [0.456, 0.699].
Просадка Java/Python-2 vs `test_real` ожидаема (сдвиг домена); паритет с `gold` — честный сигнал:
внешние контроли того же порядка трудности, что золотая разметка.
Оговорка: строгий порог 0.9898 душит recall (0.23–0.28); на рабочем 0.8743 recall выше —
пересчитать при финале, когда будет M3 (таблица выше — на E1-пороге для сопоставимости).

**Оговорки.** Java-метрики ниже Python-овых ожидаемо: другой язык, Javadoc вместо docstring,
`sig_old` — эвристика; HebCup — только позитивы (нет PR-AUC); M0 не валиден (парсить нечем).

**Угрозы валидности (фиксируем до выводов).**
- Единица анализа ≠ R1: CoCC — блок кода, а не функция; JIT/HebCup — Java-методы с
  Javadoc; Go — godoc. Сравнение с Python-парами R1 — по порядку трудности, не побуквенно.
- Промпт M3 про Python оставлен байт-в-байт на Java/Go ради сопоставимости с основным M3 —
  это гандикап M3 на внешних языках (занижает, а не завышает его метрики).
- Temperature M3 = дефолт шлюза (не фиксирован числом; см. `config/m3_judge.yaml`).
- Порог E1 0.9898 строгий (душит recall до 0.23–0.28); рабочий порог 0.8743 даст recall выше —
  при финале пересчитать оба.
- Go 0.174 — vs прокси (baseline, не финал); CI широкий [0.092, 0.341] (позитивов всего 27);
  финал — только vs консенсус после разметки.

## 2. Python-2 контроль (не из нашего датасета)

**Что.** Тот же язык, другой источник: CoCC `python_raw_ccset` (5000 вручную собранных
Python-пар, RQ4 статьи 2403.00251; скачаны полностью, пропуски релиза — см.
`docs/external_sources.md` §3). Метка из коробки по правилу статьи:
коммент изменился → 1, совпал → 0 (баланс в сборке 2092/2908; в оценке 2072/2885 —
43 пустые строки без комментария исключены: судить нечего).
Оговорка: единица — блок кода, не функция; `sig_old` — эвристика; docstring-фильтров нет.

**Результаты (заполнить после сборки + M2/M3).**

| Набор | N (pos/neg) | M2 PR-AUC [95% CI] | M2 F1 | M3 PR-AUC [95% CI] | M3 F1 | Вывод |
|---|---|---|---|---|---|---|
| `ext_python2` | 4957 (2072/2885; 43 пустые CoCC исключены) | 0.579 [0.556, 0.602] | 0.371 (0.68/0.26) | — | — | перенос на тот же язык, другой источник; ≈ gold |
| `ext_go_silver` | 381 (27/354 по прокси; gin 224 + cobra 157) | 0.174 [0.092, 0.341] | 0.114 (0.25/0.07) | — | — | vs прокси (baseline, не финал) (CI широкий, позитивов мало); финал — vs консенсус после разметки |

## 3. Go (полная разметка всего годного пула двумя судьями)

Пулы: gin 342 + cobra 393 = 735 кандидатов (по 400 коммитов; doc-cov 0.65-0.75,
стабильность ~1.0). Аннотируется весь judgeable-пул:
`data/processed/go_full381.csv` (381 строка = все 30 пилота + 351 добор;
прокси 27/381; gin 224 + cobra 157), sidecar `go_full381_sidecar.json`
(прокси — разметчикам не показывать), манифест `go_full381_manifest.json`.
354 строки с пустыми doc/diff из разметки исключены (судить нечего —
аналог 43 битых CoCC у M3).

**Разметка всего набора:**
```bash
.venv/Scripts/python.exe scripts/make_go_batches.py --input data/processed/go_full381.csv \
  --out_dir reports/go_batches_full --per_batch 12   # уже нарезаны (32 шарда), проверка ниже
.venv/Scripts/python.exe scripts/combine_go.py --check --input data/processed/go_full381.csv \
  --shards_dir reports/go_batches_full
# прогоны по prompts/go_orchestrator_{gemini,spark}.txt (32 батча x2) →
# reports/go_full381_{gemini,spark}.csv, затем:
.venv/Scripts/python.exe scripts/combine_go.py --spark reports/go_full381_spark.csv \
  --gemini reports/go_full381_gemini.csv --input data/processed/go_full381.csv \
  --out_combined reports/go_full381_combined.csv --out_disputes reports/go_full381_disputes.csv
# слепой файл для арбитра + сборка с ручным арбитражем споров:
# .venv/Scripts/python.exe scripts/combine_go.py --export-blind
# .venv/Scripts/python.exe scripts/combine_go.py --spark ... --gemini ... --human reports/go_full381_human.csv
```
Калибровка (согласие >= 0.7 — критерий приёмки): 30 калибровочных строк
входят в 381, разметка переиспользуется 1:1.
Споры: арбитраж глазами (кап ~50 вручную, остальное agreement-only);
итоговые метки споров — в CSV `id,label` → `reports/go_full381_human.csv`.

**Финал Go (после разметки):**
```bash
.venv/Scripts/python.exe scripts/build_external.py --jit_types Param,Return,Summary \
  --go_csv data/processed/go_full381.csv --go_consensus reports/go_full381_human.csv \
  --out data/processed/external_all.parquet   # метки Go = консенсус (go_silver_consensus)
.venv/Scripts/python.exe scripts/evaluate.py --dataset data/processed/external_all.parquet \
  --pred M2=reports/preds/m2_ext_all.csv --pred M3=reports/preds/m3_ext.csv \
  --threshold_fixed 0.9898 --drop_empty --out_metrics reports/external_metrics.json \
  --out_table reports/tables/external.csv --figdir reports/figures/external
```
M2/M3 на Go оцениваются против консенсуса судей (silver, не gold) — в записке так и писать;
M0 на Go скипаем (парсить нечем) — открыто, как в постановке.

## 4. Как воспроизвести

```bash
# Полный external-паркет (нужны data/external/hebcup/Dataset.zip,
# data/external/jit_raw/{Param,Return,Summary}/test.json, data/external/cocc_python/*.py)
.venv/Scripts/python.exe scripts/build_external.py --jit_types Param,Return,Summary \
  --out data/processed/external_all.parquet
# M2-предсказания (CPU, ~2 ч на 17.6k строк)
.venv/Scripts/python.exe scripts/predict_m2.py --checkpoint checkpoints/slot4_full/best \
  --dataset data/processed/external_all.parquet --out reports/preds/m2_ext_all.csv
# Отбор M3 (создаёт m3_ext.parquet на 17 560 строк + m3_manifest.csv)
.venv/Scripts/python.exe scripts/m3_judge.py export-external --dataset data/processed/external_all.parquet \
  --dir data/m3_judge_ext --select_json reports/tables/m3_select_ext.json \
  --out_parquet data/m3_judge_ext/m3_ext.parquet
# Метрики одной командой (порог M2 из E1; 43 пустые CoCC отбрасывает --drop_empty;
# M3-предсказания — reports/preds/m3_ext.csv после run/import-api)
.venv/Scripts/python.exe scripts/evaluate.py --dataset data/processed/external_all.parquet \
  --pred M2=reports/preds/m2_ext_all.csv --pred M3=reports/preds/m3_ext.csv \
  --threshold_fixed 0.9898 --drop_empty --out_metrics reports/external_metrics.json \
  --out_table reports/tables/external.csv --figdir reports/figures/external
# Go-майнинг (нужен склонированный gin + собранный gominer)
go build -o /tmp/gominer ./scripts/gominer   # в scripts/gominer: go mod init уже выполнен
.venv/Scripts/python.exe scripts/mine_go.py --repo <gin> --commits 100 \
  --gominer /tmp/gominer --out_dir data/processed/go_gin --pilot_n 30
```
