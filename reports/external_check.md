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

**M3-full скипнут (решение FD-39, зафиксировано):** полный прогон M3 по всему
external-пакету (17 559 строк: все языки, включая Go; отбор без семплирования —
`data/m3_judge_ext/m3_manifest.csv`, список `reports/tables/m3_select_ext.json`,
очищенный parquet `data/m3_judge_ext/m3_ext.parquet` без `_leak_*`/`_aux_*`;
регенерирован от финала 17 602 − 43 пустые CoCC, Go 380 консенсус без D;
stale-отбор 17 560 от прошлого пакета 17 603/прокси-Go этим перезаписан)
НЕ запускается: оценка cost/time ≈ **$3.7 / ~3 ч** (Haiku; ~800 входных + ~60
выходных токенов/запрос, workers=4). Финал FD-39 — только M2 (+ опциональная
стратпроба M3 позже). Подготовка отбора и команды ниже сохранены как есть
(артефакты отбора на месте, прогона не было):
`python scripts/m3_judge.py export-external --dataset data/processed/external_all.parquet --dir data/m3_judge_ext --select_json reports/tables/m3_select_ext.json --out_parquet data/m3_judge_ext/m3_ext.parquet`
→ `python scripts/m3_judge.py run --dataset data/m3_judge_ext/m3_ext.parquet --dir data/m3_judge_ext`
→ `python scripts/m3_judge.py import-api --dir data/m3_judge_ext --out reports/preds/m3_ext.csv --report reports/m3_judge_ext.md`.

Проверка легитимности (подготовленный, но НЕ запущенный отбор M3 — для будущей
стратпробы): в отбор попали только `diff_code`+`doc_old`
(run/import-api других колонок не читают — `_leak_doc_new` в суде не участвует,
очищенный без `_leak_*`/`_aux_*` — это `data/m3_judge_ext/m3_ext.parquet`,
а исходный `data/processed/external_all.parquet` `_leak_*` содержит (судье не отдаётся));
43 строки CoCC без комментария исключены из отбора на экспорте (судить нечего);
расхождение объёмов: `m2_ext_all.csv` покрывает 17 602 строк сборки (всего, включая 43 пустые;
до Go-консенсуса было 17 603 — строка D `go137` исключена из сборки, см. §3),
метрики M2 пересчитаны на 17 559 (`ext_python2`: 5 000 → 4 957; `dataset_rows: 17 559`,
`threshold_from: fixed` в `reports/external_metrics.json`);
пересчёт — флагом `--drop_empty`, исходный parquet не тронут (временного parquet нет);
пересечений pair_id и `function_hash` с `dataset_v1` нет (0/0);
судья Haiku в проекте ранее не использовался, генератор синтетики — другой (R7);
единственная оговорка — промпт R10 говорит «Python function», а в отборе есть
Java-методы: текст оставлен байт-в-байт ради сопоставимости с основным M3.
Метрики — тем же `scripts/evaluate.py` (`--threshold_fixed`, сплиты `ext_hebcup`/`ext_jit`/`ext_python2`/`ext_go_silver`,
без `--pred M3`: M3-full скипнут, см. выше),
смотреть только группу `all` (sig-группы для external не валидны, см. `docs/external_sources.md` §5).

**Результаты M2 (финал; M3-full скипнут — колонок M3 нет). Порог 0.9898 (E1), группа `all`.**

| Набор | N (pos/neg) | M2 PR-AUC [95% CI] | M2 F1 (P/R) | Вывод |
|---|---|---|---|---|
| `ext_hebcup` (только pos) | 8278 (8278/0) | н/д (один класс) | R@thr = 0.278 | Java-позитивы: детектит ~28% на строгом пороге |
| `ext_jit` (Param+Return+Summary test) | 3944 (1972/1972) | 0.621 [0.598, 0.644] | 0.346 (0.73/0.23) | ≈ уровень gold (0.573): перенос на Java есть |
| `ext_python2` | 4957 (2072/2885; 43 пустые CoCC исключены) | 0.579 [0.556, 0.602] | 0.371 (0.68/0.26) | ≈ уровень gold: тот же язык, другой источник держится |

Для сравнения, M2 на своём: `test_real` PR-AUC 0.984 / F1 0.952; `gold` PR-AUC 0.573 [0.456, 0.699].
Просадка Java/Python-2 vs `test_real` ожидаема (сдвиг домена); паритет с `gold` — честный сигнал:
внешние контроли того же порядка трудности, что золотая разметка.
Оговорка: строгий порог 0.9898 душит recall (0.23–0.28); таблица — на E1-пороге для сопоставимости.

**Оговорки.** Java-метрики ниже Python-овых ожидаемо: другой язык, Javadoc вместо docstring,
`sig_old` — эвристика; HebCup — только позитивы (нет PR-AUC); M0 не валиден (парсить нечем).
M0/M1 нет даже на `ext_python2`: M0 — sig-эвристика, M1 — Python-AST фичи R10,
на блоках CoCC/JIT-методах/Go неприменимы (см. `docs/external_sources.md` §5).

**Угрозы валидности (фиксируем до выводов).**
- Единица анализа ≠ R1: CoCC — блок кода, а не функция; JIT/HebCup — Java-методы с
  Javadoc; Go — godoc. Сравнение с Python-парами R1 — по порядку трудности, не побуквенно.
- Промпт M3 про Python оставлен байт-в-байт на Java/Go ради сопоставимости с основным M3 —
  это был бы гандикап M3 на внешних языках; не актуально: M3-full скипнут (см. выше).
- Temperature M3 = дефолт шлюза (не фиксирован числом; см. `config/m3_judge.yaml`) — не актуально: M3-full скипнут.
- Порог E1 0.9898 строгий (душит recall до 0.11–0.28).
- Go — финал vs консенсус двух судей (**silver, не gold**): 380 строк (A8+B19=27 pos / C353 neg;
  D `go137` исключена, см. §3); M2 PR-AUC 0.275 [0.149, 0.468], F1 0.171 (P 0.375 / R 0.111).
  CI широкий (позитивов всего 27). Старый бейзлайн vs прокси (381 строка, 27/354):
  PR-AUC 0.174 [0.092, 0.341] — в истории, не финал.

## 2. Python-2 контроль (не из нашего датасета)

**Что.** Тот же язык, другой источник: CoCC `python_raw_ccset` (5000 вручную собранных
Python-пар, RQ4 статьи 2403.00251; скачаны полностью, пропуски релиза — см.
`docs/external_sources.md` §3). Метка из коробки по правилу статьи:
коммент изменился → 1, совпал → 0 (баланс в сборке 2092/2908; в оценке 2072/2885 —
43 пустые строки без комментария исключены: судить нечего).
Оговорка: единица — блок кода, не функция; `sig_old` — эвристика; docstring-фильтров нет.

**Результаты (финал M2; M3-full скипнут — колонок M3 нет). Порог 0.9898 (E1), группа `all`.
Метки Go — консенсус двух судей (silver, не gold); M0 на Go скипнут (парсить godoc нечем).**

| Набор | N (pos/neg) | M2 PR-AUC [95% CI] | M2 F1 (P/R) | Вывод |
|---|---|---|---|---|
| `ext_python2` | 4957 (2072/2885; 43 пустые CoCC исключены) | 0.579 [0.556, 0.602] | 0.371 (0.68/0.26) | перенос на тот же язык, другой источник; ≈ gold |
| `ext_go_silver` | 380 (27/353 консенсус: A8+B19/C353; D `go137` исключена; gin 224 + cobra 157 → judgeable 381) | 0.275 [0.149, 0.468] | 0.171 (0.38/0.11) | vs консенсус (финал, silver); CI широкий, позитивов 27; старый бейзлайн vs прокси: 0.174 [0.092, 0.341] |

## 3. Go (полная разметка всего годного пула двумя судьями)

Пулы: gin 342 + cobra 393 = 735 кандидатов (по 400 коммитов; doc-cov 0.65-0.75,
стабильность ~1.0). Аннотируется весь judgeable-пул:
`data/processed/go_full381.csv` (381 строка = все 30 пилота + 351 добор;
прокси 27/381; gin 224 + cobra 157), sidecar `go_full381_sidecar.json`
(прокси — разметчикам не показывать), манифест `go_full381_manifest.json`.
354 строки с пустыми doc/diff из разметки исключены (судить нечего —
аналог 43 битых CoCC у отбора M3).

**Разметка всего набора:**
```bash
.venv/Scripts/python.exe scripts/make_go_batches.py --input data/processed/go_full381.csv \
  --out_dir reports/go_batches_full --per_batch 12   # уже нарезаны (32 шарда), проверка ниже
.venv/Scripts/python.exe scripts/combine_go.py --check --input data/processed/go_full381.csv \
  --shards_dir reports/go_batches_full
# прогоны по prompts/go_orchestrator_{gemini,spark}.txt (32 батча x2) →
# reports/go_full381_{gemini,spark}.csv, затем (согласие 361/381 = 0.948 >= 0.7 — PASS):
.venv/Scripts/python.exe scripts/combine_go.py --spark reports/go_full381_spark.csv \
  --gemini reports/go_full381_gemini.csv --input data/processed/go_full381.csv \
  --out_combined reports/go_full381_combined.csv --out_disputes reports/go_full381_disputes.csv
# слепой файл для арбитра + сборка с ручным арбитражем споров (20 споров):
# .venv/Scripts/python.exe scripts/combine_go.py --export-blind
.venv/Scripts/python.exe scripts/combine_go.py --spark reports/go_full381_spark.csv \
  --gemini reports/go_full381_gemini.csv --human reports/go_full381_human.csv \
  --input data/processed/go_full381.csv --out_combined reports/go_full381_combined.csv \
  --out_disputes reports/go_full381_disputes.csv
# итог: final A8 + B19 + C353 + D1 (позитивов 27 — совпало с прокси случайно:
# 17 строк 0→1, 16 строк 1→0, см. проверку ниже)
```
Калибровка (согласие >= 0.7 — критерий приёмки): 30 калибровочных строк
входят в 381, разметка переиспользуется 1:1. Факт: согласие 361/381 = 0.948 — PASS.
Споры: арбитраж глазами (20 споров вручную);
итоговые метки споров — в CSV `id,label` → `reports/go_full381_human.csv`
(только 20 строк споров, НЕ полный консенсус).
Полный файл консенсуса для сборки (все 381: `final` из combined) —
`reports/go_full381_consensus.csv` (A8/B19/C353/D1); собирается так:
`final` из `reports/go_full381_combined.csv` → колонки `id,label`.
В `build_external.py` передавать именно его: передача `human.csv` оставила бы 361 строку
согласия на прокси-метках (33 из них расходятся с консенсусом: 17× 0→1, 16× 1→0).
Маппинг в сборке (`scripts/build_external.py`): A/B → `label_stale=1`, C → 0
(`label_source=go_silver_consensus`), **D — не метка, а пропуск строки**
(`go_skipped_D=1`; исключена `go137`, прокси-метка 1). `pair_id` от метки не зависит
(хеш source/project/symbol/commit/code_old/code_new) — стабилен proxy→consensus,
поэтому M2-предикты Go-среза (380/380) переиспользованы без репредикта.

**Финал Go (после разметки):**
```bash
.venv/Scripts/python.exe scripts/build_external.py --jit_types Param,Return,Summary \
  --go_csv data/processed/go_full381.csv --go_consensus reports/go_full381_consensus.csv \
  --out data/processed/external_all.parquet   # Go: 380 строк (27/353), D go137 исключена
.venv/Scripts/python.exe scripts/evaluate.py --dataset data/processed/external_all.parquet \
  --pred M2=reports/preds/m2_ext_all.csv \
  --threshold_fixed 0.9898 --drop_empty --out_metrics reports/external_metrics.json \
  --out_table reports/tables/external.csv --figdir reports/figures/external
# без --pred M3: M3-full скипнут (решение FD-39, см. §1)
```
M2 на Go оценивается против консенсуса судей (silver, не gold) — в записке так и писать;
M0 на Go скипаем (парсить нечем) — открыто, как в постановке.

## 4. Как воспроизвести

```bash
# Полный external-паркет (нужны data/external/hebcup/Dataset.zip,
# data/external/jit_raw/{Param,Return,Summary}/test.json, data/external/cocc_python/*.py,
# data/processed/go_full381.csv + reports/go_full381_consensus.csv)
.venv/Scripts/python.exe scripts/build_external.py --jit_types Param,Return,Summary \
  --go_csv data/processed/go_full381.csv --go_consensus reports/go_full381_consensus.csv \
  --out data/processed/external_all.parquet
# итог сборки: 17 602 строк (HebCup 8278 + JIT 3944 + CoCC 5000 + Go 380; D go137 исключена;
# дедуп pair_id: 926 дублей HebCup). Go-метки = полный консенсус (A/B→1, C→0, D→skip).
# pair_id от метки не зависит → M2-предикты Go-среза переиспользуются без репредикта.
# JIT-репредикт + merge выполнен, метрики финал. (История: m2_ext_all.csv был собран
# со старой схемой JIT project (rsplit 2); актуальная — rsplit 3,
# см. tests/test_fd39_fixes.py::test_jit_project_keeps_dashed_owner.)
# Срез — временный (gitignore, после merge удалён); нарезка при нужде:
# .venv/Scripts/python.exe -c "import pandas as pd; df=pd.read_parquet('data/processed/external_all.parquet'); df[df.split=='ext_jit'].to_parquet('data/processed/ext_jit_slice.parquet',index=False)"
.venv/Scripts/python.exe scripts/predict_m2.py --checkpoint checkpoints/slot4_full/best \
  --dataset data/processed/ext_jit_slice.parquet --out reports/preds/m2_ext_jit_new.csv
# (merge: строки ext_jit в m2_ext_all.csv заменены на m2_ext_jit_new.csv по pair_id;
# остальные сплиты — без изменений; метрики ниже — после merge)
# M2-предсказания: полный прогон CPU ~2 ч на 17.6k строк (slot4_full/best)
# Метрики одной командой (порог M2 из E1 0.9898; 43 пустые CoCC отбрасывает --drop_empty;
# БЕЗ --pred M3: M3-full скипнут, решение FD-39)
.venv/Scripts/python.exe scripts/evaluate.py --dataset data/processed/external_all.parquet \
  --pred M2=reports/preds/m2_ext_all.csv \
  --threshold_fixed 0.9898 --drop_empty --out_metrics reports/external_metrics.json \
  --out_table reports/tables/external.csv --figdir reports/figures/external
# итог оценки: dataset_rows 17 559 (17 602 − 43 пустых CoCC)
# Отбор M3 подготовлен, но НЕ запускался (скипнут; регенерирован от финала 17 602 → 17 559):
.venv/Scripts/python.exe scripts/m3_judge.py export-external --dataset data/processed/external_all.parquet \
  --dir data/m3_judge_ext --select_json reports/tables/m3_select_ext.json \
  --out_parquet data/m3_judge_ext/m3_ext.parquet
# Go-майнинг (нужен склонированный gin + собранный gominer)
go build -o /tmp/gominer ./scripts/gominer   # в scripts/gominer: go mod init уже выполнен
.venv/Scripts/python.exe scripts/mine_go.py --repo <gin> --commits 100 \
  --gominer /tmp/gominer --out_dir data/processed/go_gin --pilot_n 30
```
