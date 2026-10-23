# FD-39: гайд разметки полного Go-набора (381 кейс, Gemini + Spark)

> Готово заранее: `data/processed/go_full381.csv` (381 строка, слепой:
> `label`/`comment` пустые; 30 калибровочных строк входят 1:1),
> `reports/go_batches_full/` (32 шарда по 10-12 кейсов, последний — 9),
> скрипт сведения `scripts/combine_go.py`.
> Арбитраж: кап ~50 вручную + правило agreement-only,
> метки споров — ровно в `reports/go_full381_human.csv`.

## 0. Проверка артефактов

```bash
.venv/Scripts/python.exe scripts/make_go_batches.py --input data/processed/go_full381.csv \
  --out_dir reports/go_batches_full --per_batch 12   # перегенерировать шарды (идемпотентно)
.venv/Scripts/python.exe scripts/combine_go.py --check --input data/processed/go_full381.csv \
  --shards_dir reports/go_batches_full   # 381/381 слепой, батчи покрывают всё
```

## 1. Прогон Gemini

Открой `prompts/go_orchestrator_gemini.txt` и выполни как оркестратор:
для каждого из 32 батчей — изолированный субагент только с текстом батча,
ответ — байт-в-байт в `reports/go_batches_full/go_batchNN.gemini.response.csv`.
Финал — `reports/go_full381_gemini.csv` (381 строка `id,label,stale_statement,why`, по `id`).

## 2. Прогон Spark

То же с `prompts/go_orchestrator_spark.txt` →
`reports/go_batches_full/go_batchNN.spark.response.csv` →
`reports/go_full381_spark.csv`. Прогоны независимы: один и тот же батч нельзя
размечать в общем контексте с другим прогоном.

## 3. Сведение и критерий приёмки

```bash
.venv/Scripts/python.exe scripts/combine_go.py --spark reports/go_full381_spark.csv \
  --gemini reports/go_full381_gemini.csv --input data/processed/go_full381.csv \
  --shards_dir reports/go_batches_full \
  --out_combined reports/go_full381_combined.csv --out_disputes reports/go_full381_disputes.csv
```

Скрипт проверяет: покрытие 381/381 без дублей, `label ∈ A/B/C/D`,
для каждого A `stale_statement` — точная подстрока `doc_old` (иначе — переразметка строк,
≤ 2 попыток, потом D с `why`). Считает долю согласия, κ Коэна (4-класс и бинарную;
строки с D из бинарной исключаются со счётчиком и WARNING), долю D каждого судьи
(порог 10% — выше только WARNING), кладёт споры в `reports/go_full381_disputes.csv`
и печатает вердикт. **Порог согласия ≥ 0.7 — критерий приёмки, иначе exit code 1.**

## 4. Арбитраж (кап ~50 вручную + правило agreement-only)

```bash
# Слепой файл для арбитра (без прокси/меток/скоров):
.venv/Scripts/python.exe scripts/combine_go.py --export-blind \
  --out_disputes reports/go_full381_disputes.csv \
  --out_blind reports/go_full381_disputes_blind.csv
```

Разбери `reports/go_full381_disputes_blind.csv` вручную (шкала R9 из `prompts/go_annotate.txt`,
жёсткое правило «нет ложного утверждения — не A»): вручную — не более ~50 спорных строк,
остальные споры — по правилу agreement-only (в консенсус не идут, строка исключается
из финальных меток). Итоговые метки споров — в `reports/go_full381_human.csv`
(`id,label`; label ∈ A/B/C, D в консенсус не попадает).
Сборка с арбитражем (метки принимаются только для строк из споров;
`final` = human для споров, consensus для согласия):

```bash
.venv/Scripts/python.exe scripts/combine_go.py --spark reports/go_full381_spark.csv \
  --gemini reports/go_full381_gemini.csv --input data/processed/go_full381.csv \
  --human reports/go_full381_human.csv \
  --out_combined reports/go_full381_combined.csv --out_disputes reports/go_full381_disputes.csv
```

## 5. Финал (сборка external с консенсусом Go)

```bash
.venv/Scripts/python.exe scripts/build_external.py --jit_types Param,Return,Summary \
  --go_csv data/processed/go_full381.csv --go_consensus reports/go_full381_consensus.csv \
  --out data/processed/external_all.parquet   # метки Go = консенсус (go_silver_consensus)
```

M2/M3 на Go оцениваются против консенсуса судей (silver, не gold) — в записке так и писать;
M0 на Go скипаем (парсить нечем) — открыто, как в постановке.
