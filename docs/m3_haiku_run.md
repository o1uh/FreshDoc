# FD-22: прогон судьи M3 на Claude Haiku 5.5 через API (ветка `feat/FD-22-m3-api-haiku`)

Судья: `claude-haiku-5-5` (Claude Haiku 5.5), Go Inference API opencode
(`POST https://opencode.ai/inference/go/anthropic/v1/messages` — оплата из
месячной квоты Go: $15 на Go / $60 на Go Plus; факт прогона ~$0.11), параметры:
`max_tokens=1000`, thinking выкл. Генератор синтетики — `deepseek-flash`,
с судьёй не совпадает (R7 соблюдено).

Отклонения от R10 (зафиксированы здесь, не замолчаны):
- `temperature=0` недостижим: шлюз отвечает 400 "`temperature` is deprecated
  for this model" на любое значение — поле не отправляется, действует дефолт шлюза;
- `max_tokens=120` физически режет валидный JSON (smoke 20.10.2026: обрезка на
  120 и на 500) — поднято до 1000; биллинг по факту (~350 ток. среднее).

Почему Haiku: Gemini 3.8 Flash рассматривался и отклонён — размечал все 200
строк gold (петля самооценки), Muse Spark — по той же причине; Haiku в проекте
не использовался. По AA Intelligence Index Haiku 5.5 (max) = 43 против
Gemini-high 41; по API пишутся реальные токены usage, есть дисковый кэш и
повторы. Оговорки: effort `max` из AA — сущность агента, в HTTP-контракте его
нет; `temperature` шлюз запрещает (дефолт шлюза). Для задачи «один JSON +
одно предложение» thinking не нужен.

## 1. Ключ

```powershell
$env:OPENCODE_API_KEY="..."  # или строка OPENCODE_API_KEY=... в .env (файл не коммитится)
```

Ключ нужен только для `run`. Без ключа работают `estimate`, `export`,
`import-api` (на готовых ответах). Ключ на диск не пишется.

## 2. Стадии запуска

```powershell
# 0. Смета (без сети и ключа): 6 запросов ~$0.001, 500 запросов ~$0.11
.venv\Scripts\python scripts/m3_judge.py estimate

# 1. Выборка (идемпотентна, без сети):
# seed 22: 300 test_real + весь gold 200; список — reports/tables/m3_select.json
.venv\Scripts\python scripts/m3_judge.py export

# 2. Smoke: первые 6 кейсов (проверка ключа, формата, токенов)
.venv\Scripts\python scripts/m3_judge.py run --dir data/m3_judge --limit 6 --workers 2
.venv\Scripts\python scripts/m3_judge.py import-api --dir data/m3_judge
# → reports/preds/m3.csv (6 строк) + отчёт; проверить средний completion в usage.csv
# и пересчитать смету при сильном расхождении с 400/350

# 3. Полный прогон: 500 одиночных запросов (точный R10, один кейс на запрос)
.venv\Scripts\python scripts/m3_judge.py run --dir data/m3_judge --workers 4
.venv\Scripts\python scripts/m3_judge.py import-api --dir data/m3_judge
# → reports/preds/m3.csv + data/m3_judge/m3_raw.jsonl + reports/m3_judge.md
```

Кэш `data/m3_judge/cache` по хэшу (модель, параметры, сообщения): повторный `run`
ключ не тратит; `--force` — перезапросить; `--limit N` — первые N строк манифеста.

## 3. Оценка

```powershell
.venv\Scripts\python scripts/evaluate.py --pred M0=reports/preds/m0.csv --pred M1=reports/preds/m1.csv --pred M3=reports/preds/m3.csv
```

M3 покрывает только test_real+gold; пустые срезы val/test_synth дают nan
(протокол R11, без усреднений). Порог F1 — по реальным строкам val.

**Не забыть на FD-23 (E1 v0): в сводную таблицу внести ОБЕ строки —
M3 (предикты `reports/preds/m3.csv` ← `data/m3_judge`) и M3max (предикты
регенерируются командой ниже из `data/m3_judge_max`, в git не хранятся).
В таблице E1 v0 у каждой строки указывать N (M0/M1 — полный test_real n=2895,
M3/M3max — сабсет n=300; gold n=200 везде) и бутстрэп-CI — прямое сравнение
PR-AUC без учёта N и CI некорректно:**
```powershell
.venv\Scripts\python scripts/m3_judge.py import-api --dir data/m3_judge_max --out <preds_max.csv> --report <report_max.md>
.venv\Scripts\python scripts/evaluate.py --pred M0=reports/preds/m0.csv --pred M1=reports/preds/m1.csv --pred M3=reports/preds/m3.csv --pred M3max=<preds_max.csv>
```

## 5. Прогон с effort=max (чувствительность, вне FD-22)

```powershell
# те же 500 строк (data/m3_judge_max/m3_manifest.csv — копия манифеста)
.venv\Scripts\python scripts/m3_judge.py run --dir data/m3_judge_max --workers 4 --effort max --max-tokens 32000
# добивка упёршихся: --max-tokens 64000 --timeout 300 (только недостающие)
.venv\Scripts\python scripts/m3_judge.py import-api --dir data/m3_judge_max --out <вне reports/preds/> --report <вне репозитория>
```

Итог 20.10.2026: принято 500/500, обрезок 0, факт $0.952 (thinking ~1.6M токенов).
PR-AUC в пределах CI базового прогона, acc@0.5 gold 0.900 vs 0.850. Вывод: 9× цена
без значимого прироста — основной остаётся обычный прогон. Флаги `--effort`,
`--max-tokens`, `--timeout` — для таких экспериментов.

## 4. Файлы

- `src/synth/opencode_client.py` — клиент (кэш, 3 повтора, смета, цены $0.10/$0.50).
- `scripts/m3_judge.py` — `estimate`/`run`/`import-api`.
- `config/m3_judge.yaml` — модель, параметры, прайс.
- `tests/test_m3_judge_api.py` — 13 тестов без сети (моки `requests.post`).
