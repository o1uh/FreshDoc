# FD-11: основная партия LLM-синтетики через DeepSeek API

Статус на 13.10.2026: генерация **выполнена** (700 P1/P2 + 700 P3, off-peak ~11:05–12:25 +7).
Ниже — параметры факта и команды для повтора/проверки.

## 1. Генератор

- Модель: `deepseek-flash` (DeepSeek-V4.1-Flash), endpoint `https://api.deepseek.com`
  (OpenAI-совместимый). Доки сверены 13.10.2026: `pricing`, `thinking_mode`,
  `create-chat-completion`.
- Параметры факта: thinking `disabled`, `temperature=0.7`, `top_p=0.95`,
  `response_format=json_object`, `max_tokens=1200` — как R7.
- История за 13.10: thinking=enabled + reasoning_effort=max давал пустой content
  (reasoning съедал весь max_tokens: finish=length, 4096/4096 reasoning; при 8192 —
  16% fails + обрезанные JSON). Проверка сырым ответом + повтором M0003_p1/M0004_p1,
  после чего возврат к non-thinking (100/100 ok). См. `docs/data_notes.md`.
- Судья M3 (FD-22) — Claude Haiku 5.5, с генератором не совпадает (требование R7).
- Условия использования: отдельного запрета на обучение по выходам в API-доках
  нет; использование учебное, некоммерческое. См. `docs/licenses.md`.

## 2. Ключ

```powershell
$env:DEEPSEEK_API_KEY="sk-..."  # или строка DEEPSEEK_API_KEY=... в .env (файл не коммитится)
```

Без ключа работают `estimate`, `export`, `import`, `audit-sample`. Ключ нужен
только для `run`. Ключ на диск никогда не пишется (только в заголовок запроса).

## 3. Стадии запуска

```powershell
# 0. Смета (без сети)
.venv\Scripts\python scripts/llm_main_batch.py estimate

# 1. Smoke: 6 функций / 12 запросов (сухой экспорт — без ключа)
.venv\Scripts\python scripts/llm_main_batch.py export --n 6 --tag smoke --seed 7

# 2. Smoke-прогон (отдельная сессия, нужен ключ)
.venv\Scripts\python scripts/llm_main_batch.py run --dir data/llm_smoke6 --workers 2
.venv\Scripts\python scripts/llm_main_batch.py import --dir data/llm_smoke6
# → reports/llm_smoke6.md + data/interim/synth_llm_smoke6.parquet
# Проверить средний completion в выводе run и пересчитать смету (раздел 4).

# 3. Минимум FD-11: 300 функций / 600 запросов
.venv\Scripts\python scripts/llm_main_batch.py export --n 300 --tag main
.venv\Scripts\python scripts/llm_main_batch.py run --dir data/llm_main300
.venv\Scripts\python scripts/llm_main_batch.py import --dir data/llm_main300

# 4. Цель FD-11: 700 функций / 1400 запросов (если smoke и 300 в порядке)
.venv\Scripts\python scripts/llm_main_batch.py export --n 700 --tag main
.venv\Scripts\python scripts/llm_main_batch.py run --dir data/llm_main700
.venv\Scripts\python scripts/llm_main_batch.py import --dir data/llm_main700
```

Правила выборки: round-robin по проектам равномерно, seed 11 (smoke — 7),
12 пилотных функций FD-10 исключены, каждая функция — не более 2 использований
(P1+P2). Кэш `data/<партия>/cache` по хэшу (модель, параметры, сообщения):
повторный `run` ключ не тратит; `--force` — перезапросить. `--limit N` в `run` —
прогнать только первые N функций (проверка ключа на 1–2 запросах).

## 4. Цена (прайс 13.10.2026, USD за 1M)

| Токены | off-peak | peak (×2) |
|---|---|---|
| Вход, cache miss | 0.15 | 0.30 |
| Вход, cache hit | 0.003 | 0.006 |
| Выход (включая reasoning) | 0.60 | 1.20 |

Пик: пн–пт **01:00–04:00 и 06:00–10:00 UTC** (без гос. праздников КНР).
В +7 это пн–пт **08:00–11:00 и 13:00–17:00**. Всё остальное, включая выходные
и праздники КНР целиком, — off-peak (вдвое дешевле). Выгодно запускать вечером
/ ночью по +7 или в выходные.

Допущения сметы (факт 13.10 по usage.csv, 2100 запросов): вход ~560 ток./запрос,
completion ~400 (P1 421 / P2 362 / P3 147; отдельные ответы упираются в max_tokens=1200,
диапазон 100–1200).

| Партия | Запросов | off-peak | peak |
|---|---|---|---|
| Smoke | 12 | $0.004 | $0.008 |
| Минимум (300 ф.) | 600 | $0.194 | $0.389 |
| Цель (700 ф.) | 1400 | $0.453 | $0.906 |

Факт: 1400 P1/P2 — $0.447 по счётчику (~$1.2 по дашборду с утренним thinking-режимом);
700 P3 — ~$0.11. Итог дня ~$1.3 (лимит $1.50 цел).

## 5. Время

Зависит от задержки одного запроса (thinking max — медленный; факт будет после
smoke). При `--workers 4` время ≈ `запросы × задержка / 4` (лимит API 2500 —
не бутылочное горлышко):

| Партия | Запросов | 10 с/запрос | 25 с/запрос | 60 с/запрос |
|---|---|---|---|---|
| Smoke | 12 | ~30 с | ~1 мин | ~3 мин |
| Минимум (300 ф.) | 600 | ~25 мин | ~1 ч | ~2.5 ч |
| Цель (700 ф.) | 1400 | ~1 ч | ~2.5 ч | ~6 ч |

Плюс повторы (до 3 на запрос) и пауза `--sleep` между вызовами.

## 6. Аудит (FD-11 шаг 4, вручную после приёмки)

```powershell
.venv\Scripts\python scripts/llm_main_batch.py audit-sample --in-path data/interim/synth_llm_main300.parquet --out reports/llm_audit_sample.csv
```

25 P1 + 25 P2 в CSV с пустыми `verdict`/`comment`. P1: стало ли неверным хотя бы
одно утверждение docstring. P2: сохранено ли поведение. Если в P2 > 20% меняют
поведение — вес P2 0.3 или исключить. Итог — в `reports/audit_v1.md`
(критерий приёмки FD-11: ≥ 200 принятых пар + аудит).

## 7. Файлы

- `src/synth/deepseek_client.py` — клиент (кэш, 3 повтора, смета).
- `scripts/llm_main_batch.py` — стадии estimate/export/run/import/audit-sample.
- `config/llm_main.yaml` — модель, параметры, объёмы, прайс.
- `tests/test_llm_main_batch.py` — 9 тестов без сети (выборка, сообщения,
  кэш-ключ, смета, roundtrip export/import, аудит-выборка).
- Пилот FD-10 (`docs/llm_run.md`, `reports/llm_pilot.md`) — исторический,
  делался через субагентов на Muse Spark 1.3; основная партия его не касается.
