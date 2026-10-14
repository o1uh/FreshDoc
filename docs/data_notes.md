# Журнал данных FreshDoc

## Решения и параметры
- Дата инициализации: 09.10.2026
- Версия Python: 3.11

### 1. Решение по инфраструктуре промежуточных данных (Kaggle)

- **Проблема:** Изоляция файловых систем `/kaggle/working` между участниками команды блокирует передачу артефактов в цепочках FD-04 -> FD-06 и FD-05 -> FD-06.
- **Решение:** 
  1. Создан промежуточный приватный датасет `freshdoc-interim` (роль `Editor` для всех участников) для синхронизации Parquet-таблиц до сборки общего датасета.
  2. Итоговый датасет сборки (FD-12) фиксируется отдельно под именем `freshdoc-dataset` (`dataset_v1.parquet`) и переводится в режим read-only для этапа обучения.

## Лог изменений данных
| Дата | Задача | Источник / Набор | Число строк | Примечания / Решения |
|---|---|---|---|---|
| 09.10.2026 | FD-03 | CoDocBench (raw jsonl) | 4573 (train: 2300, test: 2273) | Загрузка и аудит формата. Подтверждена вложенная структура version_data (0 - old, 1 - new). |
| 10.10.2026 | FD-04 | CoDocBench (pairs_codocbench) | 2997 | Успешно обработано 2997 пар (65.5%). Отсев: 1211 (фильтры R1), 365 (код не изменился), 0 (ошибки разбора). В выборке 30 пар 0% ошибок drift_type, доля unknown: 43.3%. Дефект CoDocBench (утечка проекта manim) устранен переносом проекта целиком в test_pool. Тесты пройдены. |
| 11.10.2026 | FD-07 | Пул функций v1 (functions_pool_v1) | 1474 | Вход train_pool 1476 строк; дедуп по function_hash −2; фильтры R1 −0; неразобранных 0; утечек test-проектов 0. Проектов 91. Критерий ≥1200 выполнен. |
| 12.10.2026 | FD-08 | Синтетика AST (synth_ast) | 7379 (pos 3804 / neg 3575) | Все 9 мутаций R6 (6 pos + 3 neg), seed 42, cap 30000, баланс ~1:1 (51.6/48.4 после дедупа), pos rare-first, neg с возвращением типов. По типам: none 3575, param_added 1272, exception_changed 1027, param_renamed 857, param_removed 315, return_changed 209, default_changed 124. Проходы ревью: детерминизм (crc32), скоуп-гарды, crash-None с provenance, анти-дубль return, documented-only правила. Ручная проверка 30 пар: 30/30 разумны (100%). Тесты 25/25 + полный прогон 35 passed. Команда: `python scripts/generate_synth_ast.py --seed 42 --max_pairs 30000`. |

### Итоги осмотра CoDocBench (FD-03)
- **Хэш коммита источника:** `ee0256da15a07dcf265d5313fe51a269aa7268e0`
- **Число строк:**
  - `codocbench.jsonl`: 4573
  - `train.jsonl`: 2300
  - `test.jsonl`: 2273
- **Уникальных проектов:** 201
- **Топ-10 проектов:** pandas (301), matplotlib (300), core (163), pytorch (151), cpython (145), django (133), ray (130), AzurLaneAutoScript (123), espnet (108), rasa (107)
- **Медианная длина функции:** 16.0 строк (средняя: 23.5)
- **Структура записей:** исходный код и метаданные хранятся во вложенном массиве `version_data` (`version_data[0]` — старая версия, `version_data[1]` — новая версия).

### Итоги работы адаптера CoDocBench (FD-04)
- **Исходные данные:** 4 573 записи (train: 2 300, test: 2 273).
- **Итого отобрано положительных пар:** 2 997 (до переноса manim: train 1 479 / test 1 518; после: train_pool 1 476 / test_pool 1 521)
- **Причины отсева (1 576 записей):**
  - `не_прошёл_фильтры`: 1 211 записей (train 664 + test 547) (приватные функции с `_`, docstring < 40 символов, длина вне [4, 300] строк).
  - `код_не_изменился`: 365 записей (train 157 + test 208) (дифф затрагивал только комментарии или пробелы).
  - `не_разобралось`: 0 записей (синтаксический разбор 100%).
  - `нет_обязательных_полей`: 0 записей.
- **Уникальных проектов:** 181.
- **Обнаруженный дефект CoDocBench:** проект `manim` присутствовал одновременно в `train.jsonl` и `test.jsonl`. 
  - *Решение:* все записи проекта `manim` принудительно переведены в `test_pool` для соблюдения инварианта изоляции проектов (1 479 → 1 476 / 1 518 → 1 521). Второе сырое пересечение `Python` после фильтров осталось только в train (24 строки) и не переносилось. Изоляция подтверждена: пересечение пулов равно 0.
- **Аудит 30 случайных пар:**
  - Ошибок в сопоставлении типов дрейфа: 0%.
  - Доля `unknown` (правки только тела функции без смены сигнатуры): 43.3% (13/30).
  - Представлены все 6 типов дрейфа сигнатуры: `param_added`, `param_removed`, `return_changed`, `param_renamed`, `exception_changed`, `default_changed`.

### Итоги пула функций v1 (FD-07)
- **Вход:** `data/interim/pairs_codocbench.parquet` (train_pool 1476) + `project_pools.parquet`.
- **Выход:** `data/interim/functions_pool_v1.parquet` — 1474 строк, 91 проект, колонки `project/src/function_hash`.
- **Отсев:** дубликаты hash 2, `не_разобралось` 0, `не_прошёл_фильтры` 0, утечек test-проектов 0.
- **Критерии:** ≥1200 — да (1474); изоляции — да. Команда: `python scripts/build_functions_pool.py`.

### Итоги синтетики AST (FD-08)
- **Вход:** `data/interim/functions_pool_v1.parquet` (1474 функции, 91 проект, только train_pool).
- **Выход:** `data/interim/synth_ast.parquet` — 7379 пар (pos 3804 / neg 3575).
- **Мутации:** все 9 из R6 (pos: param_renamed/added/removed, default/return/exception_changed; neg: rename_local, extract_temp, change_message). 10 стилей имён общие для pos/neg (анти-шорткат).
- **Порядок проб:** pos — rare-first без повторений (сначала `default_changed`, `return_changed`, затем остальные в случайном порядке); neg — выборка типов с возвращением (несколько пар одного типа с функции). Условия применимости НЕ ослаблялись.
- **Детерминизм:** сиды попыток через `zlib.crc32` (`_attempt_seed`); встроенный `hash()` запрещён — рандомизировался PYTHONHASHSEED (VS Code и Kaggle давали разные наборы). Доказан прогонами тестов при разных PYTHONHASHSEED; множеств в итерируемых позициях конвейера нет.
- **Ужесточения по ревью (5-й проход):** скоуп-гарды rename-мутаций (except-as, match, global/nonlocal, alias, def/class); `_ExtractTemp` только на return целевой функции; сортировка кандидатов; crash-None с provenance-меткой (BinOp, вызов/атрибут/индекс от None, await, `*`, итераторы, except-тип); анти-дубль существующего None-гарда в return. Мертвые None-ветки приняты как wart (иначе гибнут хорошие пары класса set_leverage).
- **Редкие типы:** return_changed 209 (2.8%), default_changed 124 (1.7%) — ниже 3%. Не ослаблялись сознательно (см. reports/synth_ast_stats.md). Пересчёт таблицы по типам — после FD-09 (пул v2, тот же драйвер).
- **Критерий ≥6000:** выполнен (7379). Итоговый баланс 51.6/48.4 (не ровно 1:1 из-за дедупликации одинаковых диффов).
- **Проверки:** pair_id уникален; проектов 91, утечек test-пула 0; в выходе 0 синтаксических ошибок; тесты 25/25, полный `pytest` 35 passed.
- **Ручная проверка (4-й проход):** стратифицированная выборка 30 пар — `reports/synth_ast_manual30.csv`, размечена (verdict+comment, строки сверены поименно): 30/30 разумны (100%).
- **Артефакты для E8/data_card:** exception на ~98% — добавленный guard-шаблон; pos всегда меняют сигнатуру (M0≈1.0 на test_synth — так задумано, выводы только по test_real/gold/sig_same); `Optional[` только у return_changed.

### Пилот LLM-синтетики, экспорт (FD-10)
- **Генератор:** Muse Spark 1.3 (effort xhigh). Дата экспорта: 12.10.2026. Судья M3 (Gemini 3.8 Flash, high) зарезервирован только для FD-22, для данных не используется.
- **Условия использования:** отдельных доков с запретом на генерацию обучающих данных нет; проект учебный, коммерческое использование отсутствует — риск минимален (решение команды 12.10.2026).
- **Выборка:** 12 функций, seed 42, round-robin равномерно по проектам (12 проектов: 3d-vehicle-tracking, PythonRobotics, chainer, examples, glances, kivy, langchain, mmdetection, mycroft-core, speechbrain, ulid, yewtube). Источник — только train-пул (`functions_pool_v1.parquet`).
- **Экспорт:** `data/llm_pilot/` — 4 пакета по 6 функций (`pilot_p1_batch{1,2}.prompt.txt`, `pilot_p2_batch{1,2}.prompt.txt`) + `pilot_manifest.csv`. Команда: `python scripts/llm_pilot.py --mode export`.
- **Валидаторы:** `src/synth/llm_validators.py` (сигнатура равна, docstring не изменён, дифф ≤ 8 строк, P1 — непустые подстроки stale, P2 — флаг true). Тесты: `tests/test_llm_validators.py` (11) + `tests/test_llm_batch.py` (4) = 15 passed; весь сюит — 32 passed (12.10.2026).
- **Импорт (после прогона субагентов):** `python scripts/llm_pilot.py --mode import` → `reports/llm_pilot.md` + `data/interim/synth_llm_pilot.parquet`. Критерий: принято ≥ 40%, иначе добавить `prompts/p1_fewshot_clamp.txt`.
- **Итог пилота (12.10.2026, 4 субагента Muse Spark 1.3, по одному пакету на агента):** объектов 24, принято 22 (P1 12/12 = 100%, P2 10/12 = 83.3%), доля 0.917 — критерий ≥ 0.40 выполнен, few-shot не требуется. Отказы: `diff_too_big` 2 (оба в `pilot_p2_batch2`: P0007, P0011 — переименование локальных в 4+ местах дало дифф > 8 строк, семантика сохранена). Выход: `data/interim/synth_llm_pilot.parquet` 22 строки (12 llm_p1 вес 1.0 + 10 llm_p2 вес 0.5).

### Майнинг git-истории (FD-05, шаги 0–1, 14.10.2026)
- **Шаг 0:** `data/` локально отсутствовала — склонирован CoDocBench (`git clone https://github.com/kunpai/codocbench.git data/external/codocbench`), хэш `ee0256da15a07dcf265d5313fe51a269aa7268e0` совпал с FD-03. Адаптер FD-04 перепрогонен локально для точной таблицы пулов: 2997 пар, `manim → test_pool`, `Python → train_pool` (после фильтров только train), пулов 91 train / 90 test. Команда: `python scripts/build_codocbench_pairs.py` (интерпретатор `.venv/bin/python3.11`; symlink `.venv/bin/python` бит — указывает на системный 3.14 без зависимостей, чинить отдельно).
- **Неоднозначность имён в projects.csv:** `manim` ×2 (ManimCommunity/manim, 3b1b/manim), `Python`/`python` ×2 (TheAlgorithms/Python, Yonv1943/Python). Разрешено: `Python → TheAlgorithms/Python` (пути `data_structures/...`, SHA `477b2c24` — 200 у TheAlgorithms, 422 у Yonv1943); `manim → ManimCommunity/manim` (коммит 2024-05 `1f249e45` — 200 у ManimCommunity, 422 у 3b1b; старые `manimlib/`-пути — общая история форка). Пулы: manim test, Python train.
- **Шаг 1:** `config/mining_projects.yaml` — 60 проектов по убыванию числа записей (покрытие 3848/4573 = 84.1%), train=33 / test=27. Генератор: `python scripts/build_mining_projects.py`. Маппинг `project → url` — по `repo` из projects.csv (case-insensitive) + 2 оверрайда выше.
- **Майнер:** `src/mining/miner.py` (только git CLI): `symbol` = qualname, `name` = короткое имя; сравниваются только функции из обеих версий; `doc_changed=true` → отброс; формулы `n_src_lines`/`src_old` как в FD-04; `skipped_no_doc` считается отдельно. Тесты `tests/test_mining.py` — 10 passed, весь сюит 67 passed (14.10.2026).
- **Шаг 2 (клонирование, 14.10.2026):** 59/60 репозиториев склонированы (`git clone --depth 1000`, 4 потока) в `data/repos/`, ~2.2 ГБ (позже выросло). Неудача: `frappe/frappe` — `TimeoutExpired` (1200 с) на 2-й попытке, записан в `reports/clone_failures.txt`, проект исключён из майнинга (майнер помечает `NOT CLONED` и не включает в отбор). Целостность: `git log -1` пройден для всех 59.
- **Smoke-тест (100 коммитов × 3 репо, 14.10.2026):** tornado 34 кандидата (30 body_changed, 4 signature_changed, 167 skipped_no_doc, 31.5 с → 0.315 с/коммит); TheAlgorithms/Python 64 (37/27, 0.035 с/коммит); rasa 70 (59/11, 0.067 с/коммит). `commit_errors: 0` везде. Первые 5 коммитов дали 0 кандидатов — не баг: последние коммиты tornado не трогают `.py` (только `.github/workflows`), а в остальных менялись функции без docstring. `cosmetic: 0` — чисто-косметические коммиты редки; контрольная точка для пилота (доля cosmetic должна быть > 0 в масштабе 59 проектов).
- **Фикс `cosmetic_equal` (14.10.2026):** синтетическая проверка `scripts/verify_cosmetic.py` (15 случаев по строгому R2.7) показала 4 ложных срабатывания старой реализации — она переименовывала ВСЕ имена, включая вызовы/декораторы/параметры. Исправлено: переименуются только имена, связываемые внутри функции (`Store`/`Del`-имена, `except … as`), во всех вхождениях; свободные имена (вызовы, декораторы, атрибуты, параметры) сохраняются. После фикса 15/15, весь сюит 67 passed. Примечание: переименование параметра всё равно даёт `signature_changed` (проверка сигнатуры идёт раньше косметики в `compare_pair`).

