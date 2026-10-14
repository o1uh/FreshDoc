## Что сделано
- Шаг 0: восстановлен `data/external/codocbench` (хэш `ee0256da` совпал с FD-03); таблица пулов FD-04 воспроизведена локально (91 train / 90 test, `manim → test`, `Python → train`).
- Шаг 1: `config/mining_projects.yaml` — 60 проектов по убыванию числа записей (покрытие 3848/4573 = 84%), train=33 / test=27. Дубли разрешены: `manim → ManimCommunity/manim`, `Python → TheAlgorithms/Python` (проверка SHA через GitHub API). Генератор: `scripts/build_mining_projects.py`.
- Шаг 2: клонировано 59/60 (`git clone --depth 1000`, 4 потока; `scripts/clone_repos.py`). `frappe/frappe` — таймаут 1200 с, записан в `reports/clone_failures.txt`, исключён.
- Шаг 3: `src/mining/miner.py` — только git CLI (R3): `symbol` = qualname (`Класс.метод`), `name` = короткое имя; сравниваются только функции из обеих версий файла; `doc_changed=true` → отброс; `n_src_lines`/`src_old` как в FD-04. `cosmetic_equal` строго по R2.7 (только связываемые имена; фикс после синтетической проверки `scripts/verify_cosmetic.py` — 15/15).
- Шаг 4: пилот 150 коммитов × 59 репо (`scripts/mine_negatives.py --mode pilot`): cosmetic 24 / body 6221 / sig 2344, errors 0. Отбор по формуле R3 (ranking по candidates/sec): chosen 18, оценка полного 239/240 мин.
- Шаги 5–6: полный прогон 1000 коммитов × 18 chosen (`--mode full`): `negatives_raw` = 26827 строк (body 20330 / sig 6358 / cosmetic 139), errors 0, truncated 0 везде.
- Тесты: `tests/test_mining.py` (10 кейсов, включая end-to-end на временном git-репо); полный сюит — 67 passed.

## Артефакты
- `reports/mining_pilot.json` (sec/commit, rows/commit, chosen 18, оценка 239 мин)
- `reports/clone_failures.txt` (frappe — таймаут)
- `reports/FD-05_summary.md` (этот файл)
- `docs/data_notes.md` (раздел «Майнинг git-истории», шаги 0–2, smoke, фикс cosmetic, пилот, full)
- Локально (не в git): `data/interim/mining_pilot/` (18 МБ), `data/interim/mining/`, `data/interim/negatives_raw.parquet` (26827 строк)

## Критерии приёмки FD-05
- [x] пилот проведён, оценка времени записана в `docs/data_notes.md` (239/240 мин)
- [x] ≥8000 строк cosmetic + body_changed без изменения docstring — 20469 (2.6×), иначе не понадобились доп. проекты
- [x] ошибки по проектам посчитаны, а не скрыты — `commit_errors`=0 везде, `truncated=False`, механизм `try/except` + `error_sample`

## Инварианты
- `doc_changed=True` — 0 строк (правило R3: такие пары покрыты CoDocBench)
- все 16 колонок R3 п.5 на месте, пустых ключевых полей — 0, полных дубликатов — 0
- пулы из `project_pools.parquet` (FD-04), пересечений train/test — 0; пул-покрытие урожая: все строки train 21271 / test 5556, срез FD-06 (cosmetic+body) train 16397 / test 4072
- решение для FD-06: cosmetic в реальной истории ≈ 0.7% (139 строк) → strong-негативы берутся из `synth_ast` (`rename_local`, R6)

Refs: FD-05
