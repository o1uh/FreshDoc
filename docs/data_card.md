# Карточка данных dataset_v1

Корпус `data/processed/dataset_v1.parquet`, **27 203 строки** ([../reports/dataset_v1.manifest.json](../reports/dataset_v1.manifest.json), [../reports/data_stats.md](../reports/data_stats.md), §4 табл. 4.7).

## Источники и объёмы

Входы сборки: `pairs_codocbench` 2 997, `pairs_negatives` 2 997, `synth_ast_v2` 29 515, `synth_llm_main700` 822, `project_pools` 181, `annotations_final` 200 (манифест `inputs`). Дедуп по `pair_id` −7; изоляция val: синтетика val-проектов −8 510, очистка `test_synth` от функций реальных строк −671; gold-патч FD-18: +200 строк `split=gold` (манифест `params`, `data_stats`).

Сплиты: train 21 997 / test_real 2 895 / test_synth 1 552 / val 559 / gold 200 (манифест `totals.by_split`, табл. 4.7). По источникам: `synth_ast` 20 560 / `history_neg` 2 927 / `codocbench` 2 920 / `synth_llm` 596 / `gold` 200 (манифест `totals.by_source`, табл. 4.8: `none` 13 726, `exception_changed` 4 113, `param_added` 2 678, `param_renamed` 2 525, `unknown` 1 443 и др.).

Val — только реальные строки 17 проектов (манифест `val_projects`, `data_stats`); `test_synth` — только AST-функции без реальных строк (манифест `val_isolation`).

## Лицензии

Данные ([../docs/licenses.md](../docs/licenses.md)):
- CoDocBench — явной лицензии нет (public research, цитирование через CITATION.cff). Исходники не коммитятся, обрабатываются локально; в репозитории только производные признаки.
- DeepSeek `deepseek-flash` (генератор P1/P2 основной партии) — запрета на обучение по выходам в API-доках нет; использование учебное, некоммерческое.
- Claude Haiku 5.5 — только инференс судьи M3, обучение по выходам не ведётся. Gemini и Muse Spark как судьи исключены (петля самооценки: размечали gold/пилот).

Модели-бэкбоны ([../docs/note/записка.md](../docs/note/записка.md), §2.3): разрешены permissive Apache-2.0 / MIT (`microsoft/unixcoder-base`, `ModernBERT`-семейство, `ibm-granite/*`, `codebert-base` MIT). Исключены: `Qwen2.5-Coder-3B` (Qwen Research License — запрет промышленной эксплуатации), `jina-embeddings-v3` (CC-BY-NC-4.0 — запрет коммерческого использования), `OpenCoder-1.5B` (INF Custom License).

Исключено из сборки (манифест `params.excluded`): `synth_ast` v1 (подмножество пула v2), `synth_llm_main300` (332 pair-пересечения с main700), `synth_llm` legacy (120), `synth_llm_pilot` (22), `unlabeled_sigchange` (`label_stale = −1`, вес 0.0 — только gold/replay), `p3_main700` (аугментация doc, не пары). Gold: `source=gold`, `label_source=gold_human`, вес 0.0 (манифест `gold_patch`: 140 строк test_real→gold + 60 unlabeled).

## Правила меток

| source | label_stale | вес | Примечание |
|---|---|---|---|
| `real_coupled` (codocbench) | 1 | 1.0 | связанные правки кода+doc |
| `real_weak_neg` (history) | 0 | 0.5 | обычный коммит без правки doc |
| `real_strong_neg` (cosmetic) | 0 | 1.0 | косметика; в eval всего 6 строк |
| `synth_ast` pos / neg | 1 / 0 | 1.0 | AST-мутаторы, детерминированная метка |
| `synth_llm` p1 (`llm_p1`) | 1 | 0.5 | снижен 1.0→0.5 решением 13.10 (шум 56–60%) |
| `synth_llm` p2 (`llm_p2`) | 1 (`behavior_changed`) | 0.5 | рефакторинг без смены поведения, шум 8% |
| `gold` | 0/1 human | 0.0 | арбитраж человека, в обучение не идёт |

(Веса — манифест `params.weights`, `data_stats`; аудит P1/P2 — [../reports/audit_v1.md](../reports/audit_v1.md).)

## Известные проблемы

- Шум `real_coupled` **49.5%** (854/1 725 расхождений с LLM-аудитом; итог eval-пула 20.8%). Декомпозиция: `sig_changed` согласие 72.5% (668/921), `sig_same` — 25.3% (203/804); из 1 069 расхождений 1 012 — класс C при `label_stale = 1` ([../reports/audit_v1.md](../reports/audit_v1.md), табл. 4.9). Решение: вес 1.0 оставлен, оценка на `sig_same` — только по gold.
- Шум `llm_p1` 56% (14/25) → 60% (15/25) после фильтров; порог возврата веса 1.0 (≤20%) не достигнут — вес остаётся 0.5. `llm_p2` 8% (2/25). `ast` 10.2% (158/1 552), `real_weak_neg` 3.1% (57/1 862) ([../reports/audit_v1.md](../reports/audit_v1.md)).
- M0 = 1.0 на `test_synth` (PR-AUC/F1) — артефакт стенда: метки построены по факту смены сигнатуры, а M0 читает ровно сигнатуру; выводов о качестве по `test_synth` не делать. На `sig_same` M0 слеп (Recall 0.0, PR-AUC = prevalence 0.320 на test_real) (§6.1.3, §5.2.1).
- Разметка gold: κ между двумя LLM 4-класс **0.72**, бинарная **0.76** (совпадение 88%); на 43 двойных из 60 — 0.58/0.61, выборка мала; D=0 везде ([../reports/gold_report.md](../reports/gold_report.md)). Свёртка основная A/B→1 (stale 55/200), строгая A→1 (23/200).
- Усечение контекста M2 (окно 1024, `L_doc` = 384): код **2.42%** (658), docstring 8.93% (2 429) по всему корпусу ([../reports/m2_truncation.json](../reports/m2_truncation.json), табл. 3.1).
