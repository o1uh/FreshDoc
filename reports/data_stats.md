# dataset_v1: статистика сборки (FD-12 + gold-патч FD-18)

- Строк: 27203 (входы: codocbench 2997, negatives 2997, ast_v2 29515, llm 822; дедуп pair_id −7, синтетика val-проектов −8510, виденные функции из test_synth −671; gold-патч FD-18: +200 строк split=gold)
- Сплиты: {'train': 21997, 'test_real': 2895, 'test_synth': 1552, 'val': 559, 'gold': 200}
- Вал-проектов (17): ArchiveBox, OpenBBTerminal, airflow, ansible, black, conan, django-celery-results, kivy, llama_index, localstack, mmdetection, models, python-binance, rich, sd-webui-controlnet, spaCy, speechbrain
- Val — только реальные строки; val-проекты отсутствуют в train/test_synth; test_synth — только функции без реальных строк; пересечения по function_hash train/val, train/test_real, train/test_synth отсутствуют (assert в сборке)
- sig_type пересчитан, расхождений со stored: 0
- Веса: real_coupled 1.0, strong 1.0, weak 0.5, ast 1.0, llm_p1 0.5*, llm_p2 0.5 (*решение команды); gold 0.0

## split × source × label × drift_type

| split | source | label | drift_type | n |
|---|---|---|---|---|
| gold | gold | 0 | default_changed | 10 |
| gold | gold | 0 | exception_changed | 10 |
| gold | gold | 0 | none | 69 |
| gold | gold | 0 | param_added | 4 |
| gold | gold | 0 | param_removed | 7 |
| gold | gold | 0 | param_renamed | 11 |
| gold | gold | 0 | return_changed | 9 |
| gold | gold | 0 | unknown | 25 |
| gold | gold | 1 | default_changed | 2 |
| gold | gold | 1 | exception_changed | 3 |
| gold | gold | 1 | none | 1 |
| gold | gold | 1 | param_added | 21 |
| gold | gold | 1 | param_removed | 10 |
| gold | gold | 1 | param_renamed | 5 |
| gold | gold | 1 | return_changed | 5 |
| gold | gold | 1 | unknown | 8 |
| test_real | codocbench | 1 | default_changed | 33 |
| test_real | codocbench | 1 | exception_changed | 53 |
| test_real | codocbench | 1 | param_added | 320 |
| test_real | codocbench | 1 | param_removed | 138 |
| test_real | codocbench | 1 | param_renamed | 131 |
| test_real | codocbench | 1 | return_changed | 86 |
| test_real | codocbench | 1 | unknown | 683 |
| test_real | history_neg | 0 | none | 1451 |
| test_synth | synth_ast | 0 | none | 806 |
| test_synth | synth_ast | 1 | default_changed | 32 |
| test_synth | synth_ast | 1 | exception_changed | 283 |
| test_synth | synth_ast | 1 | param_added | 172 |
| test_synth | synth_ast | 1 | param_removed | 47 |
| test_synth | synth_ast | 1 | param_renamed | 164 |
| test_synth | synth_ast | 1 | return_changed | 48 |
| train | codocbench | 1 | default_changed | 39 |
| train | codocbench | 1 | exception_changed | 96 |
| train | codocbench | 1 | param_added | 229 |
| train | codocbench | 1 | param_removed | 103 |
| train | codocbench | 1 | param_renamed | 94 |
| train | codocbench | 1 | return_changed | 65 |
| train | codocbench | 1 | unknown | 639 |
| train | history_neg | 0 | none | 1128 |
| train | synth_ast | 0 | none | 9558 |
| train | synth_ast | 1 | default_changed | 358 |
| train | synth_ast | 1 | exception_changed | 3659 |
| train | synth_ast | 1 | param_added | 1872 |
| train | synth_ast | 1 | param_removed | 794 |
| train | synth_ast | 1 | param_renamed | 2101 |
| train | synth_ast | 1 | return_changed | 666 |
| train | synth_llm | 0 | none | 365 |
| train | synth_llm | 1 | behavior_changed | 231 |
| val | codocbench | 1 | default_changed | 7 |
| val | codocbench | 1 | exception_changed | 9 |
| val | codocbench | 1 | param_added | 60 |
| val | codocbench | 1 | param_removed | 15 |
| val | codocbench | 1 | param_renamed | 19 |
| val | codocbench | 1 | return_changed | 13 |
| val | codocbench | 1 | unknown | 88 |
| val | history_neg | 0 | none | 348 |

Манифест: `dataset_v1.manifest.json` (хэши входов, параметры, версии).
Файл: `dataset_v1.parquet` (27203 строк).
