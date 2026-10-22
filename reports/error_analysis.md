# FD-26: разбор ошибок M2 (E7)

## Метод

- Модель: `microsoft/unixcoder-base @1024` (`config/train_best.yaml`),
  калиброванная `p_cal` (`reports/calibration.json`, T=2.3847),
  рабочий порог `tau=0.8742609896291749` (FD-25, precision>=0.80 на `val`).
- Ошибки: `(p_cal >= tau) != label_stale`, только `gold` + `test_real`.
  Отбор детерминирован (`scripts/select_fd26_errors.py`, сортировка по `pair_id`,
  `err_id` e001…e210; усечение тем же `encode_pair`, что видел M2).
- Разметка: две независимые слепые R9-разметки всех 210 —
  прогон Spark и прогон Gemini
  (`prompts/fd26_annotate.txt`, шарды `reports/fd26_shards/` 6×35,
  батчи `reports/fd26_batches/` 6×35);
  сведение `scripts/combine_fd26.py`, споры — вручную.
- Шкала причин: шум метки / особенность кода /
  сложный docstring / усечение входа / другое.

## Счётчики ошибок

- Всего **210**: `gold` 62 (FP 57 + FN 5), `test_real` 148 (FP 44 + FN 104).
- По источникам: `test_real` FN = 104 `real_coupled`; FP = 44 `real_weak_neg`;
  `gold` — все `gold_human`.
- По сигнатурам: `test_real` FN `none/unknown` 92/104; `test_real` FP `none/none` 44/44;
  `gold` FP: `none/unknown` 22, `renamed` 10, `default` 9, `removed` 7, прочие;
  `gold` FN: 5 (`none` 3, `exception` 1, `removed` 1).
- Усечение (флаги M2): A 2/210, doc 3/210 (строки e106, e110, e113, e146, e160).

## Причины

| Причина | Доля | Пример | Что можно сделать |
|---|---|---|---|
| шум метки | — | — | — |
| особенность кода | — | — | — |
| сложный docstring | — | — | — |
| усечение входа | — | — | — |
| другое | — | — | — |

## Примеры для записки

Первые кандидаты по уверенности M2 (итоговый выбор — после LLM и ручного разбора):

- `gold` FP: e141 `manim/LinearTransformationScene.add_title` (p=0.9867, `none/unknown`),
  e169 `matplotlib/ColorbarBase.set_label` (p=0.9865, `none/unknown`);
- `test_real` FN: e148 `manim/SceneFileWriter.clean_cache` (p=0.0144, `none/unknown`),
  e197 `matplotlib/ToolLineHandles.closest` (p=0.0233, `none/unknown`).
