# Аудит меток LLM-синтетики (FD-11, шаг 4)

Источник: `data/interim/synth_llm_main300.parquet` (300 функций, seed 11).
Выборка: `reports/llm_audit_sample.csv` — 25 P1 + 25 P2, метки датасета скрыты на момент разметки.
Дата аудита: 13.10.2026. Разметчик: агент (по строгому A-бару R9, ≤2 мин/строка).

Вопрос P1: «стало ли неверным хотя бы одно утверждение docstring?»
Вопрос P2: «сохранено ли поведение?»

## Итог

| Источник | верно | неверно | не уверен | доля ошибок |
|---|---|---|---|---|
| P1 (`llm_p1`, вес 1.0) | 10 | 14 | 1 | **14/25 = 56%** (15/25 = 60% с «не уверен») |
| P2 (`llm_p2`, вес 0.5) | 23 | 2 | 0 | **2/25 = 8%** |

## P1: что подтверждено (10)

Ранний выход без persist (MetaGPT::write), запись meta.pt со всех рангов
(pytorch-lightning::save_checkpoint — «only saved from rank 0» ложно),
игнорируемый флаг run_aggregation (promptflow::execute_flow), усечение буфера
при clear=False (rich::export_text), инверсия else-ветки (st2), срез [:100]
при «List all» (datasets::list_metrics), запись pickle раз в 100
(Theano::add_key), реверс порядка кортежа (gym::sample), 'w'→'a'
(ultralytics::yaml_save), валидации object вместо array
(pydantic::tuple_variable_schema).

## P1: шум (14 + 1)

Систематическая причина: docstring тонкие, а валидатор проверяет лишь наличие
подстроки `stale_statements` в docstring, но не её опровержимость. По строгому
бару R9-A это класс C («всё остаётся верным»): порядок выдачи/сортировка
(iglob, load_data), частота animation (PythonRobotics), граница area_threshold
(detectron2), текст getpass-приглашения (rich::input), значение --master
(microk8s), ключ price в ответе (ccxt), пустой вход → 0.0 / {} при соблюдённом
типе Returns (gensim, spaCy), рекурсия обхода (aws-sam-cli), своп аргументов
внутр. хелпера без смены семантики (cpython::assertCountEqual — возможно,
изменение вообще нейтрально), багфикс strip при тонком doc (twine::addHtml),
недокументированный multi-app (aws-sam-cli::prepare — максимум B),
недокументированный empty-response (OpenDevin::condense). Один «не уверен»:
OpenBBTerminal::tcm — docstring обрезан в выборке, дефолт provider не виден.

## P2: нарушения (2)

- datasets::to_dict: `batch_size=0` — было значение по умолчанию, стало
  `ValueError` (шаг range 0).
- ultralytics::download: str-URL + threads>1 — было `zip` по символам строки
  (сломанный путь), стало корректно. Поведение отличается (фикс латентного бага).

Остальные 23 — чистые рефакторинги (переименования, temp-переменные, цикл →
comprehension/генератор, инверсия условий). Нюансы без смены happy path:
freqtrade::main (`_CONF` при невалидном конфиге), timm::get_pretrained_cfg_value
(значение None в реестре), conan::autoreconf (ложные не-None), jina (порядок
ключей dict).

## Решения

- **P2: 8% < 20%** — порог FD-11 не превышен. Вес 0.5 сохранить.
- **P1: 56–60% шума > 25%** (порог FD-14, шаг 4) — правило требует поправить
  генерацию или снизить вес. Рекомендация: снизить вес `llm_p1` с 1.0 до 0.5
  (уровень P2) при сборке dataset_v1 (FD-12) и зафиксировать решение в
  `docs/data_notes.md`; в записке описать как «тонкие docstring, C-класс по R9».
  Альтернатива (дороже): фильтр P1 по конкретному противоречию утверждению.
  Решение — за командой, данный отчёт его не предрешает.
- Критерий приёмки FD-11 (≥200 принятых пар + аудит): 449 пар на минимуме,
  1071 на цели; аудит проведён — **выполнен** с замечанием по весу P1 выше.

## Контрольный аудит P1 после ужесточения (FD-11 fix, 15.10.2026)

Источник: `data/interim/synth_llm_main700.parquet` после фильтров
(`thin_docstring` 154, `stale_not_linked_to_diff` 93, `stale_covers_docstring` 2;
P1 580→331, P2 491 без изменений, итого 822 строки, схема R1 17 колонок).
Выборка: `reports/llm_audit_sample_v2.csv` — 25 новых P1 (seed 12, без
пересечения-гарантии со старой выборкой проверено частично: 1 повтор
`da5f55` из v1, оставлен как якорь), разметка агентом по строгому A-бару R9.

Итог: верно 10 / неверно 15 / не уверен 0 → доля ошибок **15/25 = 60%**
(было 56% до фильтров). Порог ≤20% для возврата веса 1.0 **не достигнут**.

Вывод: лексические фильтры (длина/покрытие/связность) срезали объём
(580→331, критерий ≥200 сохранён), но прецизионность на строгом баре не
выросла: остаточный шум — краевые дополнения без опровержимого утверждения
в doc (пустой вход, `.pyc`, `mkdir parents`, `backup`, порядок без спеки).
Это подтверждает риск ревью: обучение на 60%-шумных позитивах деградирует
M2 на `sig_same`. Решение: вес `llm_p1` **остаётся 0.5** при сборке FD-12
(как в исходном решении 13.10), вес `llm_p2` 0.5 без изменений. Возврат
к 1.0 — только после семантического фильтра или ручной чистки.

Подтверждённые (10): conan skip→validate, localstack инверсия
include_response_metadata, promptflow run_aggregation, NGE -1→0, langchain
blob-фильтр против tree-примера, gensim read-only→r+, ray or→and против
«if any», luigi smtp→ses против «Default is smtp», theano g++→clang++,
saleor инверсия excluded.

Шум (15): thin/edge-дополнения без утверждения в doc (uiautomator2 missing
→'', mycroft idx-base, detectron2 deepcopy, jina пустой devices, st2
тривиальный assert, theano name-фильтр, scikit-rf z0=50, datasets .lock,
langchain вердикт D, speechbrain set→sorted, conan .pyc, ultralytics mkdir,
microk8s backup, readthedocs порядок OOM/timeout, black '' vs None).

## Раздел FD-14 (аудит 100 меток)

Не заполнен — дописать после FD-12 (выборка 25×4 из `real_coupled`,
`real_weak_neg`, `ast`, `llm`).
