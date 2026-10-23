# FD-39: внешние источники (исследование, версии, хэши)

> Сырые данные лежат в `data/external/` (gitignore),
> собранные parquet — `data/processed/external_*.parquet` (gitignore), код — в репозитории.

## 1. Java: CUP / HebCup («How Far Are We»)

- Статья: Lin et al., «Automated Comment Update: How Far are We?», ICPC 2021.
- Репозиторий: https://github.com/Ringbo/HebCup (ветка master, HEAD `6f52063dbf4a`, 2021-03-30).
- Данные: `Dataset.zip` (56 245 245 байт, sha256 `95e39653c5aaf75b…`) — очищенная версия
  CUP-датасета (удалены 3 типа шумов из ASE20-статьи Liu et al.).
- Состав: `train_clean.jsonl` (80 591), `valid_clean.jsonl` (8 827), `test_clean.jsonl` (**9 204**).
- Поля: `sample_id`, `full_name` (проект), `commit_id`, `src_method`/`dst_method`,
  `src_javadoc`/`dst_javadoc` (= `src_desc`/`dst_desc`, в valid/test `javadoc` длиннее — полный текст,
  `desc` — первое предложение), `code_change_seq`, `desc_change_seq`, `dist`.
- Лицензия: MIT (репозиторий) + Zenodo-запись 4453891 (DOI 10.5281/zenodo.4453891, CC-BY-4.0).
- Важно: **все строки положительные** (co-change: код и javadoc правили вместе).
  Отрицательных в коробке нет → PR-AUC на чистом HebCup не определён; HebCup проверяет
  recall/F1 на Java-позитивах, двухклассовая оценка — по JIT (§2).
- Дедуп: в `test_clean.jsonl` 9 204 строк, уникальных `sample_id` 8 278; 666 id повторяются
  с побайтово идентичными сырыми текстами (артефакт сборки HebCup) → после дедупа
  по `pair_id` осталось **8 278** (см. `external_java.manifest.json`).
- Исходный CUP: https://github.com/Tbabm/CUP (HEAD `443cbc7798e6`, 2021-06-11),
  данные только через Google Drive (папка `1WLkg1xvfxAwzFR6NWbEqZrTvr7QgQqQOkP`),
  формат JSONL (`src_method`/`dst_method`, `src_desc`/`dst_desc`, токены). Отдельно не качали:
  HebCup — та же выборка после чистки, плюс прямая ссылка с GitHub.

## 2. Java: Panthaplackel JIT (двухклассовый)

- Статья: Panthaplackel et al., «Deep Just-In-Time Inconsistency Detection…», AAAI 2021.
- Репозиторий: https://github.com/panthap2/deep-jit-inconsistency-detection
  (HEAD `5879fd54daa4`, 2025-07-21). Лицензия **MIT**.
- Данные: Google Drive, папка `1heqEQGZHgO6gZzCjuQD1EyYertN4SAYZ`
  (`Return|Param|Summary/{train,valid,test}.json` + `resources/…`).
  Скачана полностью (`gdown --folder` → `data/external/jit_raw`).
  Объёмы test: Param 1 038, Return 1 840, Summary 1 066 (все балансы 50/50);
  train: 8 640 / 15 950 / 8 398; valid: 932 / 1 790 / 1 034.
- Поля: `id` (`Owner_repo-PR-Num-Type-N`), `label` («1» = коммент стал inconsistent),
  `comment_type`, `old/new_comment_raw`, `old/new_code_raw` (сырые тексты).
- В оценку идёт `test.json` всех трёх типов (`ext_jit`, 3 944 строки: 1 972/1 972).
- Размеченный clean test (300 шт., `resources/…/clean_test_ids.json`) — опция для чувствительного
  анализа, не обязателен для базовой оценки.

## 3. Python-2: CoCC `python_raw_ccset` (скачан полностью, 5000)

- Статья: «Are your comments outdated? Towards automatically detecting code-comment
  consistency» (arXiv 2403.00251). Репозиторий: https://github.com/chenyn273/CoCC
  (ветка main, обновлён 2024-06-17; лицензии в репо нет — перед релизом проверить).
- В статье (RQ4): «мы вручную собрали и разметили **5000** Python-примеров» из популярных
  open-source проектов (детали — таблица 12 статьи). Пакет — Drive-папка
  `12xYfd8DC66OdBy3HhZs2T8qrGBfVEPCo`, подпапка `python_raw_ccset`.
- Метки — из коробки по правилу статьи (раздел 2): комментарии до/после различаются →
  positive (1, outdated), совпадают → negative (0). В файлах меток нет, правило
  выводимо из содержимого: адаптер `cocc_record_to_pair` сравнивает `#`-строки блоков
  (старый vs новый). N.B.: единица CoCC — блок кода, не функция; docstring-уровня нет —
  зафиксировать как оговорку для FD-33 (другой юнит, шумит сильнее).
- Статус: скачано вручную целиком (5000 файлов `0.py…5006.py`, пропуски исходного релиза:
  725, 1451, 1754, 2057, 2708, 3223, 4492) в `data/external/cocc_python/`.
  gdown-автомат упёрся в лимиты Drive — скрипт `scripts/download_cocc.py` оставлен
  как резервный путь (по fileId из `data/external/cocc_ids.txt`, вежливо с бэкоффом).
- В сборке (`scripts/build_external.py`, сплит `ext_python2`): все 5000 распарсились,
  баланс **2 908 neg / 2 092 pos** (58/42); в оценке — 4 957 (43 пустые строки без
  комментария исключены на экспорте M3 и в пересчёте M2, см. `reports/external_check.md` §1).
- Запасной вариант (не понадобился — CoCC скачан; оставлено для истории): если бы CoCC
  не скачался, готовых Python-наборов с метками stale из коробки больше нет
  (проверены CodeSearchNet/CodeXGLUE, Edinburgh corpus — все без меток, только под
  silver-mismatch; CSN-S3 `s3.amazonaws.com/code-search-net/...` с 2026 отдаёт 403).
  Тогда Python-2 = silver (mismatch по DocChecker) с явной оговоркой, либо майнинг по схеме
  FD-05 на Python-проектах вне CoDocBench-пула.
- **Решение по постановке:** «Python-датасет из How Far Are We» — такого нет:
  HebCup («How Far Are We», Lin et al.) строго Java (80 591/8 827/9 204). Если имелся в виду
  Sun et al. «How Far Are We?» (саммаризация через ChatGPT, CSN-Python) — там нет stale-меток.
  Предлагается зафиксировать Python-2 = CoCC (единственный Python-набор с коробочными метками).

## 4. Go: gin + cobra (полная разметка всего годного пула)

- Репозитории: https://github.com/gin-gonic/gin и https://github.com/spf13/cobra
  (середняки с культурой godoc). Клоны `../gin-pilot`, `../cobra-pilot` (вне FreshDoc).
- Майнер: `scripts/gominer/main.go` (только stdlib: `go/parser`, `go/ast`, `go/printer`),
  пропускает `*_test.go` и `vendor/`; symbol = `Name` или `Recv.Name`;
  `scripts/mine_go.py` — драйвер по git-истории, pairing — `src/mining/go_pair.py`.
- Валидация (по 400 коммитов): gin — 435-529 функций/коммит, doc-cov 0.65-0.68,
  342 кандидата; cobra — 238-270 функций/коммит, doc-cov 0.65-0.75, 393 кандидата;
  стабильность имён ~1.0. Всего 735 кандидатов, judgeable-пул (непустые doc+diff) 381.
- Размечается ВЕСЬ judgeable-пул: `data/processed/go_full381.csv` (381 строка =
  все 30 пилота + 351 добор; прокси 27/381; gin 224 + cobra 157),
  sidecar `go_full381_sidecar.json` (прокси — разметчикам не показывать),
  манифест `go_full381_manifest.json`. Сборка: `scripts/make_go_silver.py`
  (`--must_include` подтягивает калибровочные 30 строк 1:1).
  Промежуточный silver-файл на 200 строк в финале не используется — только полный пул 381.
- Калибровка (согласие >= 0.7 — критерий приёмки): 30 калибровочных строк входят в 381 1:1,
  разметка переиспользуется; отдельные пилотные шарды из репозитория удалены —
  только полный пул 381.
  Полная разметка: шарды `reports/go_batches_full/` (32x10-12), сведение `combine_go.py`.
- Дифф строится по коду без godoc (0 `//`-строк в диффах): первая версия майнера
  строила diff по полному тексту с godoc — в части диффов были видны `//`-строки
  (сам факт правки доки). Теперь diff только по коду без godoc (поле `code`
  в `gominer`, протокол R1).
- Порог длины под однострочники ослаблен (не применять к Java/Python).

## 5. Решения адаптера (общие для external)

- Схема — R1 (`src/data/external_adapter.py`, скрипт `scripts/build_external.py`):
  сырой текст без нормализации R2 (Java/Go не разбираются Python-AST);
  `drift_type = "unknown"`, `sig_type = "unknown"`; без фильтров R1; вес 0.0 (не для обучения);
  сплиты-теги источника: `ext_hebcup`, `ext_jit`, (план) `ext_python2`, `ext_go_silver`.
- `sig_old` для Java/Go — эвристика «первая строка с `(`» (документировано в модуле).
- Оценка тем же `scripts/evaluate.py`: добавлены `--splits` (по умолчанию EVAL_SPLITS +
  все внешние сплиты; `train` никогда не оценивается) и `--threshold_fixed`
  (для наборов без `val`; значение по умолчанию — порог M2 из E1, 0.9898, и рабочий 0.8743 —
  фиксируется перед прогоном M3).
- Temperature M3 = дефолт шлюза (не фиксирован числом; см. `config/m3_judge.yaml`); промпт R10 говорит «Python function» — на внешних языках это гандикап M3, оставлен байт-в-байт ради сопоставимости (M3-full скипнут, см. `reports/external_check.md` §1).
- Группы `sig_changed`/`sig_same` для external не интерпретируются (`sig_type` всегда unknown
  попадает в `sig_changed` по маске `!= none`) — смотреть только группу `all`.
- M0 на Java/Go не валиден (парсить нечем) — открыто пишем «скип», в сводке M2/M3 (+M0 где валиден).
- `gold.csv` не трогаем: silver ни строкой (золото и серебро не смешиваем).
