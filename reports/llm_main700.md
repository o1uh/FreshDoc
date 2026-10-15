# FD-11: основная партия LLM-синтетики (P1/P2, DeepSeek API)

- Генератор: deepseek-flash (DeepSeek-V4.1-Flash), thinking=disabled, temperature=0.7, top_p=0.95, max_tokens=1200. Дата прогона: см. fetched_at в responses/.
- Судья M3 (FD-22, Gemini 3.8 Flash) для данных не используется.
- Каталог партии: data\llm_main700; выборка: все (seed — в имени каталога/документации).
- Объектов всего: 1341; принято P1: 331; принято P2: 491; пропущено skip: 0.
- Доля принятых: 0.613 (ожидание FD-11: порядка 200+200 при минимуме 300 функций).
- Контроль арифметики: слотов 1400 = разобрано 1341 + до-разбора 59; разобрано = принято 822 + отказы валидации 519

## Причины отказов

| reason | n |
|---|---|
| thin_docstring | 154 |
| code_unchanged | 145 |
| stale_not_linked_to_diff | 93 |
| diff_too_big | 68 |
| bad_json: Unterminated string starting at: line 1 column 25 (char 24) | 43 |
| signature_changed | 26 |
| docstring_changed | 20 |
| stale_statement_not_found | 9 |
| bad_json: Unterminated string starting at: line 2 column 26 (char 27) | 7 |
| stale_covers_docstring | 2 |
| bad_json: Unterminated string starting at: line 1 column 5026 (char 5025) | 1 |
| bad_json: Invalid \escape: line 1 column 1599 (char 1598) | 1 |
| bad_syntax | 1 |
| bad_json: Unterminated string starting at: line 1 column 4714 (char 4713) | 1 |
| docstring_flag_not_true | 1 |
| bad_json: Unterminated string starting at: line 1 column 3355 (char 3354) | 1 |
| bad_json: Unterminated string starting at: line 1 column 5067 (char 5066) | 1 |
| bad_json: Unterminated string starting at: line 1 column 4378 (char 4377) | 1 |
| bad_json: Unterminated string starting at: line 1 column 5780 (char 5779) | 1 |
| bad_json: Unterminated string starting at: line 1 column 3937 (char 3936) | 1 |
| bad_json: Expecting property name enclosed in double quotes: line 1 column 4902 (char 4901) | 1 |

## Фактические токены и стоимость

- Запросов с usage: 2100; вход: 1134925; completion: 651425 (в т.ч. reasoning: 0).
- Факт: off-peak $0.561 / peak $1.122.
- Пик (Mon-Fri 01:00-04:00, 06:00-10:00 UTC; +7: Mon-Fri 08:00-11:00, 13:00-17:00 (+7 UTC)). Вне пика — вдвое дешевле.

## Аудит (вручную, FD-11 шаг 4)

- Команда: `python scripts/llm_main_batch.py audit-sample --in-path <parquet> --out reports/llm_audit_sample.csv`
- Проверить 25 P1 (стало ли неверным хотя бы одно утверждение docstring) и 25 P2 (сохранено ли поведение). Если в P2 > 20% меняют поведение — вес 0.3 или исключить P2.
- Итог — в `reports/audit_v1.md`.
