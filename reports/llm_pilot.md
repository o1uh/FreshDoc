# FD-10: пробный прогон LLM-синтетики (P1/P2)

- Генератор: Muse Spark 1.3 (effort xhigh). Дата экспорта: 12.10.2026.
- Файлов промптов: 4; без ответов: 0; не разобрано: 0.
- Объектов всего: 24; принято P1: 12; принято P2: 10; пропущено skip: 0.
- Доля принятых: 0.917 (критерий FD-10: >= 0.40).

## По файлам

| file | task | accepted | rejected | error |
|---|---|---|---|---|
| pilot_p1_batch1.prompt.txt | p1 | 6 | 0 |  |
| pilot_p1_batch2.prompt.txt | p1 | 6 | 0 |  |
| pilot_p2_batch1.prompt.txt | p2 | 6 | 0 |  |
| pilot_p2_batch2.prompt.txt | p2 | 4 | 2 |  |

## Причины отказов

| reason | n |
|---|---|
| diff_too_big | 2 |

## Решение по few-shot

Доля принятых >= 0.40: few-shot не требуется.
