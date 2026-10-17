"""FD-16: отбор кандидатов золотого набора (R9).

Чистая логика; скрипт-обёртка — scripts/select_gold.py.
"""

from __future__ import annotations

import random

import pandas as pd

PROJECT_CAP = 3


def largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    """Квоты пропорционально counts методом наибольшего остатка."""
    base = sum(counts.values())
    quotas = {}
    remainders = []
    for key in sorted(counts):
        exact = total * counts[key] / base
        quotas[key] = int(exact)
        remainders.append((exact - quotas[key], key))
    short = total - sum(quotas.values())
    remainders.sort(reverse=True)
    for _, key in remainders[:short]:
        quotas[key] += 1
    return quotas


def round_robin_pick(df: pd.DataFrame, n: int, rng: random.Random,
                     cap: dict[str, int] | None = None,
                     cap_n: int = PROJECT_CAP) -> pd.DataFrame:
    """Взять n строк, циклически по проектам (детерминированно).

    cap — счётчик уже взятых строк по проектам (общий кап cap_n).
    """
    if cap is None:
        cap = {}
    pool = {}
    for project in sorted(df["project"].unique()):
        rows = df[df["project"] == project]
        idx = list(rows.index)
        rng.shuffle(idx)
        pool[project] = rows.loc[idx]
    taken = []
    while len(taken) < n:
        progressed = False
        for project in sorted(pool):
            if len(taken) >= n:
                break
            if cap.get(project, 0) >= cap_n:
                continue
            rows = pool[project]
            if rows.empty:
                continue
            taken.append(rows.index[0])
            pool[project] = rows.iloc[1:]
            cap[project] = cap.get(project, 0) + 1
            progressed = True
        if not progressed:
            raise RuntimeError(f"не хватило кандидатов: взято {len(taken)} "
                               f"из {n}")
    return df.loc[taken]


def assign_annotators(n_rows: int, n_annotators: int = 4,
                      double_n: int = 60) -> dict[int, list[int]]:
    """Раздача строк разметчикам по R9.

    Первые double_n строк — по двое разметчиков по кругу (i-й и
    (i+1)-й), остальные — по одному по кругу. Возвращает
    {разметчик: [индексы строк]}.
    """
    per: dict[int, list[int]] = {a: [] for a in range(n_annotators)}
    for k in range(n_rows):
        if k < double_n:
            per[k % n_annotators].append(k)
            per[(k + 1) % n_annotators].append(k)
        else:
            per[k % n_annotators].append(k)
    return per
