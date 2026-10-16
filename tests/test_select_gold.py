"""Tests for FD-16 gold selection (R9 quotas, caps, assignment)."""

from __future__ import annotations

import random

import pandas as pd
import pytest

from src.data.gold import (
    assign_annotators,
    largest_remainder,
    round_robin_pick,
)


def _cand(projects: dict[str, int]) -> pd.DataFrame:
    rows = []
    for project, n in projects.items():
        for i in range(n):
            rows.append({"project": project, "pair_id": f"{project}-{i}"})
    return pd.DataFrame(rows)


def test_largest_remainder_exact():
    quotas = largest_remainder({"a": 723, "b": 335, "c": 35}, 70)
    assert quotas == {"a": 46, "b": 22, "c": 2}


def test_largest_remainder_sums_to_total():
    quotas = largest_remainder({"x": 5, "y": 3, "z": 2}, 7)
    assert sum(quotas.values()) == 7
    assert quotas["x"] >= quotas["z"]


def test_round_robin_spreads_projects():
    df = _cand({"p1": 10, "p2": 10, "p3": 10})
    out = round_robin_pick(df, 9, random.Random(42))
    assert out.groupby("project").size().to_dict() == {"p1": 3, "p2": 3,
                                                       "p3": 3}


def test_round_robin_respects_cap():
    df = _cand({"p1": 10, "p2": 10})
    cap: dict[str, int] = {"p1": 3}
    out = round_robin_pick(df, 3, random.Random(42), cap)
    assert (out["project"] == "p2").all()
    assert cap == {"p1": 3, "p2": 3}


def test_round_robin_raises_when_exhausted():
    df = _cand({"p1": 2})
    with pytest.raises(RuntimeError):
        round_robin_pick(df, 5, random.Random(42), cap_n=10)


def test_round_robin_deterministic():
    df = _cand({"p1": 10, "p2": 10})
    first = round_robin_pick(df, 6, random.Random(7))["pair_id"].tolist()
    second = round_robin_pick(df, 6, random.Random(7))["pair_id"].tolist()
    assert first == second


def test_assign_annotators_r9_scheme():
    per = assign_annotators(200, 4, 60)
    assert all(len(v) == 65 for v in per.values())
    first60 = set(range(60))
    for rows in per.values():
        assert len(set(rows) & first60) == 30
        assert len(set(rows) - first60) == 35
    counts = {}
    for rows in per.values():
        for k in rows:
            counts[k] = counts.get(k, 0) + 1
    assert all(counts[k] == 2 for k in range(60))
    assert all(counts[k] == 1 for k in range(60, 200))


def test_assign_annotators_pairs_differ():
    per = assign_annotators(60, 4, 60)
    owners = {}
    for a, rows in per.items():
        for k in rows:
            owners.setdefault(k, []).append(a)
    for k in range(60):
        assert sorted(owners[k]) == sorted([k % 4, (k + 1) % 4])
        assert owners[k][0] != owners[k][1]
