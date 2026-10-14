"""Tests for FD-06 negative selection (R5, R1 filters)."""

from __future__ import annotations

import pandas as pd

from src.data.negatives import (
    apply_r1_filters,
    assign_bin,
    build_r1_rows,
    ks_stat,
    select_balanced,
)


def _doc(n: int = 60) -> str:
    return "Документация функции. " + "x" * n


def _neg_row(project="p1", name="fetch", doc=None, n_lines=20, category="body_changed",
             symbol="fetch", sha="abc123", pool_note=""):
    doc = _doc() if doc is None else doc
    code = f"def {name}(a):\n    return a"
    return {
        "project": project,
        "commit_sha": sha,
        "date": "2024-01-01",
        "file": "m.py",
        "symbol": symbol,
        "name": name,
        "category": category,
        "drift_type": "none" if category != "signature_changed" else "param_added",
        "doc_old": doc,
        "code_old": code,
        "code_new": code + "\n    # tweak",
        "sig_old": f"def {name}(a):",
        "diff_code": "@@\n-a\n+b",
        "src_old": code,
        "n_src_lines": n_lines,
        "doc_changed": False,
    }


def test_bins_edges():
    assert assign_bin(4) == "4-8"
    assert assign_bin(8) == "4-8"
    assert assign_bin(9) == "9-15"
    assert assign_bin(40) == "16-40"
    assert assign_bin(300) == "41-300"
    assert assign_bin(3) == "OUT"
    assert assign_bin(301) == "OUT"


def test_r1_filters_union_not_sum():
    df = pd.DataFrame(
        [
            _neg_row(name="_priv"),  # underscore only
            _neg_row(name="ok", doc="short"),  # doc only
            _neg_row(name="ok2", n_lines=500),  # len only
            _neg_row(name="_both", doc="short", n_lines=500),  # all three
            _neg_row(name="good"),
        ]
    )
    kept, stats = apply_r1_filters(df)
    assert stats["underscore"] == 2
    assert stats["doc_short"] == 2
    assert stats["len_out"] == 2
    assert stats["отброшено_union"] == 4
    assert len(kept) == 1


def test_roles_weights_labels():
    df = pd.DataFrame(
        [
            _neg_row(category="cosmetic", symbol="f1"),
            _neg_row(category="body_changed", symbol="f2"),
            _neg_row(category="signature_changed", symbol="f3"),
        ]
    )
    cand, un, stats = build_r1_rows(df, {"p1": "train_pool"})
    assert set(cand["label_source"]) == {"real_strong_neg", "real_weak_neg"}
    assert (cand["label_stale"] == 0).all()
    assert cand.loc[cand["label_source"] == "real_strong_neg", "weight"].eq(1.0).all()
    assert cand.loc[cand["label_source"] == "real_weak_neg", "weight"].eq(0.5).all()
    assert len(un) == 1
    assert un.iloc[0]["label_stale"] == -1
    assert un.iloc[0]["weight"] == 0.0
    assert un.iloc[0]["split_pool"] == "train_pool"
    assert cand["pair_id"].is_unique


def test_unknown_project_raises():
    df = pd.DataFrame([_neg_row(project="ghost")])
    try:
        build_r1_rows(df, {"p1": "train_pool"})
    except ValueError as e:
        assert "без пула" in str(e)
    else:
        raise AssertionError("Ожидалась ошибка проекта")


def test_select_exact_target_when_abundant():
    pos = pd.DataFrame(
        [{"split_pool": "train_pool", "n_src_lines": n} for n in (5, 6, 10, 20, 50)]
    )
    rows = []
    for n in (5, 6, 10, 20, 50):
        for i in range(4):
            rows.append(_neg_row(category="cosmetic", symbol=f"c{n}_{i}", n_lines=n))
            rows.append(_neg_row(category="body_changed", symbol=f"w{n}_{i}", n_lines=n))
    cand = pd.DataFrame(rows)
    cand, _, _ = build_r1_rows(cand, {"p1": "train_pool"})
    cand["split_pool"] = "train_pool"
    selected, deficit = select_balanced(pos, cand, seed=42)
    assert deficit["deficit"].sum() == 0
    assert len(selected) == len(pos)


def test_shortage_fills_other_type_and_deficit():
    pos = pd.DataFrame([{"split_pool": "train_pool", "n_src_lines": 5}] * 10)
    rows = [
        _neg_row(category="body_changed", symbol=f"w{i}", n_lines=5)
        for i in range(4)
    ]
    cand = pd.DataFrame(rows)
    cand, _, _ = build_r1_rows(cand, {"p1": "train_pool"})
    cand["split_pool"] = "train_pool"
    selected, deficit = select_balanced(pos, cand, seed=42)
    assert len(selected) == 4
    assert deficit.iloc[0]["deficit"] == 6
    assert deficit.iloc[0]["taken_strong"] == 0


def test_determinism():
    pos = pd.DataFrame([{"split_pool": "train_pool", "n_src_lines": 20}] * 6)
    rows = [_neg_row(category="body_changed", symbol=f"w{i}", n_lines=20) for i in range(20)]
    cand, _, _ = build_r1_rows(pd.DataFrame(rows), {"p1": "train_pool"})
    cand["split_pool"] = "train_pool"
    s1, _ = select_balanced(pos, cand, seed=42)
    s2, _ = select_balanced(pos, cand, seed=42)
    assert s1["pair_id"].tolist() == s2["pair_id"].tolist()


def test_ks_sanity():
    a = pd.Series([10, 11, 12, 13, 14] * 20)
    assert ks_stat(a, a.copy()) == 0.0
    b = pd.Series([200, 210, 220] * 30)
    assert ks_stat(a, b) > 0.5
