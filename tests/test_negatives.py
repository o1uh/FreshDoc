"""Tests for FD-06 negative selection (R5, R1 filters)."""

from __future__ import annotations

import pandas as pd

from src.data.negatives import (
    STRONG_FRAC,
    R1_COLS,
    apply_r1_filters,
    assign_bin,
    build_r1_rows,
    ks_stat,
    make_neg_pair_id,
    select_balanced,
)


def _doc(n: int = 60) -> str:
    return "Документация функции. " + "x" * n


def _neg_row(
    project="p1",
    name="fetch",
    doc=None,
    n_lines=20,
    category="body_changed",
    symbol="fetch",
    sha="abc123",
):
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
        "drift_type": (
            "none" if category != "signature_changed" else "param_added"
        ),
        "doc_old": doc,
        "code_old": code,
        "code_new": code + "\n    # tweak",
        "sig_old": f"def {name}(a):",
        "diff_code": "@@\n-a\n+b",
        "src_old": code,
        "n_src_lines": n_lines,
        "doc_changed": False,
    }


def _r1cand(rows, pool="train_pool"):
    cand, _, _, _ = build_r1_rows(pd.DataFrame(rows), {"p1": "train_pool"})
    cand["split_pool"] = pool
    return cand


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
    assert stats["dropped"] == 4
    assert len(kept) == 1


def test_bin_out():
    df = pd.DataFrame([_neg_row(name="big", n_lines=500)])
    kept, _ = apply_r1_filters(df)
    assert len(kept) == 0


def test_doc_changed_dropped():
    row = _neg_row(symbol="d1")
    row["doc_changed"] = True
    cand, un, stats, _ = build_r1_rows(pd.DataFrame([row]), {"p1": "t"})
    assert stats["doc_changed_dropped"] == 1
    assert len(cand) == 0 and len(un) == 0


def test_roles_weights_labels():
    df = pd.DataFrame(
        [
            _neg_row(category="cosmetic", symbol="f1"),
            _neg_row(category="body_changed", symbol="f2"),
            _neg_row(category="signature_changed", symbol="f3"),
        ]
    )
    cand, un, stats, _ = build_r1_rows(df, {"p1": "train_pool"})
    assert set(cand["label_source"]) == {"real_strong_neg", "real_weak_neg"}
    assert (cand["label_stale"] == 0).all()
    assert cand.loc[cand["label_source"] == "real_strong_neg",
                    "weight"].eq(1.0).all()
    assert cand.loc[cand["label_source"] == "real_weak_neg",
                    "weight"].eq(0.5).all()
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


def test_no_leak_aux_in_output():
    df = pd.DataFrame(
        [_neg_row(category="cosmetic", symbol="f1"),
         _neg_row(category="signature_changed", symbol="f2")]
    )
    cand, un, _, _ = build_r1_rows(df, {"p1": "train_pool"})
    for out in (cand, un):
        assert not [c for c in out.columns if c.startswith(("_leak", "_aux"))]
        assert set(out.columns) == set(R1_COLS)
    model_inputs = {"sig_old", "diff_code", "doc_old"}
    assert not model_inputs & {c for c in cand.columns
                               if c.startswith(("_leak", "_aux"))}


def test_unknown_category_raises():
    row = _neg_row(symbol="weird")
    row["category"] = "renamed_function"
    try:
        build_r1_rows(pd.DataFrame([row]), {"p1": "train_pool"})
    except ValueError as e:
        assert "category" in str(e)
    else:
        raise AssertionError("Ожидалась ошибка категории")


def test_pair_id_includes_code_new_and_doc():
    a = make_neg_pair_id("p", "s", "c", "old", "new1", "doc")
    b = make_neg_pair_id("p", "s", "c", "old", "new2", "doc")
    c = make_neg_pair_id("p", "s", "c", "old", "new1", "doc other")
    d = make_neg_pair_id("p", "s", "c", "old", "new1", "doc")
    assert a != b  # разный дифф — разный id
    assert a != c  # разный doc — разный id
    assert a == d  # полные дубликаты схлопываются


def test_vendoring_variants_kept_not_dropped():
    base = _neg_row(symbol="f", sha="abc123")
    r1 = dict(base)
    r2 = dict(base)
    r2["code_new"] = base["code_old"] + "\n    x = 1"
    r2["diff_code"] = "@@\n-a\n+b2"
    cand, un, stats, _ = build_r1_rows(
        pd.DataFrame([r1, r2]), {"p1": "train_pool"})
    assert len(cand) == 2  # раньше второй терялся (один pair_id)
    assert cand["pair_id"].is_unique
    assert stats["pair_id_dedup_dropped"] == 0


def test_cand_un_no_cross_collision():
    # Вендоринг-кейс из ревью: один symbol/commit/code_old,
    # в одной копии только тело, в другой — сигнатура.
    # С расширенным pair_id это разные id, ошибки нет.
    body = _neg_row(category="body_changed", symbol="Cls.m",
                    sha="deadbeef")
    sig = _neg_row(category="signature_changed", symbol="Cls.m",
                   sha="deadbeef")
    sig["code_new"] = body["code_old"].replace(
        "def fetch(a):", "def fetch(a, extra=1):")
    sig["diff_code"] = "@@\n-def fetch(a):\n+def fetch(a, extra=1):"
    cand, un, stats, _ = build_r1_rows(
        pd.DataFrame([body, sig]), {"p1": "train_pool"})
    assert len(cand) == 1 and len(un) == 1
    assert not (set(cand["pair_id"]) & set(un["pair_id"]))
    assert stats["cross_pair_id_collisions"] == 0
    assert stats["pair_id_dedup_dropped_unique"] == 0


def test_split_pool_preserved():
    df = pd.DataFrame([_neg_row(project="p1", symbol="f1")])
    cand, _, _, _ = build_r1_rows(df, {"p1": "test_pool"})
    assert cand.iloc[0]["split_pool"] == "test_pool"


def test_select_exact_target_when_abundant():
    pos = pd.DataFrame(
        [{"split_pool": "train_pool", "n_src_lines": n}
         for n in (5, 6, 10, 20, 50)]
    )
    rows = []
    for n in (5, 6, 10, 20, 50):
        for i in range(4):
            rows.append(_neg_row(category="cosmetic", symbol=f"c{n}_{i}",
                                 n_lines=n))
            rows.append(_neg_row(category="body_changed", symbol=f"w{n}_{i}",
                                 n_lines=n))
    selected, deficit = select_balanced(pos, _r1cand(rows), seed=42)
    assert deficit["deficit"].sum() == 0
    assert len(selected) == len(pos)


def test_assort_30_70_ratio():
    pos = pd.DataFrame([{"split_pool": "train_pool", "n_src_lines": 20}] * 10)
    rows = [_neg_row(category="cosmetic", symbol=f"c{i}", n_lines=20)
            for i in range(10)]
    rows += [_neg_row(category="body_changed", symbol=f"w{i}", n_lines=20)
             for i in range(10)]
    _, deficit = select_balanced(pos, _r1cand(rows), seed=42)
    assert deficit.iloc[0]["target"] == 10
    assert deficit.iloc[0]["strong_target"] == round(STRONG_FRAC * 10)
    assert STRONG_FRAC == 0.3


def test_cell_with_zero_strong():
    pos = pd.DataFrame([{"split_pool": "test_pool", "n_src_lines": 20}] * 5)
    rows = [_neg_row(category="body_changed", symbol=f"w{i}", n_lines=20)
            for i in range(10)]
    selected, deficit = select_balanced(pos, _r1cand(rows, "test_pool"),
                                        seed=42)
    assert deficit.iloc[0]["strong_taken"] == 0
    assert len(selected) == 5  # добор weak, не баг
    assert (selected["label_source"] == "real_weak_neg").all()


def test_shortage_fills_other_type_and_deficit():
    pos = pd.DataFrame([{"split_pool": "train_pool", "n_src_lines": 5}] * 10)
    rows = [_neg_row(category="body_changed", symbol=f"w{i}", n_lines=5)
            for i in range(4)]
    selected, deficit = select_balanced(pos, _r1cand(rows), seed=42)
    assert len(selected) == 4
    assert deficit.iloc[0]["deficit"] == 6
    assert deficit.iloc[0]["taken_strong"] == 0


def test_deficit_threshold():
    pos = pd.DataFrame([{"split_pool": "train_pool", "n_src_lines": 5}] * 10)
    rows = [_neg_row(category="body_changed", symbol="w0", n_lines=5)]
    _, deficit = select_balanced(pos, _r1cand(rows), seed=42)
    rate = deficit.iloc[0]["deficit_rate"]
    assert rate == 0.9
    assert bool(rate > 0.2) is True  # флаг возврата в FD-05


def test_determinism():
    pos = pd.DataFrame([{"split_pool": "train_pool", "n_src_lines": 20}] * 6)
    rows = [_neg_row(category="body_changed", symbol=f"w{i}", n_lines=20)
            for i in range(20)]
    cand = _r1cand(rows)
    s1, _ = select_balanced(pos, cand, seed=42)
    s2, _ = select_balanced(pos, cand, seed=42)
    assert s1["pair_id"].tolist() == s2["pair_id"].tolist()


def test_ks_improvement():
    pos = pd.DataFrame([{"split_pool": "train_pool", "n_src_lines": n}
                        for n in (5, 6, 7, 20, 21, 22, 50, 51)])
    rows = [_neg_row(category="body_changed", symbol=f"s{i}", n_lines=5)
            for i in range(30)]
    rows += [_neg_row(category="body_changed", symbol=f"m{i}", n_lines=20)
             for i in range(30)]
    rows += [_neg_row(category="body_changed", symbol=f"l{i}", n_lines=200)
             for i in range(30)]
    cand = _r1cand(rows)
    before = ks_stat(pos["n_src_lines"], cand["n_src_lines"])
    selected, _ = select_balanced(pos, cand, seed=42)
    after = ks_stat(pos["n_src_lines"], selected["n_src_lines"])
    assert after <= before


def test_ks_sanity():
    a = pd.Series([10, 11, 12, 13, 14] * 20)
    assert ks_stat(a, a.copy()) == 0.0
    b = pd.Series([200, 210, 220] * 30)
    assert ks_stat(a, b) > 0.5
