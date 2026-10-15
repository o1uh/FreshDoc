"""Tests for FD-12 dataset assembly (R1 splits, R8 dedup/sig_type)."""

from __future__ import annotations

import pandas as pd

from src.data.dataset import (
    REAL_SOURCES,
    SYNTH_SOURCES,
    WEIGHTS,
    apply_weights,
    assign_splits,
    assert_known_projects,
    bucket100,
    compute_val_projects,
    dedup_by_pair_id,
    drop_leaked_test_synth,
    drop_val_project_synthetic,
    hash_overlap,
    projects_of,
    recompute_sig_type,
)

CODE_OLD = "def f(a, b=1):\n    return a"
CODE_SAME = "def f(a, b=1):\n    return a"
CODE_RENAMED = "def f(x, b=1):\n    return x"
CODE_ADDED = "def f(a, b=1, c=2):\n    return a"

DOC = "Do thing with documented params a and b. " + "x" * 40


def _row(pair_id, source, project, fhash, code_new, drift, sig, label, label_source, weight=1.0):
    return {
        "pair_id": pair_id, "source": source, "project": project, "commit_sha": "abc",
        "symbol": "f", "function_hash": fhash, "sig_old": "def f(a, b=1):",
        "code_old": CODE_OLD, "code_new": code_new, "diff_code": "diff",
        "doc_old": DOC, "label_stale": label, "drift_type": drift, "sig_type": sig,
        "label_source": label_source, "weight": weight, "n_src_lines": 10,
    }


def _find_project(prefix, want_val):
    i = 0
    while True:
        name = f"{prefix}_{i}"
        is_val = bucket100(name) >= 80
        if is_val == want_val:
            return name
        i += 1


def test_bucket100_deterministic_and_bounded():
    assert bucket100("abc") == bucket100("abc")
    assert 0 <= bucket100("abc") < 100
    assert len({bucket100(f"p{i}") for i in range(200)}) > 50


def test_compute_val_projects_only_train_pool_and_cutoff():
    pool_of = {"a": "train_pool", "b": "train_pool", "c": "test_pool"}
    val = compute_val_projects(pool_of)
    assert "c" not in val
    assert val == {p for p in ("a", "b") if bucket100(p) >= 80}


def test_assert_known_projects_strict():
    assert_known_projects(["a"], {"a": "train_pool"})
    try:
        assert_known_projects(["typo_proj"], {"a": "train_pool"})
    except ValueError:
        pass
    else:
        raise AssertionError("ожидался ValueError для неизвестного проекта")


def test_recompute_sig_type():
    drift, ok = recompute_sig_type(CODE_OLD, CODE_RENAMED, "none")
    assert ok and drift == "param_renamed"
    drift, ok = recompute_sig_type(CODE_OLD, CODE_SAME, "param_added")
    assert ok and drift == "none"
    drift, ok = recompute_sig_type("not python ((((", CODE_SAME, "fallback")
    assert not ok and drift == "fallback"


def test_dedup_keeps_first_by_priority():
    real = pd.DataFrame([_row("dup", "codocbench", "p", "h1", CODE_ADDED, "param_added", "param_added", 1, "real_coupled")])
    synth = pd.DataFrame([_row("dup", "synth_ast", "p", "h1", CODE_ADDED, "param_added", "param_added", 1, "ast")])
    df, before, removed, dup_rows = dedup_by_pair_id([real, synth])
    assert before == 2 and removed == 1 and dup_rows == 2
    assert len(df) == 1 and df.iloc[0]["source"] == "codocbench"


def test_drop_val_project_synthetic_full():
    df = pd.DataFrame([
        _row("t1", "synth_ast", "vp", "h1", CODE_ADDED, "param_added", "param_added", 1, "ast"),
        _row("t2", "synth_llm", "vp", "h2", CODE_SAME, "none", "none", 0, "llm_p2", 0.5),
        _row("t3", "synth_ast", "tp", "h3", CODE_ADDED, "param_added", "param_added", 1, "ast"),
        _row("r1", "codocbench", "vp", "h4", CODE_ADDED, "param_added", "param_added", 1, "real_coupled"),
    ])
    kept, n = drop_val_project_synthetic(df, {"vp"})
    assert n == 2
    assert set(kept["pair_id"]) == {"t3", "r1"}


def test_assign_splits_and_isolation_mini_pipeline():
    proj_train = _find_project("proj", want_val=False)
    proj_val = _find_project("proj", want_val=True)
    pool_of = {proj_train: "train_pool", proj_val: "train_pool", "ext": "test_pool"}
    val_projects = compute_val_projects(pool_of)
    assert proj_val in val_projects and proj_train not in val_projects

    rows = [
        _row("c1", "codocbench", "ext", "hx1", CODE_ADDED, "param_added", "param_added", 1, "real_coupled"),
        _row("c2", "codocbench", proj_train, "hx2", CODE_ADDED, "param_added", "param_added", 1, "real_coupled"),
        _row("c3", "codocbench", proj_val, "hx3", CODE_ADDED, "param_added", "param_added", 1, "real_coupled"),
        _row("n1", "history_neg", proj_val, "hx4", CODE_SAME, "none", "none", 0, "real_weak_neg", 0.5),
        _row("s1", "synth_ast", proj_val, "hx5", CODE_ADDED, "param_added", "param_added", 1, "ast"),
        _row("s2", "synth_ast", proj_train, "hx6", CODE_ADDED, "param_added", "param_added", 1, "ast"),
        _row("s3", "synth_llm", proj_train, "hx7", CODE_SAME, "none", "none", 0, "llm_p2", 0.5),
    ]
    df = pd.DataFrame(rows)
    df, _, _, _ = dedup_by_pair_id([df])
    df, _ = drop_val_project_synthetic(df, val_projects)
    assert "s1" not in set(df["pair_id"])
    df["split"] = assign_splits(df, pool_of, val_projects).values

    by_id = dict(zip(df["pair_id"], df["split"]))
    assert by_id["c1"] == "test_real"
    assert by_id["c2"] == "train"
    assert by_id["c3"] == "val"
    assert by_id["n1"] == "val"
    # val — только реальные строки
    assert set(df.loc[df["split"] == "val", "source"].unique()) <= set(REAL_SOURCES)
    # val-проектов нет в test_synth
    assert not (projects_of(df, "test_synth") & val_projects)
    # нет общих хешей train/val и train/test_real
    assert not hash_overlap(df, "train", "val")
    assert not hash_overlap(df, "train", "test_real")
    # test_pool-проекты — только в test_real
    assert "ext" not in projects_of(df, "train") | projects_of(df, "val")


def _find_hash(bucket_ge_90):
    i = 0
    while True:
        h = f"leakh{i:04d}"
        if (bucket100(h) >= 90) == bucket_ge_90:
            return h
        i += 1


def test_drop_leaked_test_synth():
    proj_train = _find_project("proj", want_val=False)
    pool_of = {proj_train: "train_pool"}
    val_projects = compute_val_projects(pool_of)
    leak_hash = _find_hash(bucket_ge_90=True)
    fresh_hash = _find_hash(bucket_ge_90=False)
    rows = [
        _row("r1", "codocbench", proj_train, leak_hash, CODE_ADDED, "param_added", "param_added", 1, "real_coupled"),
        _row("s1", "synth_ast", proj_train, leak_hash, CODE_ADDED, "param_added", "param_added", 1, "ast"),
        _row("s2", "synth_ast", proj_train, fresh_hash, CODE_ADDED, "param_added", "param_added", 1, "ast"),
    ]
    df = pd.DataFrame(rows)
    df, _, _, _ = dedup_by_pair_id([df])
    df, _ = drop_val_project_synthetic(df, val_projects)
    df["split"] = assign_splits(df, pool_of, val_projects).values
    assert dict(zip(df["pair_id"], df["split"]))["s1"] == "test_synth"
    df, n = drop_leaked_test_synth(df)
    assert n == 1
    assert "s1" not in set(df["pair_id"]) and "r1" in set(df["pair_id"])
    assert not hash_overlap(df, "train", "test_synth")


def test_apply_weights_llm_p1_override():
    df = pd.DataFrame([
        _row("a", "synth_llm", "p", "h1", CODE_SAME, "behavior_changed", "none", 1, "llm_p1", 1.0),
        _row("b", "codocbench", "p", "h2", CODE_ADDED, "param_added", "param_added", 1, "real_coupled", 1.0),
    ])
    df, fixed = apply_weights(df)
    assert fixed == 1
    assert float(df.loc[df["label_source"] == "llm_p1", "weight"].iloc[0]) == WEIGHTS["llm_p1"] == 0.5
    assert set(SYNTH_SOURCES) == {"synth_ast", "synth_llm"}
