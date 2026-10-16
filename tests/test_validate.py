"""Tests for FD-13 dataset quality checks (R8).

Фиктивный мини-датасет (без реальных данных — гоняется в CI):
реальные проекты pa (train) / pb (val) / pc (test_real), синтетика проекта
pa в train/test_synth на непересекающихся хешах.
"""

from __future__ import annotations

import pandas as pd

from src.data.validate import (
    MODEL_INPUT_COLS,
    all_ok,
    check_drift_label_consistency,
    check_filters_uniform,
    check_label_source_valid,
    check_label_values,
    check_model_inputs_isolated,
    check_pair_id_unique,
    check_real_project_single_split,
    check_sig_type_valid,
    check_splits_valid,
    check_synth_function_single_split,
    check_synth_only_train_pool,
    check_synth_split_allowed,
    check_train_hash_isolation,
    check_weights_consistent,
    run_all_checks,
    select_model_inputs,
)

DOC = "Do thing with documented params a and b. " + "x" * 40


def _row(pair_id, source, project, fhash, split, label, drift, label_source,
         weight, symbol="fetch", doc=DOC, n_lines=10, sig="none",
         code_old="def fetch(a):\n    return a"):
    return {
        "pair_id": pair_id, "source": source, "project": project,
        "commit_sha": "abc", "symbol": symbol, "function_hash": fhash,
        "sig_old": "def fetch(a):", "code_old": code_old,
        "code_new": code_old, "diff_code": "@@\n-a\n+b",
        "doc_old": doc, "label_stale": label, "drift_type": drift,
        "sig_type": sig, "label_source": label_source, "weight": weight,
        "n_src_lines": n_lines, "split": split,
        "_leak_doc_new": "new", "_leak_diff_docstring": "d",
        "_leak_commit_message": "m", "_aux_src_old": "src",
    }


def _valid_df() -> pd.DataFrame:
    rows = [
        _row("c1", "codocbench", "pa", "h1", "train", 1, "param_added",
             "real_coupled", 1.0, sig="param_added",
             code_old="def fetch(a):\n    return a"),
        _row("n1", "history_neg", "pa", "h2", "train", 0, "none",
             "real_weak_neg", 0.5),
        _row("c2", "codocbench", "pb", "h3", "val", 1, "unknown",
             "real_coupled", 1.0),
        _row("n2", "history_neg", "pb", "h4", "val", 0, "none",
             "real_strong_neg", 1.0),
        _row("c3", "codocbench", "pc", "h5", "test_real", 1, "param_renamed",
             "real_coupled", 1.0, sig="param_renamed"),
        _row("n3", "history_neg", "pc", "h6", "test_real", 0, "none",
             "real_weak_neg", 0.5),
        _row("s1", "synth_ast", "pa", "h7", "train", 1, "param_added",
             "ast", 1.0, sig="param_added"),
        _row("s2", "synth_ast", "pa", "h8", "train", 0, "none", "ast", 1.0),
        _row("s3", "synth_llm", "pa", "h9", "train", 1, "behavior_changed",
             "llm_p1", 0.5),
        _row("s4", "synth_ast", "pa", "h10", "test_synth", 1, "return_changed",
             "ast", 1.0, sig="return_changed"),
        _row("s5", "synth_llm", "pa", "h11", "train", 0, "none",
             "llm_p2", 0.5),
    ]
    return pd.DataFrame(rows)


def _pools():
    return {"pa": "train_pool", "pb": "train_pool", "pc": "test_pool"}


def test_all_pass_on_valid_mini_dataset():
    df = _valid_df()
    results = run_all_checks(df, _pools())
    assert all_ok(results), [(r.name, r.detail) for r in results if not r.ok]


def test_pair_id_duplicate():
    df = _valid_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    r = check_pair_id_unique(df)
    assert not r.ok and r.n_fail == 2


def test_empty_doc_and_bad_label():
    df = _valid_df()
    df.loc[df["pair_id"] == "c1", "doc_old"] = ""
    df.loc[df["pair_id"] == "n1", "label_stale"] = -1
    assert not check_label_values(df).ok
    # пустой doc ловится и фильтром, и nonempty-проверкой через run_all
    results = {r.name: r for r in run_all_checks(df, _pools())}
    assert not results["label_values"].ok
    assert not results["filters_uniform"].ok


def test_filters_uniform_each_reason():
    df = _valid_df()
    df.loc[df["pair_id"] == "c1", "symbol"] = "pkg._priv"
    df.loc[df["pair_id"] == "n1", "doc_old"] = "short"
    df.loc[df["pair_id"] == "c2", "n_src_lines"] = 500
    r = check_filters_uniform(df)
    assert not r.ok and r.n_fail == 3
    assert "имя с _: 1" in r.detail and "doc<40: 1" in r.detail


def test_synth_wrong_split():
    df = _valid_df()
    df.loc[df["pair_id"] == "s2", "split"] = "val"
    assert not check_synth_split_allowed(df).ok


def test_synth_hash_multi_split():
    df = _valid_df()
    # тот же хеш h7 в train и test_synth
    df.loc[df["pair_id"] == "s4", "function_hash"] = "h7"
    r = check_synth_function_single_split(df)
    assert not r.ok and r.n_fail == 1


def test_real_project_multi_split():
    df = _valid_df()
    df.loc[df["pair_id"] == "n1", "split"] = "val"  # проект pa в train и val
    r = check_real_project_single_split(df)
    assert not r.ok and "pa" in r.detail


def test_train_hash_leak_across_projects():
    """Вендоринг: тот же код (хеш) в train и test_real у разных проектов."""
    df = _valid_df()
    df.loc[df["pair_id"] == "c3", "function_hash"] = "h1"  # хеш train-строки
    r = check_train_hash_isolation(df)
    assert not r.ok and "test_real: 1" in r.detail


def test_train_hash_leak_test_synth():
    df = _valid_df()
    df.loc[df["pair_id"] == "s4", "function_hash"] = "h2"
    assert not check_train_hash_isolation(df).ok


def test_drift_mismatch_each_source():
    df = _valid_df()
    df.loc[df["pair_id"] == "s1", "drift_type"] = "none"       # ast pos
    df.loc[df["pair_id"] == "s2", "drift_type"] = "param_added"  # ast neg
    df.loc[df["pair_id"] == "s3", "drift_type"] = "none"       # llm_p1
    df.loc[df["pair_id"] == "n1", "label_stale"] = 1           # history_neg
    df.loc[df["pair_id"] == "c1", "drift_type"] = "none"       # codocbench
    r = check_drift_label_consistency(df)
    assert not r.ok
    assert "ast pos" in r.detail and "ast neg" in r.detail
    assert "llm_p1" in r.detail and "history_neg" in r.detail
    assert "codocbench" in r.detail


def test_model_inputs_have_no_leak_columns():
    assert MODEL_INPUT_COLS == ["sig_old", "diff_code", "doc_old"]
    assert not any(c.startswith(("_leak_", "_aux_")) for c in MODEL_INPUT_COLS)
    assert check_model_inputs_isolated(_valid_df()).ok


def test_select_model_inputs_returns_only_contract_columns():
    out = select_model_inputs(_valid_df())
    assert list(out.columns) == ["sig_old", "diff_code", "doc_old"]
    assert not any(c.startswith(("_leak_", "_aux_")) for c in out.columns)


def test_model_inputs_catches_leak_in_contract(monkeypatch):
    """Дефект 1: проверка живая — утечка в контракте краснеет."""
    import src.data.validate as V
    monkeypatch.setattr(V, "MODEL_INPUT_COLS", ["sig_old", "diff_code", "_leak_doc_new"])
    r = check_model_inputs_isolated(_valid_df())  # _leak_doc_new есть в таблице
    assert not r.ok and "_leak_doc_new" in r.detail


def test_label_source_vocabulary():
    """Дефект 3: вне-словарные значения краснеют, словарные проходят."""
    assert check_label_source_valid(_valid_df()).ok
    df = _valid_df()
    df.loc[df["pair_id"] == "c1", "label_source"] = "typo_src"
    r = check_label_source_valid(df)
    assert not r.ok and "typo_src" in r.detail


def test_unknown_label_source_fails_weights():
    """Дефект 2: неизвестный источник больше не проходит молча."""
    df = _valid_df()
    df.loc[df["pair_id"] == "c1", "label_source"] = "typo_src"
    r = check_weights_consistent(df)
    assert not r.ok and "typo_src" in r.detail


def test_gold_human_weight_zero_ok_but_unlabeled_fails_weights():
    base = _valid_df().iloc[[0]].copy()
    gold = base.assign(pair_id="g1", source="gold", project="pc",
                       split="gold", label_source="gold_human", weight=0.0,
                       function_hash="h_gold", drift_type="param_added")
    assert check_weights_consistent(gold).ok
    unl = base.assign(pair_id="u1", label_source="real_unlabeled", weight=0.0,
                      function_hash="h_unl")
    r = check_weights_consistent(unl)
    assert not r.ok and "real_unlabeled" in r.detail


def test_weights_mismatch():
    df = _valid_df()
    df.loc[df["pair_id"] == "n1", "weight"] = 1.0  # weak должен быть 0.5
    r = check_weights_consistent(df)
    assert not r.ok and "real_weak_neg" in r.detail


def test_sig_type_and_split_invalid():
    df = _valid_df()
    df.loc[df["pair_id"] == "c1", "sig_type"] = "behavior_changed"
    assert not check_sig_type_valid(df).ok
    df2 = _valid_df()
    df2.loc[df2["pair_id"] == "c1", "split"] = "quiz"
    assert not check_splits_valid(df2).ok


def test_synth_only_train_pool():
    df = _valid_df()
    assert check_synth_only_train_pool(df, _pools()).ok
    df.loc[df["pair_id"] == "s1", "project"] = "pc"  # test-пул
    r = check_synth_only_train_pool(df, _pools())
    assert not r.ok and "pc" in r.detail


def test_pool_check_skipped_without_pools():
    results = run_all_checks(_valid_df(), None)
    assert all_ok(results)
    assert "synth_only_train_pool" not in [r.name for r in results]


def test_missing_columns_fail_gracefully():
    df = _valid_df().drop(columns=["split", "function_hash"])
    results = run_all_checks(df, None)
    assert not all_ok(results)
    by_name = {r.name: r for r in results}
    assert not by_name["splits_valid"].ok
    assert "нет колонок" in by_name["splits_valid"].detail
    assert not by_name["train_hash_isolation"].ok
