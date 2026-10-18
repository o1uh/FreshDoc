"""FD-20: тесты бейзлайнов M0/M1 (правило, признаки R10, обучение)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.baselines import (
    ALLOWED_FEATURE_COLS,
    FEATURE_COLS,
    count_diff_lines,
    extract_features,
    predict_m0,
    predict_proba_m1,
    train_m1,
)

OLD = (
    "def fetch(url, timeout=10, retries=3, *, verbose=False) -> str:\n"
    "    body = _get(url, timeout)\n"
    "    return body.decode()"
)
NEW_RENAMED = (
    "def fetch(endpoint, timeout=10, retries=3, *, verbose=False) -> str:\n"
    "    body = _get(endpoint, timeout)\n"
    "    return body.decode()"
)
NEW_COSMETIC = (
    "def fetch(url, timeout=10, retries=3, *, verbose=False) -> str:\n"
    "    res = _get(url, timeout)\n"
    "    return res.decode()"
)
DOC = (
    "Fetch a URL.\n\nArgs:\n    url: the address to fetch.\n"
    "    timeout: seconds to wait. Defaults to 10.\n\n"
    "Returns:\n    The response body as a string."
)
DIFF_RENAMED = (
    "--- a/code.py\n+++ b/code.py\n@@ -1 +1 @@\n"
    "-def fetch(url, timeout=10) -> str:\n"
    "+def fetch(endpoint, timeout=10) -> str:"
)


def _row(**kw) -> pd.DataFrame:
    base = {
        "pair_id": "t1",
        "sig_old": "def fetch(url, timeout=10) -> str:",
        "diff_code": DIFF_RENAMED,
        "doc_old": DOC,
        "code_old": OLD,
        "code_new": NEW_RENAMED,
        "sig_type": "param_renamed",
        "split": "train",
        "source": "synth_ast",
        "label_stale": 1,
        "weight": 1.0,
    }
    base.update(kw)
    return pd.DataFrame([base])


def test_m0_rule_by_sig_type():
    df = pd.DataFrame([
        {"pair_id": "a", "sig_type": "none"},
        {"pair_id": "b", "sig_type": "param_added"},
        {"pair_id": "c", "sig_type": "exception_changed"},
    ])
    p = predict_m0(df)
    assert p.tolist() == [0.0, 1.0, 1.0]


def test_diff_counts_skip_headers():
    added, removed = count_diff_lines(DIFF_RENAMED)
    assert (added, removed) == (1, 1)
    a2, r2 = count_diff_lines("--- a\n+++ b\n context\n+new\n-old")
    assert (a2, r2) == (1, 1)
    assert count_diff_lines("") == (0, 0)
    assert count_diff_lines(None) == (0, 0)


def test_features_columns_and_no_nan():
    X = extract_features(_row())
    assert list(X.columns) == FEATURE_COLS
    assert not X.isna().any().any()


def test_features_link_renamed_mentioned():
    X = extract_features(_row())
    assert X.loc[0, "sig_param_renamed"] == 1.0
    assert X.loc[0, "link_removed_mentioned"] == 1.0  # 'url' есть в docstring
    assert X.loc[0, "n_params_mentioned"] >= 1.0
    assert X.loc[0, "has_args"] == 1.0
    assert X.loc[0, "has_returns"] == 1.0
    assert X.loc[0, "has_default_word"] == 1.0


def test_features_cosmetic_no_link():
    X = extract_features(_row(code_old=OLD, code_new=NEW_COSMETIC,
                              sig_type="none",
                              diff_code="--- a\n+++ b\n@@\n-    body = _get(url)\n+    res = _get(url)"))
    assert X.loc[0, "sig_none"] == 1.0
    assert X.loc[0, "link_removed_mentioned"] == 0.0
    assert X.loc[0, "diff_added"] == 1.0
    assert X.loc[0, "diff_removed"] == 1.0


def test_features_do_not_use_forbidden_columns():
    """Перестановка project/source/label не меняет признаки."""
    df1 = _row(project="aaa", source="codocbench", label_stale=1,
               drift_type="param_renamed", weight=1.0,
               _leak_doc_new="NEW", _aux_src_old="SRC")
    df2 = _row(project="zzz", source="history_neg", label_stale=0,
               drift_type="none", weight=0.5,
               _leak_doc_new="OTHER", _aux_src_old="OTHER SRC")
    keep = [c for c in df1.columns if c in ALLOWED_FEATURE_COLS]
    assert keep, "нет разрешённых колонок"
    pd.testing.assert_frame_equal(
        extract_features(df1).reset_index(drop=True),
        extract_features(df2).reset_index(drop=True),
    )


def test_train_m1_tiny_separable():
    pos = _row(pair_id="p", sig_type="param_added", label_stale=1)
    neg = _row(pair_id="n", sig_type="none", code_new=NEW_COSMETIC,
               label_stale=0,
               diff_code="--- a\n+++ b\n@@\n-    body = _get(url)\n+    res = _get(url)")
    df = pd.concat([pos, neg] * 10, ignore_index=True)
    X = extract_features(df)
    y = df["label_stale"].to_numpy(dtype=int)
    w = np.ones(len(df))
    clf = train_m1(X, y, sample_weight=w, seed=42, max_iter=50,
                   min_samples_leaf=1)
    p = predict_proba_m1(clf, X)
    assert ((p >= 0.0) & (p <= 1.0)).all()
    assert float(p[y == 1].mean()) > float(p[y == 0].mean())


def _flags(doc: str):
    X = extract_features(_row(doc_old=doc))
    return (X.loc[0, "has_args"], X.loc[0, "has_returns"], X.loc[0, "has_raises"])


def test_section_regexes_require_format_marker():
    # Проза без маркера — не секция (бывшие ложные срабатывания).
    assert _flags("Return the prototype structure of JobConfig")[1] == 0.0
    assert _flags("Return list of models, sorted alphabetically")[1] == 0.0
    assert _flags("Handles network errors gracefully.")[2] == 0.0
    assert _flags("Get app info.\n\nReturn example:\n    {\"a\": 1}")[1] == 0.0
    # Настоящие секции в трёх форматах.
    assert _flags("Fetch a URL.\n\nArgs:\n    url: the address")[0] == 1.0
    assert _flags("Fetch a URL.\n\nReturns:\n    The body")[1] == 1.0
    assert _flags("Fetch.\n\nRaises:\n    ValueError: if empty")[2] == 1.0
    assert _flags("Fetch.\n\nParameters\n----------\nx: int")[0] == 1.0
    assert _flags(":param str symbol: unified market symbol")[0] == 1.0
    assert _flags(":returns Order[]: a list of orders")[1] == 1.0
    assert _flags(":raises ValueError: if empty")[2] == 1.0


def test_parse_sigs_independent_broken_new():
    """Битый code_new не обнуляет признаки старой сигнатуры."""
    from src.models.baselines import _parse_sigs

    sold, snew = _parse_sigs(OLD, "def broken(((", "def fetch(url, timeout=10)")
    assert sold is not None and snew is None
    X = extract_features(_row(code_old=OLD, code_new="def broken(((",
                              sig_type="param_renamed"))
    assert X.loc[0, "n_params_old"] == 4.0
    assert X.loc[0, "n_params_mentioned"] >= 1.0
    # Резерв n_sig_changes по sig_type, раз сравнить не с чем.
    assert X.loc[0, "n_sig_changes"] == 1.0


def test_parse_sigs_sig_old_fallback():
    """Пустой code_old: старая сигнатура восстанавливается из sig_old."""
    X = extract_features(_row(code_old="", code_new=NEW_COSMETIC,
                              sig_old="def fetch(url, timeout=10) -> str:",
                              sig_type="none"))
    assert X.loc[0, "n_params_old"] == 2.0
    assert X.loc[0, "n_params_mentioned"] >= 1.0
