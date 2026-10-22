"""Tests for FD-26 selection/combine/batches/CI (real imports, hermetic fixtures)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import combine_fd26
import select_fd26_errors
from src.fd26util import contains_quote, key_of, norm_text


def test_key_of_shared_between_scripts():
    import make_fd26_batches  # noqa: F401 — модуль батчинга импортируется
    assert select_fd26_errors.key_of is key_of
    assert combine_fd26.key_of is key_of


def test_key_of_exact_digest_and_separator():
    # Буквальный эталон: sha1("a\x00b"), полный hexdigest.
    assert key_of("a", "b") == "4a3dec2d1f8245280855c42db0ee4239f917fdb8"
    assert key_of("ab", "c") == "dbdd4f85d8a56500aa5c9c8a0d456f96280c92e5"
    assert key_of("a", "bc") == "0b2749668f0ea8df8a630da13f0d218709efd5ca"
    assert key_of("ab", "c") != key_of("a", "bc")


def test_key_of_nan_guard_exact():
    assert key_of(float("nan"), float("nan")) == key_of("", "")
    assert key_of(None, "x") == key_of("", "x")


def test_quote_normalization_crlf_and_spaces():
    doc = "Line one.\r\nReturns:\r\n    int: value.  "
    assert contains_quote(doc, "int: value.")
    assert contains_quote(doc, '  "int: value."  ')
    assert not contains_quote(doc, "")
    assert not contains_quote(doc, "missing statement")
    assert not contains_quote("only one line", "One")


def test_validate_quotes_all_branches(tmp_path):
    full = pd.DataFrame([{"err_id": "e1", "doc_old": "alpha beta gamma"},
                         {"err_id": "e2", "doc_old": "alpha beta gamma"}])
    ok_a = pd.DataFrame([{"err_id": "e1", "label": "A", "stale_statement": "beta",
                          "why": "x"}])
    combine_fd26.validate_quotes(full, ok_a, "t")  # не падает
    bad_a = pd.DataFrame([{"err_id": "e1", "label": "A", "stale_statement": "zzz",
                           "why": "x"}])
    with pytest.raises(SystemExit):
        combine_fd26.validate_quotes(full, bad_a, "t")
    empty_a = pd.DataFrame([{"err_id": "e1", "label": "A", "stale_statement": "",
                             "why": "x"}])
    with pytest.raises(SystemExit):
        combine_fd26.validate_quotes(full, empty_a, "t")
    for label in ("B", "C", "D"):
        bad = pd.DataFrame([{"err_id": "e1", "label": label,
                             "stale_statement": "beta", "why": "x"}])
        with pytest.raises(SystemExit):
            combine_fd26.validate_quotes(full, bad, "t")
        ok = pd.DataFrame([{"err_id": "e1", "label": label,
                            "stale_statement": "", "why": "x"}])
        combine_fd26.validate_quotes(full, ok, "t")


def test_assign_cause_priority_and_rules():
    ac = combine_fd26.assign_cause
    assert ac("D", 0, 1, "FP", "none", "False", "False") == "needs_context"
    assert ac("C", 1, 0, "FN", "none", True, False) == "noise"  # нативный bool
    assert ac("C", 0, 1, "FP", "none", "False", True) == "truncation"
    assert ac("A", 1, 0, "FN", "return_changed", "False", "False") == "fn_miss"
    assert ac("C", 0, 1, "FP", "param_renamed", "False", "False") == "sig_overtrigger"
    assert ac("C", 0, 1, "FP", "none", "False", "False") == "neutral_refactor"


def test_assign_cause_rejects_bad_inputs():
    ac = combine_fd26.assign_cause
    with pytest.raises(ValueError):
        ac("", 1, 0, "FN", "none", "False", "False")
    with pytest.raises(ValueError):
        ac("X", 1, 0, "FN", "none", "False", "False")
    with pytest.raises(ValueError):
        ac("A", 1, 0, "XX", "none", "False", "False")
    with pytest.raises(ValueError):
        ac("C", 0, 1, "FP", "", "False", "False")
    with pytest.raises(ValueError):
        ac("C", 0, 1, "FP", None, "False", "False")


def test_load_llm_rejects_bad_labels_and_dups(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("err_id,label,stale_statement,why\ne001,Z,,\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        combine_fd26.load_llm(bad, "t")
    dup = tmp_path / "dup.csv"
    dup.write_text("err_id,label,stale_statement,why\ne001,C,,\ne001,C,,\n",
                   encoding="utf-8")
    with pytest.raises(SystemExit):
        combine_fd26.load_llm(dup, "t")


def _mini_artifacts(tmp_path: Path, with_leak: bool = False):
    full = pd.DataFrame([{
        "err_id": "e001", "pair_id": "p1", "split": "gold", "source": "s",
        "project": "pr", "symbol": "f", "sig_old": "def f():",
        "diff_code": "DIFF", "doc_old": "Docs here.",
        "label_stale": "1", "drift_type": "unknown", "sig_type": "none",
        "label_source": "gold_human", "n_src_lines": "5", "p_stale": "0.1",
        "m2_pred": "0", "error_type": "FN", "truncated": "False",
        "doc_truncated": "False", "n_a_tok": "3", "n_b_tok": "2"}])
    anno_cols = ["err_id", "project", "symbol", "sig_old", "diff_code", "doc_old",
                 "sig_type", "n_src_lines"]
    if with_leak:
        anno_cols = anno_cols + ["split", "source"]
    anno = full[anno_cols].copy()
    manifest = {"n_total": 1, "n_gold": 1, "n_test_real": 0,
                "n_truncated_A": 0, "n_truncated_doc": 0,
                "err_id_to_pair_id": {"e001": "p1"},
                "keys": {"e001": key_of("DIFF", "Docs here.")}}
    dp = tmp_path / "d"
    dp.mkdir()
    fp, ap, mp = dp / "full.csv", dp / "anno.csv", dp / "m.json"
    full.to_csv(fp, index=False)
    anno.to_csv(ap, index=False)
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    sd = dp / "shards"
    sd.mkdir()
    anno.to_csv(sd / "fd26_shard_00.csv", index=False)
    return fp, ap, mp, sd


def test_check_artifacts_hermetic_ok(tmp_path):
    fp, ap, mp, sd = _mini_artifacts(tmp_path)
    assert combine_fd26.check_artifacts(fp, ap, mp, sd) == 0


def test_check_artifacts_rejects_leak_and_bad_key(tmp_path):
    fp, ap, mp, sd = _mini_artifacts(tmp_path, with_leak=True)
    with pytest.raises(AssertionError):
        combine_fd26.check_artifacts(fp, ap, mp, sd)


def test_blind_columns_have_no_leak():
    blind = {"err_id", "project", "symbol", "sig_old", "diff_code", "doc_old",
             "sig_type", "n_src_lines"}
    forbidden = {"split", "source", "label_stale", "m2_p", "m2_pred",
                 "error_type", "truncated", "doc_truncated",
                 "spark", "gemini", "consensus", "human", "final", "cause"}
    assert not (blind & forbidden)


def test_cause_ci_script(tmp_path, monkeypatch):
    import fd26_cause_ci
    df = pd.DataFrame({
        "err_id": ["e1", "e2", "e3"],
        "split": ["gold", "gold", "test_real"],
        "cause": ["noise", "sig_overtrigger", "noise"]})
    inp = tmp_path / "comb.csv"
    outp = tmp_path / "ci.json"
    df.to_csv(inp, index=False)
    monkeypatch.setattr(sys, "argv", ["fd26_cause_ci.py", "--combined", str(inp),
                                      "--out", str(outp), "--n_boot", "50", "--seed", "42"])
    fd26_cause_ci.main()
    ci = json.loads(outp.read_text(encoding="utf-8"))
    assert ci["n_boot"] == 50 and ci["seed"] == 42
    assert set(ci["splits"]) == {"gold", "test_real"}
    assert set(ci["causes"]) == {"noise", "sig_overtrigger"}
    for info in ci["splits"].values():
        assert all(v["count"] >= 0 for v in info["causes"].values())
        assert abs(sum(v["share"] for v in info["causes"].values()) - 1.0) < 1e-9


def test_norm_text_strips_quotes():
    assert norm_text('  "abc"  ') == "abc"
    assert norm_text("a\r\nb") == "a\nb"
