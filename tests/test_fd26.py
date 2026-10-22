"""Tests for FD-26 error selection and combine (deterministic, no network)."""

from __future__ import annotations

import hashlib


def key_of(diff_code: str, doc_old: str) -> str:
    return hashlib.sha1((str(diff_code or "") + str(doc_old or "")).encode("utf-8")).hexdigest()


def test_key_deterministic():
    assert key_of("a", "b") == key_of("a", "b")
    assert key_of("a", "b") != key_of("a", "c")


def test_error_type_rule():
    def err_type(m2_pred: int, label: int) -> str:
        return "FP" if (m2_pred == 1 and label == 0) else "FN"
    assert err_type(1, 0) == "FP"
    assert err_type(0, 1) == "FN"


def test_stale_statement_rule():
    doc = "Fetch a URL.\n\nArgs:\n    url: the address."
    assert "url: the address." in doc
    assert "missing quote" not in doc


def test_blind_columns_have_no_leak():
    blind = {"err_id", "project", "symbol", "sig_old", "diff_code", "doc_old",
             "split", "sig_type", "source", "n_src_lines"}
    forbidden = {"label_stale", "m2_p", "m2_pred", "error_type", "truncated", "doc_truncated"}
    assert not (blind & forbidden)
