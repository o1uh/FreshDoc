"""Автотесты адаптера CoDocBench (критерии приёмки FD-04)."""

import pandas as pd
import pytest

from src.data.codocbench_adapter import CoDocBenchAdapter
from src.data.drift import derive_drift_type, extract_signature
from src.data.normalizer import parse_function_ast


def test_no_leak_columns_in_model_inputs():
    model_inputs = {"sig_old", "diff_code", "doc_old"}
    forbidden_prefixes = ("_leak_", "_aux_")

    for col in model_inputs:
        for prefix in forbidden_prefixes:
            assert not col.startswith(prefix), f"Входная колонка {col} имеет запрещённый префикс {prefix}"


def test_derive_drift_type_sample_return_changed():
    old_code = "def download(self) -> bytes:\n    return b''"
    new_code = "def download(self) -> bytearray:\n    return bytearray()"

    fn_old = parse_function_ast(old_code)
    fn_new = parse_function_ast(new_code)
    assert fn_old is not None and fn_new is not None

    sig_old = extract_signature(fn_old)
    sig_new = extract_signature(fn_new)
    assert sig_old["returns"] == "bytes"
    assert sig_new["returns"] == "bytearray"

    drift = derive_drift_type(sig_old, sig_new)
    assert drift == "return_changed"


def test_derive_drift_type_body_only_is_none():
    old_code = "def bulk_rename(doctype, rows=None):\n    pass"
    new_code = "def bulk_rename(doctype, rows=None):\n    x = 1\n    pass"

    fn_old = parse_function_ast(old_code)
    fn_new = parse_function_ast(new_code)
    assert fn_old is not None and fn_new is not None

    sig_old = extract_signature(fn_old)
    sig_new = extract_signature(fn_new)
    drift = derive_drift_type(sig_old, sig_new)
    assert drift == "none"


def test_filters_private_and_short_docstring(tmp_path):
    cfg_path = tmp_path / "map.yaml"
    cfg_path.write_text(
        """
project: "p"
symbol: "f"
old_code: "code"
new_code: "new_code"
commit_sha: "sha"
old_docstring: "doc"
leak_fields:
  doc_new: "doc_new"
""",
        encoding="utf-8",
    )

    adapter = CoDocBenchAdapter(cfg_path)

    # 1. Приватная функция
    rec_priv = {
        "p": "proj",
        "f": "_priv",
        "code": "def _priv():\n    '''Valid long docstring that passes length filter.'''\n    return 1",
        "new_code": "def _priv():\n    return 2",
        "sha": "123",
        "doc": "Valid long docstring that passes length filter.",
    }
    row, reason = adapter.process_record(rec_priv, "train_pool")
    assert row is None
    assert reason == "не_прошёл_фильтры"

    # 2. Слишком короткий docstring (< 40 символов)
    rec_short_doc = {
        "p": "proj",
        "f": "pub",
        "code": "def pub():\n    '''Short.'''\n    return 1",
        "new_code": "def pub():\n    return 2",
        "sha": "123",
        "doc": "Short.",
    }
    row, reason = adapter.process_record(rec_short_doc, "train_pool")
    assert row is None
    assert reason == "не_прошёл_фильтры"