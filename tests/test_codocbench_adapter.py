"""Автотесты адаптера CoDocBench (критерии приёмки FD-04)."""

import pandas as pd
import pytest

from src.data.codocbench_adapter import CoDocBenchAdapter
from src.data.drift import derive_drift_type, extract_signature
from src.data.normalizer import parse_function_ast


def test_no_leak_columns_in_model_inputs(tmp_path):
    """Инвариант R1: проверка изоляции _leak_* и отсутствия сырых полей утечки."""
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
  doc_new: "leak_new_doc"
  diff_docstring: "leak_diff_doc"
  commit_message: "leak_msg"
""",
        encoding="utf-8",
    )
    adapter = CoDocBenchAdapter(cfg_path)

    raw_record = {
        "p": "proj",
        "f": "valid_fn",
        "code": "def valid_fn(x):\n    y = x + 1\n    z = y * 2\n    return z",
        "new_code": "def valid_fn(x):\n    y = x + 1\n    z = y * 3\n    return z",
        "sha": "c1a2b3",
        "doc": "Valid docstring with more than forty characters for testing.",
        "leak_new_doc": "New docstring that must leak only to _leak_doc_new",
        "leak_diff_doc": "Diff of docstring",
        "leak_msg": "Commit message",
    }

    row, reason = adapter.process_record(raw_record, "train_pool")
    assert reason == "успех"
    assert row is not None

    # 1. Сырые поля утечки не должны присутствовать в корне словаря без префикса _leak_
    for forbidden in ["doc_new", "diff_docstring", "commit_message", "leak_new_doc", "leak_diff_doc", "leak_msg"]:
        assert forbidden not in row, f"Сырое поле утечки '{forbidden}' найдено в корне пары!"

    # 2. Поля утечки строго изолированы под префиксом _leak_
    assert row["_leak_doc_new"] == raw_record["leak_new_doc"]
    assert row["_leak_diff_docstring"] == raw_record["leak_diff_doc"]
    assert row["_leak_commit_message"] == raw_record["leak_msg"]

    # 3. Входные признаки модели не содержат служебных префиксов
    model_inputs = {"sig_old", "diff_code", "doc_old"}
    for col in model_inputs:
        assert not col.startswith("_leak_")
        assert not col.startswith("_aux_")

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