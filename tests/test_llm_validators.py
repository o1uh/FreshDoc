import json

from src.synth.llm_validators import normalize_ws, parse_batch_response, validate_p1, validate_p2


def _old_src() -> str:
    # Docstring ≥80 символов и ≥3 строк — проходит thin-фильтр P1 (FD-11 fix).
    return (
        "def clamp(value, low=0, high=100):\n"
        '    """Limit value to the range [low, high].\n'
        "    Values above high are returned as high.\n"
        '    Values below low are returned as low."""\n'
        "    return max(low, min(value, high))"
    )


def _p1_new() -> str:
    return (
        "def clamp(value, low=0, high=100):\n"
        '    """Limit value to the range [low, high].\n'
        "    Values above high are returned as high.\n"
        '    Values below low are returned as low."""\n'
        "    return max(low, value) if value <= high else low"
    )


def _p2_new() -> str:
    return (
        "def clamp(value, low=0, high=100):\n"
        '    """Limit value to the range [low, high].\n'
        "    Values above high are returned as high.\n"
        '    Values below low are returned as low."""\n'
        "    res = max(low, min(value, high))\n"
        "    return res"
    )


def _p1_obj(new_src: str | None = None) -> dict:
    return {
        "new_function_source": _p1_new() if new_src is None else new_src,
        "change_summary": "changed",
        "stale_statements": ["Values above high are returned as high."],
        "why_stale": "now returns low",
    }


def test_p1_valid_accepted():
    ok, reason, parts = validate_p1(_old_src(), _p1_obj())
    assert ok and reason == "ok"
    assert parts is not None and parts["diff"] and parts["doc_old"]


def test_p2_valid_accepted():
    ok, reason, parts = validate_p2(
        _old_src(),
        {"new_function_source": _p2_new(), "change_summary": "refactor", "docstring_still_true": True},
    )
    assert ok and reason == "ok"
    assert parts is not None


def test_parse_batch_response_fenced_array():
    text = 'вот ответ:\n```json\n[{"id": "P0001", "skip": true}]\n```\nконец'
    objects, error = parse_batch_response(text)
    assert error is None
    assert objects == [{"id": "P0001", "skip": True}]


def test_parse_batch_response_bad_json():
    objects, error = parse_batch_response("это вообще не json {{{")
    assert objects == [] and error is not None


def test_reject_signature_changed():
    new_src = (
        "def clamp(value, low=0, high=100, strict=False):\n"
        '    """Limit value to the range [low, high].\n'
        "    Values above high are returned as high.\n"
        '    Values below low are returned as low."""\n'
        "    return max(low, min(value, high))"
    )
    ok, reason, _ = validate_p1(_old_src(), _p1_obj(new_src))
    assert not ok and reason == "signature_changed"


def test_reject_docstring_changed():
    new_src = _p1_new().replace("Values above high are returned as high.", "Values are clamped.")
    ok, reason, _ = validate_p1(_old_src(), _p1_obj(new_src))
    assert not ok and reason == "docstring_changed"


def test_reject_diff_too_big():
    body = "\n".join(f"    tmp{i} = {i}" for i in range(12)) + "\n    return max(low, min(value, high))"
    new_src = (
        "def clamp(value, low=0, high=100):\n"
        '    """Limit value to the range [low, high].\n'
        "    Values above high are returned as high.\n"
        '    Values below low are returned as low."""\n' + body
    )
    ok, reason, _ = validate_p1(_old_src(), _p1_obj(new_src))
    assert not ok and reason == "diff_too_big"


def test_reject_empty_stale_statements():
    obj = _p1_obj()
    obj["stale_statements"] = []
    ok, reason, _ = validate_p1(_old_src(), obj)
    assert not ok and reason == "empty_stale_statements"


def test_reject_stale_statement_not_found():
    obj = _p1_obj()
    obj["stale_statements"] = ["Такого предложения в докстринге нет."]
    ok, reason, _ = validate_p1(_old_src(), obj)
    assert not ok and reason == "stale_statement_not_found"


def test_reject_p2_flag_not_true():
    ok, reason, _ = validate_p2(
        _old_src(),
        {"new_function_source": _p2_new(), "change_summary": "refactor", "docstring_still_true": False},
    )
    assert not ok and reason == "docstring_flag_not_true"


def test_normalize_ws_collapses():
    assert normalize_ws("a  \n  b") == "a b"


# --- FD-11 fix: новые фильтры P1 (thin / coverage / lexical) ---


def test_reject_thin_docstring():
    thin_src = (
        "def tiny(x):\n"
        '    """Do thing."""\n'
        "    return x + 1"
    )
    thin_new = (
        "def tiny(x):\n"
        '    """Do thing."""\n'
        "    return x + 2"
    )
    obj = {
        "new_function_source": thin_new,
        "change_summary": "changed",
        "stale_statements": ["Do thing."],
        "why_stale": "changed",
    }
    ok, reason, _ = validate_p1(thin_src, obj)
    assert not ok and reason == "thin_docstring"


def test_reject_stale_not_linked_to_diff():
    # cpython::assertCountEqual — своп аргументов внутр. хелпера, цитируется
    # общее предложение; значимого пересечения с диффом нет.
    old_src = (
        "def assertCountEqual(self, actual, expected, msg=None):\n"
        '    """An unordered sequence specific comparison. It asserts that\n'
        "    actual and expected have the same element counts. The method\n"
        "    checks frequencies with a helper and reports mismatches\n"
        '    in detail for debugging purposes."""\n'
        "    if actual == expected:\n"
        "        return\n"
        "    differences = _count_diff_hashable(actual_seq, expected_seq)\n"
        "    if differences:\n"
        "        raise AssertionError(differences)"
    )
    new_src = old_src.replace(
        "_count_diff_hashable(actual_seq, expected_seq)",
        "_count_diff_hashable(expected_seq, actual_seq)",
    )
    obj = {
        "new_function_source": new_src,
        "change_summary": "swapped helper args",
        "stale_statements": ["An unordered sequence specific comparison."],
        "why_stale": "order now matters",
    }
    ok, reason, _ = validate_p1(old_src, obj)
    assert not ok and reason == "stale_not_linked_to_diff"


def test_reject_stale_covers_docstring():
    short_src = (
        "def add_html(self, source):\n"
        '    """Adds HTML source code to this TiddlyWiki. Extended description\n'
        "    of storage and rendering pipeline for the QA sample.\"\"\"\n"
        "    self.store(source)\n"
        "    self.render()"
    )
    short_new = (
        "def add_html(self, source):\n"
        '    """Adds HTML source code to this TiddlyWiki. Extended description\n'
        "    of storage and rendering pipeline for the QA sample.\"\"\"\n"
        "    self.store(source.strip())\n"
        "    self.render()"
    )
    obj = {
        "new_function_source": short_new,
        "change_summary": "strip",
        "stale_statements": [
            "Adds HTML source code to this TiddlyWiki. Extended description\n"
            "    of storage and rendering pipeline for the QA sample."
        ],
        "why_stale": "strip now",
    }
    ok, reason, _ = validate_p1(short_src, obj)
    assert not ok and reason in ("thin_docstring", "stale_covers_docstring", "stale_not_linked_to_diff")


def test_p1_accepts_concrete_param_or_return():
    # Конкретное опровергнутое условие возврата проходит все три фильтра.
    ok, reason, parts = validate_p1(_old_src(), _p1_obj())
    assert ok and reason == "ok"
    assert parts is not None
    assert "Values above high are returned as high." in parts["doc_old"]
