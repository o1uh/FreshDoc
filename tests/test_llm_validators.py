import json

from src.synth.llm_validators import normalize_ws, parse_batch_response, validate_p1, validate_p2


def _old_src() -> str:
    return (
        "def clamp(value, low=0, high=100):\n"
        '    """Limit value to the range [low, high]. Values above high are returned as high."""\n'
        "    return max(low, min(value, high))"
    )


def _p1_new() -> str:
    return (
        "def clamp(value, low=0, high=100):\n"
        '    """Limit value to the range [low, high]. Values above high are returned as high."""\n'
        "    return max(low, value) if value <= high else low"
    )


def _p2_new() -> str:
    return (
        "def clamp(value, low=0, high=100):\n"
        '    """Limit value to the range [low, high]. Values above high are returned as high."""\n'
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
        '    """Limit value to the range [low, high]. Values above high are returned as high."""\n'
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
        '    """Limit value to the range [low, high]. Values above high are returned as high."""\n' + body
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
