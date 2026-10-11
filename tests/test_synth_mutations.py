import ast
import random

from src.data.drift import derive_drift_type, extract_signature
from src.data.normalizer import (
    extract_docstring_and_clean_node,
    normalize_code_from_node,
    parse_function_ast,
)
from src.synth.mutations import (
    MUTATIONS,
    mut_change_message,
    mut_param_removed,
    mut_return_changed,
)

SAMPLE = '''def fetch(url, timeout=10, retries=3, *, verbose=False) -> str:
    """Fetch a URL.

    Args:
        url: the address to fetch.
        timeout: seconds to wait. Defaults to 10.
        retries: number of retries.
        verbose: print progress.

    Returns:
        The response body as a string.

    Raises:
        ValueError: if url is empty.
    """
    if not url:
        raise ValueError("empty url or bad scheme")
    body = _get(url, timeout)
    print("fetched ok from remote host")
    return body.decode("utf-8")
'''


def _setup():
    fn = parse_function_ast(SAMPLE)
    assert fn is not None
    doc, fn_no_doc = extract_docstring_and_clean_node(fn)
    return doc, fn_no_doc


def _check(mtype: str, seed: int = 0):
    func, label, expected = MUTATIONS[mtype]
    doc, fn_no_doc = _setup()
    code_old = normalize_code_from_node(fn_no_doc)
    sig_old = extract_signature(fn_no_doc)
    new_fn = func(fn_no_doc, doc, random.Random(seed))
    assert new_fn is not None, f"{mtype}: вернул None на образце R6"
    code_new = normalize_code_from_node(new_fn)
    assert code_new != code_old, f"{mtype}: код не изменился"
    drift = derive_drift_type(sig_old, extract_signature(new_fn))
    assert drift == expected, f"{mtype}: страховка {drift} != {expected}"
    # docstring не попал в код.
    assert "Fetch a URL" not in code_new
    return code_new, drift


def test_pos_param_renamed():
    _check("param_renamed")


def test_pos_param_added():
    _check("param_added")


def test_pos_param_removed():
    _check("param_removed")


def test_pos_default_changed():
    _check("default_changed")


def test_pos_return_changed():
    _check("return_changed")


def test_pos_exception_changed():
    _check("exception_changed")


def test_neg_rename_local():
    code_new, drift = _check("rename_local")
    assert drift == "none"


def test_neg_extract_temp():
    code_new, drift = _check("extract_temp")
    assert drift == "none"


def test_neg_change_message():
    code_new, drift = _check("change_message")
    assert drift == "none"


def test_param_removed_store_inapplicable():
    src = (
        "def acc(x):\n"
        "    '''Add one to x value and return the increased total amount here.'''\n"
        "    x = x + 1\n"
        "    return x\n"
    )
    fn = parse_function_ast(src)
    assert fn is not None
    doc, fn_no_doc = extract_docstring_and_clean_node(fn)
    assert mut_param_removed(fn_no_doc, doc, random.Random(0)) is None


def test_change_message_only_calls():
    src = (
        "def conv(flag):\n"
        "    '''Convert currency flag to code string, print status at the end.'''\n"
        "    code = 'USD' if flag else 'BTC'\n"
        "    print(\"done\")\n"
        "    return code\n"
    )
    fn = parse_function_ast(src)
    assert fn is not None
    doc, fn_no_doc = extract_docstring_and_clean_node(fn)
    new_fn = mut_change_message(fn_no_doc, doc, random.Random(0))
    assert new_fn is not None
    strs = [n.value for n in ast.walk(new_fn)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert "USD" in strs and "BTC" in strs
    assert "done" not in strs


def test_return_changed_optional_inapplicable():
    for ann in ("Optional[str]", "Union[str, None]"):
        src = (
            f"def maybe(x) -> {ann}:\n"
            "    '''Return the value or nothing when missing input data given.'''\n"
            "    return str(x)\n"
        )
        fn = parse_function_ast(src)
        assert fn is not None
        doc, fn_no_doc = extract_docstring_and_clean_node(fn)
        assert mut_return_changed(fn_no_doc, doc, random.Random(0)) is None
