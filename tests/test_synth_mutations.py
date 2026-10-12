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


def test_param_removed_none_default_call_guarded():
    src = (
        "def convert(items, ind2lab=None):\n"
        "    '''Convert batch items with the ind2lab mapping when given.'''\n"
        "    if ind2lab is not None:\n"
        "        items = ind2lab(items)\n"
        "    return items\n"
    )
    fn = parse_function_ast(src)
    assert fn is not None
    doc, fn_no_doc = extract_docstring_and_clean_node(fn)
    assert mut_param_removed(fn_no_doc, doc, random.Random(0)) is None


def test_param_removed_no_default_with_reads_inapplicable():
    src = (
        "def convert_page_format(value):\n"
        "    '''Convert value descriptor to pixel dimensions tuple.'''\n"
        "    if value in FORMATS:\n"
        "        return FORMATS[value]\n"
        "    return None\n"
    )
    fn = parse_function_ast(src)
    assert fn is not None
    doc, fn_no_doc = extract_docstring_and_clean_node(fn)
    assert mut_param_removed(fn_no_doc, doc, random.Random(0)) is None


def test_param_removed_prose_mention_not_documented():
    src = (
        "def create_orders(self, orders, params={}):\n"
        "    '''Create orders, each object should contain symbol, amount, price and params.'''\n"
        "    return self.post(orders, params)\n"
    )
    fn = parse_function_ast(src)
    assert fn is not None
    doc, fn_no_doc = extract_docstring_and_clean_node(fn)
    assert mut_param_removed(fn_no_doc, doc, random.Random(0)) is None


def test_exception_replace_undocumented_type_falls_back_to_guard():
    from src.data.normalizer import normalize_code_from_node
    from src.synth.mutations import mut_exception_changed

    src = (
        "def solve(data):\n"
        "    '''Solve the factorization problem for data matrix.'''\n"
        "    if data is None:\n"
        "        raise ValueError('bad data')\n"
        "    return fit(data)\n"
    )
    fn = parse_function_ast(src)
    assert fn is not None
    doc, fn_no_doc = extract_docstring_and_clean_node(fn)
    new_fn = mut_exception_changed(fn_no_doc, doc, random.Random(0))
    assert new_fn is not None
    code_new = normalize_code_from_node(new_fn)
    assert "ValueError" in code_new
    assert "if not data:" in code_new
    from src.data.drift import derive_drift_type, extract_signature
    from src.synth.mutations import MUTATIONS as _M

    assert derive_drift_type(extract_signature(fn_no_doc), extract_signature(new_fn)) == _M["exception_changed"][2]


def test_default_changed_value_only_without_default_word():
    from src.synth.mutations import mut_default_changed

    src = (
        "def nice_duration(duration, speech=True):\n"
        "    '''Convert duration: format for speech (True) or display (False).'''\n"
        "    return fmt(duration, speech)\n"
    )
    fn = parse_function_ast(src)
    assert fn is not None
    doc, fn_no_doc = extract_docstring_and_clean_node(fn)
    assert mut_default_changed(fn_no_doc, doc, random.Random(0)) is None


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


def test_default_changed_empty_default_no_false_hit():
    from src.synth.mutations import mut_default_changed

    src = (
        "def is_line_short_enough(line_str=\"\"):\n"
        "    '''Uses the provided line_str rendering, if any, otherwise computes a new one.'''\n"
        "    return len(line_str) < 80\n"
    )
    fn = parse_function_ast(src)
    assert fn is not None
    doc, fn_no_doc = extract_docstring_and_clean_node(fn)
    assert mut_default_changed(fn_no_doc, doc, random.Random(0)) is None


def test_generate_neg_samples_types_with_replacement():
    import pandas as pd

    from src.synth.generate import generate_for_pool

    rows = []
    for i in range(5):
        src = (
            f"def f{i}(aaa):\n"
            f"    '''Process aaa value and return the computed result number {i} here.'''\n"
            f"    tmp_local = aaa + {i}\n"
            f"    aux_val = tmp_local * 2\n"
            "    print(aux_val)\n"
        )
        rows.append({"project": "p", "src": src, "function_hash": f"h{i}"})
    df_pool = pd.DataFrame(rows)
    df, stats = generate_for_pool(df_pool, seed=42)
    neg = df[df["label_stale"] == 0]
    # Применим только rename_local: без возвращения типов потолок был бы 5 пар.
    assert len(neg) > 5
    assert neg["pair_id"].is_unique
