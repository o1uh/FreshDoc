import pandas as pd

from src.data.normalizer import (
    extract_docstring_and_clean_node,
    normalize_code_from_node,
    parse_function_ast,
)
from src.synth.pool import build_functions_pool, build_functions_pool_v2, make_function_hash


def _doc(n: int = 60) -> str:
    return "Документация функции. " + "x" * n


def _src(name: str = "fetch", doc_len: int = 60, n_body: int = 5) -> str:
    body = "\n".join(f"    v{i} = {i}" for i in range(n_body))
    return f"def {name}(url, timeout=10):\n    '''{_doc(doc_len)}'''\n{body}\n    return v0"


def _pairs(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_no_test_pool_projects():
    df_pairs = _pairs(
        [
            {"project": "p_train", "split_pool": "train_pool", "function_hash": "h1", "_aux_src_old": _src()},
            {"project": "p_test", "split_pool": "train_pool", "function_hash": "h2", "_aux_src_old": _src()},
        ]
    )
    df_pools = pd.DataFrame(
        [{"project": "p_train", "pool": "train_pool"}, {"project": "p_test", "pool": "test_pool"}]
    )
    try:
        build_functions_pool(df_pairs, df_pools)
    except ValueError as e:
        assert "test-проектов" in str(e)
    else:
        raise AssertionError("Ожидалась ошибка утечки test-проекта")


def test_dedup_by_function_hash():
    src = _src()
    df_pairs = _pairs(
        [
            {"project": "p1", "split_pool": "train_pool", "function_hash": "h1", "_aux_src_old": src},
            {"project": "p1", "split_pool": "train_pool", "function_hash": "h1", "_aux_src_old": src},
        ]
    )
    df_pools = pd.DataFrame([{"project": "p1", "pool": "train_pool"}])
    df_pool, stats = build_functions_pool(df_pairs, df_pools)
    assert stats["дубликаты_hash"] == 1
    assert len(df_pool) == 1


def test_r1_filters_private_short_doc():
    df_pairs = _pairs(
        [
            {"project": "p1", "split_pool": "train_pool", "function_hash": "h1", "_aux_src_old": _src(name="_priv")},
            {"project": "p1", "split_pool": "train_pool", "function_hash": "h2", "_aux_src_old": _src(doc_len=5)},
            {"project": "p1", "split_pool": "train_pool", "function_hash": "h3", "_aux_src_old": _src()},
        ]
    )
    df_pools = pd.DataFrame([{"project": "p1", "pool": "train_pool"}])
    df_pool, stats = build_functions_pool(df_pairs, df_pools)
    assert stats["не_прошёл_фильтры"] == 2
    assert len(df_pool) == 1
    assert list(df_pool.columns) == ["project", "src", "function_hash"]


def test_only_train_pool_taken():
    df_pairs = _pairs(
        [
            {"project": "p1", "split_pool": "train_pool", "function_hash": "h1", "_aux_src_old": _src()},
            {"project": "p2", "split_pool": "test_pool", "function_hash": "h2", "_aux_src_old": _src()},
        ]
    )
    df_pools = pd.DataFrame(
        [{"project": "p1", "pool": "train_pool"}, {"project": "p2", "pool": "test_pool"}]
    )
    df_pool, stats = build_functions_pool(df_pairs, df_pools)
    assert stats["вход_train_rows"] == 1
    assert set(df_pool["project"]) == {"p1"}


def _neg(project="p1", src=None, doc_changed=False):
    return {"project": project, "src_old": src or _src(), "doc_changed": doc_changed}


def _pools_v1():
    df_v1 = pd.DataFrame(
        [{"project": "p1", "src": _src(), "function_hash": "hv1"}]
    )
    df_pools = pd.DataFrame(
        [{"project": "p1", "pool": "train_pool"}, {"project": "p2", "pool": "test_pool"}]
    )
    return df_v1, df_pools


def test_v2_only_train_and_no_doc_changed():
    df_v1, df_pools = _pools_v1()
    df_neg = _pairs(
        [
            _neg("p1"),
            _neg("p2"),
            {"project": "p1", "src_old": _src(), "doc_changed": True},
        ]
    )
    df_v2, stats = build_functions_pool_v2(df_v1, df_neg, df_pools)
    assert stats["neg_не_train_pool"] == 1
    assert stats["neg_doc_changed"] == 1
    assert stats["источник2_успех"] == 1
    assert len(df_v2) == len(df_v1) + 1
    assert set(df_v2["project"]) == {"p1"}


def test_v2_dedup_inside_and_against_v1():
    df_v1, df_pools = _pools_v1()
    src_dup = _src()
    fn = parse_function_ast(src_dup)
    assert fn is not None
    _, fn_no_doc = extract_docstring_and_clean_node(fn)
    h_dup = make_function_hash(normalize_code_from_node(fn_no_doc))
    df_v1 = pd.DataFrame([{"project": "p1", "src": src_dup, "function_hash": h_dup}])
    df_neg = _pairs([_neg("p1", src_dup), _neg("p1", src_dup)])
    df_v2, stats = build_functions_pool_v2(df_v1, df_neg, df_pools)
    assert stats["дубликаты_против_v1"] == 2
    assert stats["дубликаты_hash_внутри_источника2"] == 0
    assert len(df_v2) == 1

    df_v1b = pd.DataFrame(
        [{"project": "p1", "src": _src(name="fetch"), "function_hash": "other"}]
    )
    df_neg2 = _pairs([_neg("p1", src_dup), _neg("p1", src_dup)])
    _, stats2 = build_functions_pool_v2(df_v1b, df_neg2, df_pools)
    assert stats2["дубликаты_hash_внутри_источника2"] == 1
    assert stats2["источник2_успех"] == 1


def test_v2_r1_filters():
    df_v1, df_pools = _pools_v1()
    df_neg = _pairs(
        [
            _neg("p1", _src(name="_priv")),
            _neg("p1", _src(doc_len=5)),
            _neg("p1", _src()),
        ]
    )
    df_v2, stats = build_functions_pool_v2(df_v1, df_neg, df_pools)
    assert stats["не_прошёл_фильтры"] == 2
    assert stats["источник2_успех"] == 1


def test_v2_function_hash_matches_r2():
    df_v1, df_pools = _pools_v1()
    src = _src()
    df_neg = _pairs([_neg("p1", src)])
    df_v2, _ = build_functions_pool_v2(df_v1, df_neg, df_pools)
    fn = parse_function_ast(src)
    assert fn is not None
    _, fn_no_doc = extract_docstring_and_clean_node(fn)
    expected = make_function_hash(normalize_code_from_node(fn_no_doc))
    got = df_v2[df_v2["function_hash"] == expected]
    assert len(got) == 1
    assert list(df_v2.columns) == ["project", "src", "function_hash"]


def test_v2_final_leak_raises():
    df_v1 = pd.DataFrame(
        [{"project": "p2", "src": _src(), "function_hash": "hv1"}]
    )
    df_pools = pd.DataFrame(
        [{"project": "p1", "pool": "train_pool"}, {"project": "p2", "pool": "test_pool"}]
    )
    df_neg = _pairs([_neg("p1")])
    try:
        build_functions_pool_v2(df_v1, df_neg, df_pools)
    except ValueError as e:
        assert "test-проекты" in str(e)
    else:
        raise AssertionError("Ожидалась ошибка утечки test-проекта в v2")


def test_v2_excludes_test_functions_by_default():
    df_v1, df_pools = _pools_v1()
    df_neg = _pairs(
        [
            _neg("p1", _src(name="test_something")),
            _neg("p1", _src()),
        ]
    )
    df_v2, stats = build_functions_pool_v2(df_v1, df_neg, df_pools)
    assert stats["отсев_test_функции"] == 1
    assert stats["источник2_успех"] == 1
    assert not df_v2["src"].str.contains("def test_").any()

    df_v2b, statsb = build_functions_pool_v2(
        df_v1, df_neg, df_pools, exclude_tests=False
    )
    assert statsb["отсев_test_функции"] == 0
    assert statsb["источник2_успех"] == 2
