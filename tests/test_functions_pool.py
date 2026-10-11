import pandas as pd

from src.synth.pool import build_functions_pool


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
