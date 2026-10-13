import json
from pathlib import Path

import pandas as pd
import pytest

from src.synth.deepseek_client import (
    DeepSeekParams,
    build_messages,
    cost_table,
    cost_usd,
    load_api_key,
    request_key,
)
from scripts.llm_main_batch import (
    cmd_audit_sample,
    cmd_export,
    cmd_import,
    sample_main_functions,
)


def _tiny_src(name: str, default_b: int = 1) -> str:
    return (
        f"def {name}(a, b={default_b}):\n"
        f'    """Add a and b. Default of b is {default_b}. Result is their sum."""\n'
        "    return a + b"
    )


def _tiny_pool(n: int = 12) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({"project": f"proj{i % 3}", "src": _tiny_src(f"func{i}", i + 1),
                     "function_hash": f"h{i:04d}"})
    return pd.DataFrame(rows)


def test_sample_deterministic_uniform_and_excludes_pilot():
    df = _tiny_pool(12)
    first = sample_main_functions(df, 9, seed=11, exclude_hashes={"h0000", "h0001", "h0002"})
    second = sample_main_functions(df, 9, seed=11, exclude_hashes={"h0000", "h0001", "h0002"})
    assert first["m_id"].tolist() == second["m_id"].tolist()
    assert first["m_id"].tolist() == [f"M{i + 1:04d}" for i in range(9)]
    assert not set(first["function_hash"]) & {"h0000", "h0001", "h0002"}
    counts = first["project"].value_counts()
    assert max(counts) - min(counts) <= 1


def test_sample_each_function_used_at_most_twice():
    # P1+P2 на функцию = 2 использования; повторная выборка тех же функций запрещена исключением
    df = _tiny_pool(6)
    sample = sample_main_functions(df, 6, seed=7)
    assert sample["function_hash"].nunique() == 6


def test_build_messages_substitutes_function_and_mentions_json(tmp_path):
    prompts_dir = Path("prompts")
    src = _tiny_src("demo")
    for task in ("p1", "p2"):
        messages = build_messages(task, src, prompts_dir)
        assert [m["role"] for m in messages] == ["system", "user"]
        assert "def demo" in messages[1]["content"]
        assert "{function_source}" not in messages[1]["content"]
        assert "json" in (messages[0]["content"] + messages[1]["content"]).lower()


def test_build_messages_rejects_p3(tmp_path):
    with pytest.raises(ValueError):
        build_messages("p3", _tiny_src("demo"), Path("prompts"))


def test_request_key_stable_and_task_sensitive():
    prompts_dir = Path("prompts")
    src = _tiny_src("demo")
    params = DeepSeekParams()
    m1 = build_messages("p1", src, prompts_dir)
    assert request_key("deepseek-flash", params, m1) == request_key("deepseek-flash", params, m1)
    m2 = build_messages("p2", src, prompts_dir)
    assert request_key("deepseek-flash", params, m1) != request_key("deepseek-flash", params, m2)


def test_cost_math_peak_is_double_offpeak():
    assert cost_usd(600, 550, 1200, peak=True) == pytest.approx(2 * cost_usd(600, 550, 1200, peak=False))
    rows = cost_table([600])
    assert rows[0]["usd_peak"] == pytest.approx(0.389, abs=0.01)
    assert rows[0]["usd_offpeak"] == pytest.approx(0.194, abs=0.01)


def test_load_api_key_from_env_and_dotenv(tmp_path):
    assert load_api_key(env={"DEEPSEEK_API_KEY": " sk-123 "}) == "sk-123"
    dotenv = tmp_path / ".env"
    dotenv.write_text("OTHER=1\nDEEPSEEK_API_KEY=sk-dotenv\n", encoding="utf-8")
    assert load_api_key(env={}, dotenv_path=dotenv) == "sk-dotenv"
    with pytest.raises(RuntimeError):
        load_api_key(env={}, dotenv_path=tmp_path / "missing.env")


def _ns(**kwargs):
    from types import SimpleNamespace
    return SimpleNamespace(**kwargs)


def test_export_import_roundtrip_single_object(tmp_path):
    pool_path = tmp_path / "pool.parquet"
    _tiny_pool(12).to_parquet(pool_path, index=False)
    batch_dir = tmp_path / "batch"
    cmd_export(_ns(n=4, tag="t", dir=str(batch_dir), pool=str(pool_path),
                   prompts="prompts", seed=11, include_pilot=True))
    import csv as _csv
    manifest = list(_csv.DictReader(open(batch_dir / "main_manifest.csv", encoding="utf-8")))
    assert len(manifest) == 4
    req_files = sorted((batch_dir / "requests").glob("M*_p*.json"))
    assert len(req_files) == 8
    # поддельные ответы API: одиночный JSON-объект (не массив)
    pool = pd.read_parquet(pool_path)
    src_by_hash = {r["function_hash"]: r["src"] for _, r in pool.iterrows()}

    def stale_of(src: str) -> str:
        for line in src.splitlines():
            if "Add a and b" in line:
                return "Add a and b."
        raise AssertionError

    for req_path in req_files:
        req = json.loads(req_path.read_text(encoding="utf-8"))
        src = src_by_hash[req["function_hash"]]
        if req["task"] == "p1":
            obj = {"new_function_source": src.replace("return a + b", "return a - b"),
                   "change_summary": "changed op", "stale_statements": [stale_of(src)],
                   "why_stale": "subtraction now"}
        else:
            obj = {"new_function_source": src.replace("return a + b", "res = a + b\n    return res"),
                   "change_summary": "temp var", "docstring_still_true": True}
        (batch_dir / "responses").mkdir(exist_ok=True)
        (batch_dir / "responses" / f"{req_path.stem}.response.json").write_text(
            json.dumps({"m_id": req["m_id"], "task": req["task"], "content": json.dumps(obj),
                        "usage": {"prompt_tokens": 500, "prompt_cache_hit_tokens": 0,
                                  "prompt_cache_miss_tokens": 500, "completion_tokens": 900,
                                  "reasoning_tokens": 600, "total_tokens": 1400}}),
            encoding="utf-8")
    report = tmp_path / "r.md"
    out_parquet = tmp_path / "o.parquet"
    cmd_import(_ns(dir=str(batch_dir), pool=str(pool_path), report=str(report),
                   out=str(out_parquet), n=4))
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "deepseek-flash" in text
    df_out = pd.read_parquet(out_parquet)
    assert len(df_out) == 8
    assert set(df_out["label_source"]) == {"llm_p1", "llm_p2"}


def test_audit_sample_stratified(tmp_path):
    rows = []
    for i in range(30):
        rows.append({"pair_id": f"p1_{i}", "label_source": "llm_p1", "project": "a",
                     "symbol": "f", "sig_old": "def f():", "diff_code": "-x\n+y", "doc_old": "Doc."})
    for i in range(30):
        rows.append({"pair_id": f"p2_{i}", "label_source": "llm_p2", "project": "a",
                     "symbol": "f", "sig_old": "def f():", "diff_code": "-x\n+y", "doc_old": "Doc."})
    in_path = tmp_path / "in.parquet"
    pd.DataFrame(rows).to_parquet(in_path, index=False)
    out = tmp_path / "audit.csv"
    cmd_audit_sample(_ns(in_path=str(in_path), out=str(out), seed=11))
    df = pd.read_csv(out, dtype=str)
    assert len(df) == 50
    assert (df["label_source"] == "llm_p1").sum() == 25
    assert (df["label_source"] == "llm_p2").sum() == 25
    assert "verdict" in df.columns
