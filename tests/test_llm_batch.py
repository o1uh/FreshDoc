import json
from pathlib import Path

import pandas as pd

from src.synth.llm_batch import (
    MANIFEST_NAME,
    export_pilot_batches,
    import_pilot_responses,
    render_batch_task,
    sample_pilot_functions,
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
        project = f"proj{i % 3}"
        rows.append({"project": project, "src": _tiny_src(f"func{i}"), "function_hash": f"h{i:04d}"})
    return pd.DataFrame(rows)


def test_sample_uniform_and_deterministic():
    df = _tiny_pool(12)
    first = sample_pilot_functions(df, 12, 42)
    second = sample_pilot_functions(df, 12, 42)
    assert first["pilot_id"].tolist() == second["pilot_id"].tolist()
    assert first["pilot_id"].tolist() == [f"P{i + 1:04d}" for i in range(12)]
    counts = first["project"].value_counts()
    assert set(counts.index) == {"proj0", "proj1", "proj2"}
    assert max(counts) - min(counts) <= 1


def test_render_batch_task_replaces_markers():
    prompts_dir = Path("prompts")
    single = (prompts_dir / "p1_behavior.txt").read_text(encoding="utf-8")
    rendered = render_batch_task(single, 6)
    assert "{function_source}" not in rendered
    assert "<function>" not in rendered
    assert "skip" in rendered
    assert "JSON array" in rendered


def _p1_edit(src: str) -> str:
    return src.replace("return a + b", "return a - b")


def _p2_edit(src: str) -> str:
    return src.replace("return a + b", "res = a + b\n    return res")


def _stale_of(src: str) -> str:
    for line in src.splitlines():
        if "Add a and b" in line:
            return "Add a and b."
    raise AssertionError("no stale candidate")


def test_export_import_roundtrip(tmp_path):
    prompts_dir = Path("prompts")
    pool_path = tmp_path / "pool.parquet"
    _tiny_pool(12).to_parquet(pool_path, index=False)
    pilot_dir = tmp_path / "pilot"
    info = export_pilot_batches(pd.read_parquet(pool_path), pilot_dir, prompts_dir, seed=42)
    assert info["n_functions"] == 12
    assert len(info["prompts"]) == 4
    assert (pilot_dir / MANIFEST_NAME).exists()
    manifest = pd.read_csv(pilot_dir / MANIFEST_NAME, dtype=str)
    by_id = {r["pilot_id"]: r for _, r in manifest.iterrows()}
    pool = pd.read_parquet(pool_path)
    src_by_hash = {r["function_hash"]: r["src"] for _, r in pool.iterrows()}
    for prompt_name in info["prompts"]:
        prompt_path = pilot_dir / prompt_name
        assert prompt_path.exists()
        objects = []
        for _, row in manifest.iterrows():
            src = src_by_hash[row["function_hash"]]
            if "_p1_" in prompt_name:
                objects.append(
                    {
                        "id": row["pilot_id"],
                        "new_function_source": _p1_edit(src),
                        "change_summary": "changed op",
                        "stale_statements": [_stale_of(src)],
                        "why_stale": "subtraction now",
                    }
                )
            else:
                objects.append(
                    {
                        "id": row["pilot_id"],
                        "new_function_source": _p2_edit(src),
                        "change_summary": "temp var",
                        "docstring_still_true": True,
                    }
                )
            if len(objects) == 6:
                break
        response_path = prompt_path.with_suffix("").with_suffix(".response.txt")
        response_path.write_text("```json\n" + json.dumps(objects) + "\n```", encoding="utf-8")
    report_path = tmp_path / "llm_pilot.md"
    out_parquet = tmp_path / "synth_llm_pilot.parquet"
    stats = import_pilot_responses(pool_path, pilot_dir, report_path, out_parquet)
    assert stats["accepted_p1"] == 12
    assert stats["accepted_p2"] == 12
    assert stats["accept_rate"] == 1.0
    assert report_path.exists()
    df_out = pd.read_parquet(out_parquet)
    assert len(df_out) == 24
    assert set(df_out["label_source"]) == {"llm_p1", "llm_p2"}


def test_import_handles_skip_and_unknown_id(tmp_path):
    prompts_dir = Path("prompts")
    pool_path = tmp_path / "pool.parquet"
    _tiny_pool(12).to_parquet(pool_path, index=False)
    pilot_dir = tmp_path / "pilot"
    export_pilot_batches(pd.read_parquet(pool_path), pilot_dir, prompts_dir, seed=42)
    prompt_path = pilot_dir / "pilot_p1_batch1.prompt.txt"
    payload = [
        {"id": "P0001", "skip": True},
        {"id": "NOPE", "skip": True},
    ]
    prompt_path.with_suffix("").with_suffix(".response.txt").write_text(json.dumps(payload), encoding="utf-8")
    stats = import_pilot_responses(pool_path, pilot_dir, tmp_path / "r.md", tmp_path / "o.parquet")
    assert stats["skipped"] == 1
    assert stats["unknown_id"] == 1
