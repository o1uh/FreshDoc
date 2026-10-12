import json
from pathlib import Path

import pandas as pd

from src.synth.llm_batch import export_p3_batches, import_p3_responses, render_batch_task
from src.synth.llm_validators import must_keep_identifiers, validate_p3

ORIG = (
    "Add a and b. Default of b is 1. Result is their sum. "
    "Uses helper calc_total with timeout_seconds=30. Returns None on empty input."
)


def _para_ok() -> str:
    return (
        "Compute the sum of a and b (b defaults to 1). "
        "Delegates to helper calc_total with timeout_seconds=30. "
        "Gives None back when the input is empty."
    )


def test_p3_valid_accepted():
    ok, reason, parts = validate_p3(ORIG, {"docstring": _para_ok(), "style": "plain"})
    assert ok and reason == "ok"
    assert parts is not None and parts["style"] == "plain"


def test_p3_reject_length_ratio():
    ok, reason, _ = validate_p3(ORIG, {"docstring": "Sum a and b.", "style": "plain"})
    assert not ok and reason == "length_ratio"


def test_p3_reject_numbers_changed():
    para = _para_ok().replace("timeout_seconds=30", "timeout_seconds=60")
    ok, reason, _ = validate_p3(ORIG, {"docstring": para, "style": "plain"})
    assert not ok and reason == "numbers_changed"


def test_p3_reject_identifier_lost():
    para = _para_ok().replace("calc_total", "helper")
    ok, reason, _ = validate_p3(ORIG, {"docstring": para, "style": "plain"})
    assert not ok and reason.startswith("identifier_lost")


def test_p3_reject_unchanged():
    ok, reason, _ = validate_p3(ORIG, {"docstring": ORIG, "style": "plain"})
    assert not ok and reason == "docstring_unchanged"


def test_must_keep_identifiers():
    keep = must_keep_identifiers(ORIG)
    assert "calc_total" in keep and "timeout_seconds" in keep
    assert "and" not in keep


def test_p3_batch_render_and_roundtrip(tmp_path):
    prompts_dir = Path("prompts")
    rendered = render_batch_task((prompts_dir / "p3_paraphrase.txt").read_text(encoding="utf-8"), 2)
    assert "{function_source}" not in rendered and "skip" in rendered
    src = (
        "def add(a, b=1):\n"
        '    """Add a and b. Default of b is 1. Result is their sum."""\n'
        "    return a + b"
    )
    pool = pd.DataFrame([{"project": "p", "src": src, "function_hash": f"h{i}"} for i in range(4)])
    pool_path = tmp_path / "pool.parquet"
    pool.to_parquet(pool_path, index=False)
    p3_dir = tmp_path / "p3"
    info = export_p3_batches(pool, p3_dir, prompts_dir, n=4, batch_size=2, seed=42)
    assert info["n_functions"] == 4 and len(info["prompts"]) == 2
    manifest = pd.read_csv(p3_dir / "p3_manifest.csv", dtype=str)
    for prompt_name in info["prompts"]:
        prompt_path = p3_dir / prompt_name
        ids = manifest["pilot_id"].tolist()
        chunk_ids = ids[:2] if "batch1" in prompt_name else ids[2:]
        objects = [
            {"id": pid, "docstring": "Compute the sum of a and b (b defaults to 1). Outcome is their total.", "style": "plain"}
            for pid in chunk_ids
        ]
        prompt_path.with_suffix("").with_suffix(".response.txt").write_text(json.dumps(objects), encoding="utf-8")
    out_parquet = tmp_path / "docstring_paraphrases.parquet"
    stats = import_p3_responses(pool_path, p3_dir, out_parquet)
    assert stats["accepted"] == 4
    df = pd.read_parquet(out_parquet)
    assert list(df.columns) == ["function_hash", "doc_orig", "doc_para", "style"]
    assert len(df) == 4
