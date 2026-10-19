"""FD-22: тесты нулевого судьи M3 (промпт R10, разбор, валидаторы)."""

from pathlib import Path

import pytest

from src.models.m3_judge import (
    ALLOWED_DRIFT,
    JUDGE_MODEL,
    parse_batch_response,
    render_single,
    validate_judge_object,
)


def test_single_prompt_placeholders(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "m3_judge.txt").write_text("A<diff>{diff_code}</diff>B<docstring>{doc_old}</docstring>C",
                                    encoding="utf-8")
    out = render_single("DIFF", "DOC", prompts_dir=d)
    assert out == "A<diff>DIFF</diff>B<docstring>DOC</docstring>C"
    bad = tmp_path / "bad_prompts"
    bad.mkdir()
    (bad / "m3_judge.txt").write_text("без плейсхолдеров", encoding="utf-8")
    with pytest.raises(ValueError):
        render_single("x", "y", prompts_dir=bad)


def test_parse_fenced_array_and_single_object():
    text = 'вот:\n```json\n[{"id": "j00001", "skip": true}]\n```\nконец'
    objects, error = parse_batch_response(text)
    assert error is None and objects == [{"id": "j00001", "skip": True}]
    objects, error = parse_batch_response('{"id": "j1", "stale": true, "confidence": 0.9, '
                                          '"drift_type": "param_added", "reason": "r"}')
    assert error is None and len(objects) == 1


def test_parse_bad_json():
    objects, error = parse_batch_response("это вообще не json {{{")
    assert objects == [] and error is not None


def test_validate_true_maps_confidence():
    ok, reason, parts = validate_judge_object(
        {"id": "j00001", "stale": True, "confidence": 0.8,
         "drift_type": "param_added", "reason": "param renamed"})
    assert ok and reason == "ok"
    assert parts is not None and parts["p_stale"] == 0.8
    assert parts["type_pred"] == "param_added"


def test_validate_false_inverts_confidence():
    ok, _, parts = validate_judge_object(
        {"id": "j2", "stale": False, "confidence": 0.8,
         "drift_type": "none", "reason": "docs still true"})
    assert ok and parts is not None and abs(parts["p_stale"] - 0.2) < 1e-12


def test_validate_rejects_bad_fields():
    bad_conf = {"id": "j1", "stale": True, "confidence": 1.5,
                "drift_type": "none", "reason": "r"}
    ok, reason, _ = validate_judge_object(bad_conf)
    assert not ok and reason == "bad_confidence"
    bad_drift = {"id": "j1", "stale": True, "confidence": 0.9,
                 "drift_type": "changed_everything", "reason": "r"}
    ok, reason, _ = validate_judge_object(bad_drift)
    assert not ok and reason.startswith("bad_drift_type")
    bad_stale = {"id": "j1", "stale": "yes", "confidence": 0.9,
                 "drift_type": "none", "reason": "r"}
    ok, reason, _ = validate_judge_object(bad_stale)
    assert not ok and reason == "bad_stale"
    skip = {"id": "j1", "skip": True}
    ok, reason, _ = validate_judge_object(skip)
    assert not ok and reason == "skipped"


def test_allowed_drift_covers_r10():
    assert {"none", "param_renamed", "param_removed", "param_added", "default_changed",
            "return_changed", "exception_changed", "behavior_changed", "unknown"} <= ALLOWED_DRIFT


def test_judge_differs_from_generator():
    # R7: судья M3 обязан отличаться от генератора синтетики (deepseek).
    # Gemini/Spark отклонены отдельно: участвовали в разметке gold.
    assert "haiku" in JUDGE_MODEL.lower()
    assert "deepseek" not in JUDGE_MODEL.lower()
    assert "gemini" not in JUDGE_MODEL.lower()
    assert "spark" not in JUDGE_MODEL.lower()


def test_real_prompt_template_r10(tmp_path=None):
    repo_prompts = Path(__file__).resolve().parent.parent / "prompts" / "m3_judge.txt"
    text = repo_prompts.read_text(encoding="utf-8")
    assert "{diff_code}" in text and "{doc_old}" in text
    assert "confidence" in text and "drift_type" in text
    assert "temperature" not in text  # параметры — в config/m3_judge.yaml, не в промпте
