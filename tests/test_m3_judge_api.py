"""FD-22 (API-вариант): тесты клиента Inference API и команд run/import-api.

Сеть мокается (monkeypatch requests.post); реального ключа не нужно.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.synth import opencode_client as oc
from src.synth.opencode_client import OpenCodeParams


def _params() -> OpenCodeParams:
    return OpenCodeParams(model="claude-haiku-5-5", temperature=0.0, max_tokens=120)


def _msg_ok(text: str = '{"stale": true, "confidence": 0.8, "drift_type": "param_added", "reason": "r"}'):
    return {
        "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 410, "output_tokens": 60,
                  "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    }


class _Resp:
    def __init__(self, status: int, payload: dict | None = None, text: str = ""):
        self.status_code = status
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _patch_post(monkeypatch, responses: list):
    calls = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        item = responses[min(calls["n"] - 1, len(responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(oc.requests, "post", fake_post)
    monkeypatch.setattr(oc.time, "sleep", lambda *a, **k: None)
    return calls


def _messages():
    return [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]


def test_build_messages_splits_system_user(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "m3_judge.txt").write_text(
        "SYSTEM: be strict.\nUSER: Decide.\n<diff>{diff_code}</diff>\n<docstring>{doc_old}</docstring>",
        encoding="utf-8")
    msgs = oc.build_messages("D", "O", d)
    assert msgs[0] == {"role": "system", "content": "be strict."}
    assert "<diff>D</diff>" in msgs[1]["content"] and "<docstring>O</docstring>" in msgs[1]["content"]


def test_split_template_rejects_bad():
    with pytest.raises(ValueError):
        oc.split_template("no user section here")
    with pytest.raises(ValueError):
        oc.split_template("SYSTEM: s\nUSER: no placeholders")


def test_request_key_deterministic():
    m = _messages()
    assert oc.request_key("x", _params(), m) == oc.request_key("x", _params(), m)
    assert oc.request_key("x", _params(), m) != oc.request_key("y", _params(), m)


def test_call_success_and_cache(monkeypatch, tmp_path):
    calls = _patch_post(monkeypatch, [_Resp(200, _msg_ok())])
    out = oc.call_opencode(_messages(), "key", cache_dir=tmp_path, params=_params())
    assert out["content"].startswith('{"stale"')
    assert out["usage"]["prompt_tokens"] == 410 and out["usage"]["completion_tokens"] == 60
    assert out["cached"] is False and calls["n"] == 1
    # повтор — из кэша, без сети
    out2 = oc.call_opencode(_messages(), "key", cache_dir=tmp_path, params=_params())
    assert out2["cached"] is True and calls["n"] == 1


def test_call_retries_429_then_ok(monkeypatch, tmp_path):
    calls = _patch_post(monkeypatch, [_Resp(429, text="rate"), _Resp(200, _msg_ok())])
    out = oc.call_opencode(_messages(), "key", cache_dir=tmp_path, params=_params())
    assert out["attempts"] == 2 and calls["n"] == 2


def test_call_empty_content_then_ok(monkeypatch, tmp_path):
    calls = _patch_post(monkeypatch, [_Resp(200, _msg_ok("")), _Resp(200, _msg_ok())])
    out = oc.call_opencode(_messages(), "key", cache_dir=tmp_path, params=_params())
    assert out["attempts"] == 2 and calls["n"] == 2


def test_call_exhausted_raises(monkeypatch, tmp_path):
    _patch_post(monkeypatch, [_Resp(503, text="down")])
    with pytest.raises(RuntimeError, match="попыток исчерпаны"):
        oc.call_opencode(_messages(), "key", cache_dir=tmp_path, params=_params())


def test_call_http_error_raises_fast(monkeypatch, tmp_path):
    _patch_post(monkeypatch, [_Resp(401, text="bad key")])
    with pytest.raises(RuntimeError, match="HTTP 401"):
        oc.call_opencode(_messages(), "key", cache_dir=tmp_path, params=_params())


def test_load_api_key_env_and_missing(monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "sk-test")
    assert oc.load_api_key() == "sk-test"
    monkeypatch.delenv("OPENCODE_API_KEY")
    with pytest.raises(RuntimeError, match="Нет OPENCODE_API_KEY"):
        oc.load_api_key(env={}, dotenv_path=Path("/no/such/.env"))


def test_cost_math():
    # 500 запросов × (400×0.10 + 350×0.50)/1M = 0.1075
    assert oc.cost_usd(500) == pytest.approx(0.1075)
    assert oc.cost_usd(6) == pytest.approx(0.00129)


def test_normalize_usage_reads_thinking_tokens():
    raw = {"input_tokens": 439, "output_tokens": 1497,
           "output_tokens_details": {"thinking_tokens": 1383}}
    u = oc._normalize_usage(raw)
    assert u["reasoning_tokens"] == 1383
    assert u["completion_tokens"] == 1497
    assert oc._normalize_usage({})["reasoning_tokens"] == 0


def test_estimate_runs_without_net_or_key(capsys):
    import sys
    from scripts.m3_judge import cmd_estimate

    class A:
        config = "config/m3_judge.yaml"

    cmd_estimate(A())
    out = capsys.readouterr().out
    assert "Claude Haiku 5.5" in out and "500" in out


def _mini_api_dir(tmp_path: Path) -> Path:
    d = tmp_path / "m3"
    (d / "responses").mkdir(parents=True)
    pd.DataFrame([{"m3_id": "j00001", "pair_id": "p1", "split": "test_real"},
                  {"m3_id": "j00002", "pair_id": "p2", "split": "gold"}]).to_csv(
        d / "m3_manifest.csv", index=False)
    (d / "responses" / "m3_j00001.response.json").write_text(json.dumps({
        "m3_id": "j00001", "pair_id": "p1", "model": "claude-haiku-5-5",
        "content": '{"stale": true, "confidence": 0.8, "drift_type": "param_added", "reason": "renamed"}',
        "usage": {"prompt_tokens": 400, "completion_tokens": 50}}), encoding="utf-8")
    (d / "responses" / "m3_j00002.response.json").write_text(json.dumps({
        "m3_id": "j00002", "pair_id": "p2", "model": "claude-haiku-5-5",
        "content": '[{"id": "j00002", "skip": true}]',
        "usage": {"prompt_tokens": 400, "completion_tokens": 10}}), encoding="utf-8")
    return d


def test_import_api_maps_and_skips(tmp_path):
    from scripts.m3_judge import cmd_import_api

    d = _mini_api_dir(tmp_path)

    class A:
        dir = str(d)
        out = str(d / "m3.csv")
        report = str(d / "m3_judge.md")
        config = "config/m3_judge.yaml"

    cmd_import_api(A())
    pred = pd.read_csv(d / "m3.csv")
    assert len(pred) == 1 and pred.iloc[0]["pair_id"] == "p1"
    assert abs(pred.iloc[0]["p_stale"] - 0.8) < 1e-12
    assert pred.iloc[0]["type_pred"] == "param_added"
    rep = (d / "m3_judge.md").read_text(encoding="utf-8")
    assert "Claude Haiku 5.5" in rep and "skipped" in rep


def test_api_params_from_config():
    from scripts.m3_judge import _api_params

    p = _api_params({"api": {"model": "claude-haiku-5-5", "temperature": 0.0, "max_tokens": 120}})
    assert p.model == "claude-haiku-5-5" and p.temperature == 0.0 and p.max_tokens == 120
    p2 = _api_params({"api": {"model": "claude-haiku-5-5", "temperature": None, "max_tokens": 120}})
    assert p2.temperature is None


def test_temperature_none_omitted_from_body(monkeypatch, tmp_path):
    bodies: list[dict] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        bodies.append(json)
        return _Resp(200, _msg_ok())

    monkeypatch.setattr(oc.requests, "post", fake_post)
    monkeypatch.setattr(oc.time, "sleep", lambda *a, **k: None)
    oc.call_opencode(_messages(), "key", cache_dir=tmp_path,
                     params=OpenCodeParams(temperature=None))
    assert "temperature" not in bodies[0]
    assert bodies[0]["max_tokens"] == 1000
    assert "thinking" not in bodies[0] and "output_config" not in bodies[0]


def test_effort_max_sends_thinking_and_output_config(monkeypatch, tmp_path):
    bodies: list[dict] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        bodies.append(json)
        return _Resp(200, _msg_ok())

    monkeypatch.setattr(oc.requests, "post", fake_post)
    monkeypatch.setattr(oc.time, "sleep", lambda *a, **k: None)
    oc.call_opencode(_messages(), "key", cache_dir=tmp_path,
                     params=OpenCodeParams(temperature=None, max_tokens=4000, effort="max"))
    assert bodies[0]["thinking"] == {"type": "adaptive"}
    assert bodies[0]["output_config"] == {"effort": "max"}
    assert bodies[0]["max_tokens"] == 4000


def test_request_key_differs_by_effort():
    m = _messages()
    a = OpenCodeParams(temperature=None, max_tokens=1000, effort=None)
    b = OpenCodeParams(temperature=None, max_tokens=4000, effort="max")
    assert oc.request_key("x", a, m) != oc.request_key("x", b, m)
