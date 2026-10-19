"""FD-22: нулевой LLM-судья M3 (R10). Общие примитивы: точный одиночный
промпт R10, разбор JSON-ответов, валидаторы, p_stale из confidence.

Судья — Claude Haiku 5.5 через Go Inference API opencode
(см. src/synth/opencode_client.py). Выбор: Gemini 3.8 Flash рассматривался
и отклонён — он размечал все 200 строк gold-кандидатов (reports/gold_report.md),
судить по ним же — петля самооценки; Muse Spark отклонён по той же причине
(60 unlabeled + eval-аудит + пилот). Haiku в проекте ранее не использовался
(ноль пересечений), от генератора синтетики DeepSeek отличается (R7 соблюдено).

p_stale: из логарифмов вероятностей при наличии API; в Messages API их нет —
используется поле confidence (R10 fallback).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROMPT_NAME = "m3_judge.txt"

#: Допустимые значения drift_type судьи (R10: none + 6 сигнатурных +
#: behavior_changed (LLM P1) + unknown для неясных случаев).
ALLOWED_DRIFT = frozenset({
    "none",
    "param_renamed",
    "param_removed",
    "param_added",
    "default_changed",
    "return_changed",
    "exception_changed",
    "behavior_changed",
    "unknown",
})

#: Судья M3 (FD-22). Генератор синтетики — DeepSeek, совпадение запрещено (R7).
#: Gemini/Spark отклонены: участвовали в разметке gold (см. модуль docstring).
JUDGE_MODEL = "claude-haiku-5-5"
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 120


def render_single(diff_code: str, doc_old: str, prompts_dir: Path | str = "prompts") -> str:
    """Одиночный промпт R10 с подставленными diff и docstring."""
    template = (Path(prompts_dir) / PROMPT_NAME).read_text(encoding="utf-8")
    if "{diff_code}" not in template or "{doc_old}" not in template:
        raise ValueError("В prompts/m3_judge.txt нет плейсхолдеров {diff_code}/{doc_old}")
    return template.replace("{diff_code}", diff_code or "").replace("{doc_old}", doc_old or "")


def extract_json_slice(text: str) -> str:
    cleaned = (text or "").strip()
    blocks = re.findall(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    candidates: list[str] = []
    for block in blocks:
        block = block.strip()
        if block.lower().startswith("json"):
            block = block[4:].strip()
        candidates.append(block)
    candidates.append(cleaned)
    for cand in candidates:
        starts = [i for i in (cand.find("{"), cand.find("[")) if i >= 0]
        if not starts:
            continue
        start = min(starts)
        closer = "}" if cand[start] == "{" else "]"
        end = cand.rfind(closer)
        if end <= start:
            continue
        snippet = cand[start:end + 1]
        try:
            json.loads(snippet)
            return snippet
        except json.JSONDecodeError:
            continue
    return cleaned


def parse_batch_response(text: str) -> tuple[list[dict[str, Any]], str | None]:
    """Разбор ответа чата: JSON-массив (или одиночный объект) с полями M3."""
    try:
        payload = json.loads(extract_json_slice(text))
    except json.JSONDecodeError as exc:
        return [], f"bad_json: {exc}"
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return [], "bad_json: корень не object/array"
    objects: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            return [], "bad_json: элемент массива не object"
        objects.append(item)
    return objects, None


def validate_judge_object(obj: dict[str, Any]) -> tuple[bool, str, dict[str, Any] | None]:
    """Проверка одного объекта ответа судьи. Возвращает (ok, reason, parts)."""
    if not isinstance(obj, dict):
        return False, "bad_object", None
    if obj.get("skip") is True:
        return False, "skipped", None
    if "id" not in obj:
        return False, "missing_id", None
    if "stale" not in obj:
        return False, "missing_stale", None
    stale = obj["stale"]
    if not isinstance(stale, bool):
        return False, "bad_stale", None
    if "confidence" not in obj:
        return False, "missing_confidence", None
    try:
        conf = float(obj["confidence"])
    except (TypeError, ValueError):
        return False, "bad_confidence", None
    if not (0.0 <= conf <= 1.0):
        return False, "bad_confidence", None
    drift = obj.get("drift_type", "unknown")
    if not isinstance(drift, str) or drift not in ALLOWED_DRIFT:
        return False, f"bad_drift_type:{drift!r}", None
    reason = obj.get("reason", "")
    if not isinstance(reason, str) or not reason.strip():
        return False, "missing_reason", None
    p_stale = conf if stale else 1.0 - conf
    parts = {
        "m3_id": str(obj["id"]),
        "p_stale": float(p_stale),
        "type_pred": str(drift),
        "stale": bool(stale),
        "confidence": float(conf),
        "reason": reason.strip(),
    }
    return True, "ok", parts
