from __future__ import annotations

import json
import re
from typing import Any

from src.data.drift import derive_drift_type, extract_signature
from src.data.normalizer import (
    compute_code_diff,
    extract_docstring_and_clean_node,
    normalize_code_from_node,
    parse_function_ast,
)

MAX_DIFF_LINES = 8


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


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
        snippet = cand[start : end + 1]
        try:
            json.loads(snippet)
            return snippet
        except json.JSONDecodeError:
            continue
    return cleaned


def parse_batch_response(text: str) -> tuple[list[dict[str, Any]], str | None]:
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


def _count_diff_lines(diff: str) -> int:
    total = 0
    for line in (diff or "").splitlines():
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith(("+", "-")):
            total += 1
    return total


def _check_structure(src_old: str, new_src: Any) -> tuple[bool, str, dict[str, str] | None]:
    if not isinstance(new_src, str) or not new_src.strip():
        return False, "empty_new_source", None
    old_fn = parse_function_ast(src_old)
    if old_fn is None:
        return False, "bad_old_source", None
    new_fn = parse_function_ast(new_src)
    if new_fn is None:
        return False, "bad_syntax", None
    old_doc, old_no_doc = extract_docstring_and_clean_node(old_fn)
    new_doc, new_no_doc = extract_docstring_and_clean_node(new_fn)
    if normalize_ws(new_doc or "") != normalize_ws(old_doc or ""):
        return False, "docstring_changed", None
    try:
        sig_old = extract_signature(old_no_doc)
        sig_new = extract_signature(new_no_doc)
    except Exception:
        return False, "bad_signature", None
    if sig_old != sig_new:
        return False, "signature_changed", None
    try:
        code_old = normalize_code_from_node(old_no_doc)
        code_new = normalize_code_from_node(new_no_doc)
    except Exception:
        return False, "bad_normalize", None
    if code_new == code_old:
        return False, "code_unchanged", None
    diff = compute_code_diff(code_old, code_new)
    if _count_diff_lines(diff) > MAX_DIFF_LINES:
        return False, "diff_too_big", None
    if derive_drift_type(sig_old, sig_new) != "none":
        return False, "signature_changed", None
    return True, "ok", {
        "code_old": code_old,
        "code_new": code_new,
        "diff": diff,
        "doc_old": (old_doc or "").strip(),
    }


def validate_p1(src_old: str, obj: dict[str, Any]) -> tuple[bool, str, dict[str, str] | None]:
    if not isinstance(obj, dict):
        return False, "bad_object", None
    if obj.get("skip") is True:
        return False, "skipped", None
    ok, reason, parts = _check_structure(src_old, obj.get("new_function_source"))
    if not ok:
        return False, reason, None
    assert parts is not None
    stmts = obj.get("stale_statements")
    if not isinstance(stmts, list) or not stmts:
        return False, "empty_stale_statements", None
    doc_norm = normalize_ws(parts["doc_old"])
    for stmt in stmts:
        if not isinstance(stmt, str) or not stmt.strip():
            return False, "bad_stale_statement", None
        if normalize_ws(stmt) not in doc_norm:
            return False, "stale_statement_not_found", None
    return True, "ok", parts


def validate_p2(src_old: str, obj: dict[str, Any]) -> tuple[bool, str, dict[str, str] | None]:
    if not isinstance(obj, dict):
        return False, "bad_object", None
    if obj.get("skip") is True:
        return False, "skipped", None
    ok, reason, parts = _check_structure(src_old, obj.get("new_function_source"))
    if not ok:
        return False, reason, None
    if obj.get("docstring_still_true") is not True:
        return False, "docstring_flag_not_true", None
    return True, "ok", parts


_NUMBER_RE = re.compile(r"\d[\d_]*(?:\.\d+)?")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LITERAL_KEEP = frozenset({"True", "False", "None"})
P3_MIN_RATIO = 0.6
P3_MAX_RATIO = 1.6


def must_keep_identifiers(doc_orig: str) -> list[str]:
    keep: set[str] = set()
    for tok in _IDENT_RE.findall(doc_orig or ""):
        if "_" in tok or tok.isupper() or any(c.isdigit() for c in tok) or tok in _LITERAL_KEEP:
            keep.add(tok)
    return sorted(keep)


def validate_p3(doc_orig: str, obj: dict[str, Any]) -> tuple[bool, str, dict[str, str] | None]:
    if not isinstance(obj, dict):
        return False, "bad_object", None
    if obj.get("skip") is True:
        return False, "skipped", None
    para = obj.get("docstring")
    if not isinstance(para, str) or not para.strip():
        return False, "empty_docstring", None
    para = para.strip()
    orig = (doc_orig or "").strip()
    if not orig:
        return False, "empty_orig", None
    if normalize_ws(para) == normalize_ws(orig):
        return False, "docstring_unchanged", None
    ratio = len(para) / len(orig)
    if not (P3_MIN_RATIO <= ratio <= P3_MAX_RATIO):
        return False, "length_ratio", None
    if sorted(_NUMBER_RE.findall(orig)) != sorted(_NUMBER_RE.findall(para)):
        return False, "numbers_changed", None
    for tok in must_keep_identifiers(orig):
        if tok not in para:
            return False, f"identifier_lost:{tok}", None
    return True, "ok", {"doc_para": para, "style": str(obj.get("style") or "unknown")}
