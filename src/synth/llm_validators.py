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

# FD-11 fix (ревью P1, 56% шума): ужесточение validate_p1 без повторных API-вызовов.
# Три фильтра применяются ПОСЛЕ базовой проверки подстроки stale_statements:
#  1) thin_docstring — однострочные/короткие docstring не содержат проверяемого
#     поведения без смены сигнатуры (класс C по R9);
#  2) stale_covers_docstring — цитата покрывает почти весь короткий docstring
#     (признак галлюцинации: модель выдумывает деталь и цитирует общее описание);
#  3) stale_not_linked_to_diff — ни один значимый токен диффа нет в stale±1.
# Пороги подобраны на аудите 25 P1 (reports/audit_v1.md): thin OR даёт 154/580
# отказов, coverage 88/580; в сумме с лексикой ожидается 580→250–320 пар.
P1_MIN_DOC_CHARS = 80
P1_MIN_DOC_LINES = 3
P1_MAX_STALE_COVERAGE = 0.8
P1_COVERAGE_DOC_MAX = 150

_NUMBER_RE = re.compile(r"\d[\d_]*(?:\.\d+)?")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


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
    # --- FD-11: ужесточение P1 (без повторных API-вызовов) ---
    doc_old = parts["doc_old"]
    # 1) Тонкий docstring: нет спецификации поведения без смены сигнатуры.
    if len(doc_old.strip()) < P1_MIN_DOC_CHARS or (doc_old.strip().count("\n") + 1) < P1_MIN_DOC_LINES:
        return False, "thin_docstring", None
    # 2) Цитата покрывает почти весь короткий docstring (галлюцинация).
    doc_len = max(1, len(normalize_ws(doc_old)))
    max_stale = max(len(normalize_ws(s)) for s in stmts if isinstance(s, str))
    if len(doc_old.strip()) < P1_COVERAGE_DOC_MAX and max_stale / doc_len >= P1_MAX_STALE_COVERAGE:
        return False, "stale_covers_docstring", None
    # 3) Лексическая связность: значимый токен диффа в stale±1.
    if not _has_lexical_link(parts["diff"], doc_old, stmts):
        return False, "stale_not_linked_to_diff", None
    return True, "ok", parts


# --- Лексическая связность P1 (relevance check) ---

_P1_STOP = frozenset({
    # Python-служебные и английские стоп-слова; True/False/None, числа,
    # list/tuple/dict намеренно ОСТАВЛЕНЫ — они значимы для поведения.
    "def", "return", "if", "else", "elif", "for", "while", "in", "is", "not",
    "and", "or", "import", "from", "as", "with", "try", "except", "finally",
    "raise", "assert", "pass", "break", "continue", "lambda", "self", "cls",
    "super", "print", "open", "close", "class",
    "a", "an", "the", "to", "of", "on", "it", "at", "by", "be", "are",
    "this", "that", "you", "your", "its", "will", "can", "has", "have",
})

_P1_SENT_SPLIT = re.compile(r"(?<=[.!?:;])\s+|\n+")


def _split_ident(token: str) -> list[str]:
    parts = re.split(r"_+", token)
    out: list[str] = []
    for part in parts:
        camel = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", part)
        if camel:
            out.extend(camel)
        elif part:
            out.append(part)
    return out


def _stem_token(word: str) -> str:
    word = word.lower()
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("ses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokens_of_text(text: str) -> set[str]:
    toks: set[str] = set()
    for raw in _IDENT_RE.findall(text or ""):
        for sub in _split_ident(raw):
            if len(sub) < 3:
                continue
            low = sub.lower()
            if low in _P1_STOP:
                continue
            toks.add(_stem_token(low))
    for num in _NUMBER_RE.findall(text or ""):
        toks.add(num.lower())
    return toks


def _diff_significant_tokens(diff: str) -> set[str]:
    toks: set[str] = set()
    for line in (diff or "").splitlines():
        if line.startswith(("+++", "---", "@@")):
            continue
        if not line.startswith(("+", "-")):
            continue
        toks |= _tokens_of_text(line[1:])
    return toks


def _split_sentences(text: str) -> list[str]:
    return [p.strip() for p in _P1_SENT_SPLIT.split((text or "").strip()) if p.strip()]


def _stale_context(doc_old: str, stmts: list[Any]) -> str:
    sents = _split_sentences(doc_old)
    idxs: list[int] = []
    for stmt in stmts:
        if not isinstance(stmt, str):
            continue
        norm = normalize_ws(stmt).lower()
        best: int | None = None
        for i, sent in enumerate(sents):
            sent_norm = normalize_ws(sent).lower()
            if sent_norm in norm or norm in sent_norm:
                best = i
                break
        if best is None:
            frag = norm[:30]
            for i, sent in enumerate(sents):
                if frag and frag in normalize_ws(sent).lower():
                    best = i
                    break
        if best is not None:
            idxs.append(best)
    if not idxs:
        return " ".join(s for s in stmts if isinstance(s, str))
    lo = max(0, min(idxs) - 1)
    hi = min(len(sents), max(idxs) + 2)
    return " ".join(sents[lo:hi])


def _tokens_linked(diff_toks: set[str], ctx_toks: set[str]) -> bool:
    if diff_toks & ctx_toks:
        return True
    # Подстрока длиной ≥4 (doc в document, record в _record_buffer после сплита
    # уже покрыто; это страховка для docs/Document и подобных).
    for d in diff_toks:
        if len(d) < 4:
            continue
        for c in ctx_toks:
            if len(c) < 4:
                continue
            if d in c or c in d:
                return True
    return False


def _has_lexical_link(diff: str, doc_old: str, stmts: list[Any]) -> bool:
    diff_toks = _diff_significant_tokens(diff)
    if not diff_toks:
        return False
    ctx = _stale_context(doc_old, stmts)
    ctx_toks = _tokens_of_text(ctx)
    return _tokens_linked(diff_toks, ctx_toks)


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


_LITERAL_KEEP = frozenset({"True", "False", "None"})
P3_MIN_RATIO = 0.6
P3_MAX_RATIO = 1.6


# Заголовки секций docstring (Google/NumPy-стиль) — не идентификаторы,
# хотя подходят под эвристику ALLCAPS (13.10: ~20 ложных identifier_lost).
_SECTION_HEADERS = frozenset({
    "ARGS", "ARGUMENTS", "PARAMETERS", "PARAMS", "RETURNS", "RETURN",
    "RAISES", "YIELDS", "YIELD", "EXAMPLES", "EXAMPLE", "NOTES", "NOTE",
})


def must_keep_identifiers(doc_orig: str) -> list[str]:
    keep: set[str] = set()
    for tok in _IDENT_RE.findall(doc_orig or ""):
        if tok in _SECTION_HEADERS:
            continue
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
