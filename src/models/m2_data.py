"""FD-21: построение входа M2 по R10.

Вход: A = sig_old + diff (ханки с сигнатурой первыми) + B = doc_old (до 192 токенов).
Формат зависит от семейства токенизатора:
- RoBERTa (UniXcoder, CodeBERT): <s> A </s></s> B </s>  (4 спецтокена)
- ModernBERT: [CLS] A [SEP] B [SEP]  (3 спецтокена)

Бюджет: len(B) <= 192 токена; len(A) <= max_len - len(B) - n_special.
Усечение фиксируется флагом truncated (для шага 4 FD-21).
"""
from __future__ import annotations


#: 8 классов головы дрейфа R10 (класса none в голове нет).
DRIFT_CLASSES = [
    "param_renamed",
    "param_removed",
    "param_added",
    "default_changed",
    "return_changed",
    "exception_changed",
    "behavior_changed",
    "unknown",
]

DRIFT_TO_ID = {c: i for i, c in enumerate(DRIFT_CLASSES)}
UNKNOWN_ID = DRIFT_TO_ID["unknown"]

DOC_BUDGET_TOKENS = 192


def drift_to_id(drift_type: str) -> int:
    """ID класса дрейфа; none/пустое -> -100 (маскируется в лоссе)."""
    d = str(drift_type)
    if d == "none" or not d:
        return -100
    return DRIFT_TO_ID.get(d, -100)


def is_sig_hunk(hunk: str) -> bool:
    """Ханк затрагивает сигнатуру: есть +/- строки с def."""
    for line in hunk.splitlines():
        s = line.strip()
        if s.startswith(("-def", "+def", "-async def", "+async def")):
            return True
        if (line.startswith("-") or line.startswith("+")) and "def " in line:
            return True
    return False


def split_hunks(diff_code: str) -> list[str]:
    """Разбить unified diff на ханки по заголовкам @@."""
    if not diff_code:
        return []
    hunks: list[str] = []
    cur: list[str] = []
    for line in str(diff_code).splitlines():
        if line.startswith("@@"):
            if cur:
                hunks.append("\n".join(cur))
            cur = [line]
        else:
            if cur:
                cur.append(line)
            # строки заголовка diff (---/+++/diff --git) вне ханков пропускаем
    if cur:
        hunks.append("\n".join(cur))
    if not hunks:
        # дифф без @@-заголовков (короткий): один кусок
        body = [ln for ln in str(diff_code).splitlines()
                if not ln.startswith(("---", "+++"))]
        return ["\n".join(body)] if body else []
    return hunks


def prioritize_diff(diff_code: str) -> str:
    """Ханки с сигнатурой первыми (в исходном порядке внутри групп)."""
    hunks = split_hunks(diff_code)
    sig = [h for h in hunks if is_sig_hunk(h)]
    rest = [h for h in hunks if not is_sig_hunk(h)]
    return "\n".join(sig + rest)


def n_special_tokens(tokenizer) -> int:
    """4 для RoBERTa-семейства (<s>), 3 для ModernBERT ([CLS])."""
    cls = getattr(tokenizer, "cls_token", "") or ""
    if cls == "[CLS]":
        return 3
    return 4


def build_text_a(sig_old: str, diff_code: str) -> str:
    """A = сигнатура + приоритизированный дифф."""
    sig = (sig_old or "").strip()
    diff = prioritize_diff(diff_code or "")
    if sig and diff:
        return sig + "\n" + diff
    return sig or diff


def encode_pair(tokenizer, sig_old: str, diff_code: str, doc_old: str,
                 max_len: int = 1024,
                 doc_budget: int = DOC_BUDGET_TOKENS) -> dict:
    """Закодировать пару в input_ids/attention_mask по R10.

    Возвращает dict(input_ids, attention_mask, truncated, n_a_tok, n_b_tok).
    truncated=True, если A пришлось резать под бюджет.
    """
    text_a = build_text_a(sig_old, diff_code)
    doc = doc_old or ""
    # B: первые doc_budget токенов без спецтокенов
    b_ids: list[int] = tokenizer.encode(doc, add_special_tokens=False)[:doc_budget]
    n_special = n_special_tokens(tokenizer)
    budget_a = max(0, max_len - len(b_ids) - n_special)
    a_ids_full: list[int] = tokenizer.encode(text_a, add_special_tokens=False)
    truncated = len(a_ids_full) > budget_a
    if truncated:
        # построчная страховка: режем дифф по строкам с пометкой ...
        lines = text_a.splitlines()
        # бинарный поиск по числу строк не нужен: режем с конца до влезания
        cut = text_a
        # быстрая оценка: обрезаем токены, но помечаем усечение маркером
        a_ids = a_ids_full[:budget_a]
    else:
        a_ids = a_ids_full
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        raise ValueError("У токенизатора нет cls/sep токенов")
    if n_special == 3:
        # ModernBERT: [CLS] A [SEP] B [SEP]
        input_ids = [cls_id] + a_ids + [sep_id] + b_ids + [sep_id]
    else:
        # RoBERTa: <s> A </s></s> B </s>
        input_ids = [cls_id] + a_ids + [sep_id, sep_id] + b_ids + [sep_id]
    # добивка маркера усечения в декодируемом виде не добавляется в ids
    # (флаг truncated фиксируется отдельно для отчёта шага 4 FD-21)
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "truncated": bool(truncated),
        "n_a_tok": len(a_ids),
        "n_b_tok": len(b_ids),
    }
