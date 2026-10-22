"""FD-26: общие helpers отбора и сведения (детерминизм, нормализация).

key_of: sha1(diff + \\x00 + doc) — нулевой байт-разделитель исключает
коллизии конкатенации вида ("ab","c")/("a","bc"); явная защита от NaN
(pd.isna -> ""), чтобы float('nan') не превращался в литерал "nan".
"""

from __future__ import annotations

import hashlib

import pandas as pd


def safe_str(val) -> str:
    """Строка без литерала "nan": NaN/None -> ""."""
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val)


def key_of(diff_code, doc_old) -> str:
    """Детерминированный ключ пары для манифеста (полный sha1 hex)."""
    payload = (safe_str(diff_code) + "\x00" + safe_str(doc_old)).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def norm_text(val) -> str:
    """Нормализация для проверки цитат: CRLF->LF, strip кавычек/пробелов."""
    s = safe_str(val).replace("\r\n", "\n")
    return s.strip().strip("\"'").strip()


def contains_quote(doc_old, stale_statement) -> bool:
    """Цитата засчитывается, если нормализованная подстрока входит в
    нормализованный docstring (устойчиво к CRLF/LF и концевым пробелам)."""
    s = norm_text(stale_statement)
    if not s:
        return False
    return s in norm_text(doc_old)
