"""Обнаружение разделов docstring (Args/Returns/Raises).

Единый источник правды для M1 (``src/models/baselines.py``) и FD-15
(``src/data/metadata_extractor.py``): маркеры Google-стиля (``Args:``),
RST (``:param:``), NumPy-подчёркивания (``Parameters\\n----------``).
Голые слова без маркера секцией НЕ считаются.
"""
from __future__ import annotations

import re
from typing import Tuple

_RE_ARGS = re.compile(
    r"(?m)^\s*(?:args|arguments|parameters|params)\s*:"
    r"|^\s*(?:args|arguments|parameters|params)\s*\n\s*[-=]{2,}"
    r"|:param\b"
)
_RE_RETURNS = re.compile(
    r"(?m)^\s*(?:returns?|return value|rtype)\s*:"
    r"|^\s*returns?\s*\n\s*[-=]{2,}"
    r"|^\s*(?:возврат|возвращает)\s*:"
    r"|:returns?\b|:return\b|:rtype\b"
)
_RE_RAISES = re.compile(
    r"(?m)^\s*(?:raises?|exceptions?)\s*:"
    r"|^\s*(?:raises?|exceptions?)\s*\n\s*[-=]{2,}"
    r"|^\s*(?:исключения?|ошибки?)\s*:"
    r"|:raises?\b|:exception\b"
)


def doc_section_flags(doc: str) -> Tuple[int, int, int]:
    """Флаги (has_args, has_returns, has_raises) для docstring."""
    if not doc:
        return 0, 0, 0
    low = doc.lower()
    return (
        int(bool(_RE_ARGS.search(low))),
        int(bool(_RE_RETURNS.search(low))),
        int(bool(_RE_RAISES.search(low))),
    )
