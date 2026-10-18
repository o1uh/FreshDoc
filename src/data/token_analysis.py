"""Token frequency analysis for FD-15 shortcut detection.

Computes **per-document (per-pair) frequency**: for each token, what
fraction of positive pairs contains it vs what fraction of negative
pairs contains it.  Only **added lines** of the diff (lines starting
with ``+``) are considered, as required by workflow.md step 2.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple


def _added_tokens(diff: str) -> List[str]:
    """Extract whitespace-split tokens from added lines of a diff.

    Added lines start with ``+`` (but not ``+++`` which is a header).
    """
    tokens: list[str] = []
    for line in diff.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("+") and not stripped.startswith("+++"):
            # Remove the leading '+' marker before tokenising
            tokens.extend(stripped[1:].split())
    return tokens


def document_frequencies(
    pairs: List[Tuple[str, int]],
) -> Dict[str, Tuple[float, float]]:
    """Per-document token frequencies for positive and negative pairs.

    For every token *t* computes:
    - ``pos_frac``: fraction of **positive** pairs whose added diff
      lines contain *t* (at least once).
    - ``neg_frac``: fraction of **negative** pairs whose added diff
      lines contain *t*.

    Args:
        pairs: ``(diff_code, label)`` where ``label`` 1 = stale,
               0 = not stale.

    Returns:
        ``{token: (pos_frac, neg_frac)}``
    """
    pos_docs: Counter[str] = Counter()
    neg_docs: Counter[str] = Counter()
    n_pos = 0
    n_neg = 0

    for diff, label in pairs:
        unique_tokens = set(_added_tokens(diff))
        if label == 1:
            n_pos += 1
            pos_docs.update(unique_tokens)
        else:
            n_neg += 1
            neg_docs.update(unique_tokens)

    n_pos = max(n_pos, 1)
    n_neg = max(n_neg, 1)

    result: Dict[str, Tuple[float, float]] = {}
    for t in set(pos_docs) | set(neg_docs):
        result[t] = (pos_docs[t] / n_pos, neg_docs[t] / n_neg)
    return result


def filter_shortcut_tokens(
    freqs: Dict[str, Tuple[float, float]],
    gap: float = 0.30,
) -> List[Tuple[str, float, float]]:
    """Return tokens where ``pos_frac - neg_frac > gap``.

    This matches workflow.md: "выписать токены с разницей > 0,3".

    Returns:
        Sorted list of ``(token, pos_frac, neg_frac)`` by descending
        gap.
    """
    hits = [
        (tok, pf, nf)
        for tok, (pf, nf) in freqs.items()
        if pf - nf > gap
    ]
    hits.sort(key=lambda x: x[1] - x[2], reverse=True)
    return hits


def top_suspicious(
    freqs: Dict[str, Tuple[float, float]],
    n: int = 30,
) -> List[Tuple[str, float, float, float]]:
    """Return top-*n* tokens by ``pos_frac - neg_frac`` (descending).

    Useful for the report even when no token crosses the 0.3 threshold.

    Returns:
        ``[(token, pos_frac, neg_frac, gap), ...]``
    """
    ranked = [
        (tok, pf, nf, pf - nf)
        for tok, (pf, nf) in freqs.items()
    ]
    ranked.sort(key=lambda x: x[3], reverse=True)
    return ranked[:n]
