"""Token frequency analysis for FD-15 shortcut detection.

Computes **per-document (per-pair) frequency**: for each token, what
fraction of positive pairs contains it vs what fraction of negative
pairs contains it. Only **added lines** of the diff (lines starting
with ``+``) are considered, as required by workflow.md step 2.
"""
from __future__ import annotations

import keyword
import re
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
    absolute: bool = True,
) -> List[Tuple[str, float, float]]:
    """Return tokens with ``|pos_frac - neg_frac| > gap``.

    This matches workflow.md: "выписать токены с разницей > 0,3"
    (difference read as absolute value, so negative-side artefacts
    are caught too).

    Returns:
        Sorted list of ``(token, pos_frac, neg_frac)`` by descending
        absolute gap.
    """
    if absolute:
        hits = [
            (tok, pf, nf)
            for tok, (pf, nf) in freqs.items()
            if abs(pf - nf) > gap
        ]
        hits.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)
    else:
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
    by_abs: bool = False,
) -> List[Tuple[str, float, float, float]]:
    """Return top-*n* tokens by gap (descending).

    By default sorts by ``pos_frac - neg_frac``; with ``by_abs=True``
    sorts by absolute gap so negative-side markers surface too.

    Returns:
        ``[(token, pos_frac, neg_frac, gap), ...]``
    """
    ranked = [
        (tok, pf, nf, pf - nf)
        for tok, (pf, nf) in freqs.items()
    ]
    if by_abs:
        ranked.sort(key=lambda x: abs(x[3]), reverse=True)
    else:
        ranked.sort(key=lambda x: x[3], reverse=True)
    return ranked[:n]


_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: Identifier suffixes typical of synthetic renaming (R6 pools use the
#: same styles for positive and negative renames, so a large gap here
#: would mean the pools leaked the class).
_SUFFIX_RE = re.compile(
    r"_(tmp|temp|new|old|bak|copy|v\d+|2)$|^(tmp|res|out|result_value)$",
    re.IGNORECASE,
)

#: Literal constants (booleans, None, numbers). Large gaps outside the
#: structural ones (None from return_changed, bool flips from
#: default_changed) would signal template constants leaking the class.
_LITERAL_RE = re.compile(
    r"^(?:true|false|none|\d+\.?\d*)$", re.IGNORECASE
)

_STRUCTURAL_WORDS = frozenset(
    list(keyword.kwlist)
    + ["none", "true", "false", "optional", "def", "return", "raise"]
)


def word_token_gaps(
    pairs: List[Tuple[str, int]],
) -> Dict[str, Tuple[float, float]]:
    """Per-document frequencies with ``\\w+`` tokenisation (lowercased).

    Complements the whitespace tokenisation: catches identifier
    suffixes and constants that whitespace splitting dilutes
    (e.g. ``verbose_tmp=False,`` hides the ``_tmp`` suffix).
    """
    pos_docs: Counter[str] = Counter()
    neg_docs: Counter[str] = Counter()
    n_pos = 0
    n_neg = 0
    for diff, label in pairs:
        added = " ".join(
            line.lstrip()[1:]
            for line in diff.splitlines()
            if line.lstrip().startswith("+")
            and not line.lstrip().startswith("+++")
        )
        toks = set(_WORD_RE.findall(added.lower()))
        if label == 1:
            n_pos += 1
            pos_docs.update(toks)
        else:
            n_neg += 1
            neg_docs.update(toks)
    n_pos = max(n_pos, 1)
    n_neg = max(n_neg, 1)
    return {
        t: (pos_docs[t] / n_pos, neg_docs[t] / n_neg)
        for t in set(pos_docs) | set(neg_docs)
    }


def max_suffix_gap(
    word_freqs: Dict[str, Tuple[float, float]],
) -> Tuple[float, str]:
    """Largest absolute gap among name-suffix-like tokens.

    Returns ``(max_gap, token)``; ``(0.0, "")`` when no such token.
    """
    best = (0.0, "")
    for tok, (pf, nf) in word_freqs.items():
        if _SUFFIX_RE.search(tok):
            g = abs(pf - nf)
            if g > best[0]:
                best = (g, tok)
    return best


def max_literal_gap(
    word_freqs: Dict[str, Tuple[float, float]],
    exclude_structural: bool = True,
) -> Tuple[float, str]:
    """Largest absolute gap among literal-constant tokens.

    With ``exclude_structural=True`` the expected structural
    literals (None/True/False from ``return_changed`` /
    ``default_changed``) are skipped, isolating template-constant
    artefacts. Returns ``(max_gap, token)``.
    """
    best = (0.0, "")
    for tok, (pf, nf) in word_freqs.items():
        if not _LITERAL_RE.match(tok):
            continue
        if exclude_structural and tok in _STRUCTURAL_WORDS:
            continue
        g = abs(pf - nf)
        if g > best[0]:
            best = (g, tok)
    return best
