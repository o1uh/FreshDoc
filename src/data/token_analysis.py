from __future__ import annotations

from collections import Counter
from typing import List, Tuple, Dict


def token_frequencies(pairs: List[Tuple[str, int]]) -> Dict[str, Tuple[float, float]]:
    """Calculate token frequencies in positive and negative pairs.

    Args:
        pairs: List of ``(diff_code, label)`` where ``label`` is 1 for positive
               (stale) and 0 for negative.
    Returns:
        Mapping ``token -> (pos_frac, neg_frac)`` where fractions are relative to
        the total number of tokens in each class.
    """
    pos_counter: Counter[str] = Counter()
    neg_counter: Counter[str] = Counter()
    pos_total = 0
    neg_total = 0
    for diff, label in pairs:
        tokens = diff.split()
        if label == 1:
            pos_counter.update(tokens)
            pos_total += len(tokens)
        else:
            neg_counter.update(tokens)
            neg_total += len(tokens)
    # Prevent division by zero
    pos_total = max(pos_total, 1)
    neg_total = max(neg_total, 1)
    result: Dict[str, Tuple[float, float]] = {}
    all_tokens = set(pos_counter) | set(neg_counter)
    for t in all_tokens:
        result[t] = (pos_counter[t] / pos_total, neg_counter[t] / neg_total)
    return result


def filter_shortcut_tokens(freqs: Dict[str, Tuple[float, float]], pos_thr: float = 0.30, neg_thr: float = 0.05) -> List[str]:
    """Return tokens that satisfy the shortcut criteria.

    A *shortcut* token appears in at least ``pos_thr`` proportion of positive
    tokens and less than ``neg_thr`` proportion of negative tokens.
    """
    return [tok for tok, (pos_frac, neg_frac) in freqs.items() if pos_frac >= pos_thr and neg_frac < neg_thr]
