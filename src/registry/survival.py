"""FD-29 DLC: survival analysis of registry warnings (post-processing only).

Reads flag→fix durations (observed) and open-flag ages (right-censored) and
builds the empirical Kaplan-Meier survival curve: X — days since the flag,
Y — share of alerts still open. Pure functions, no product code touched.
"""

from __future__ import annotations


def kaplan_meier(
    durations: list[float], observed: list[bool]
) -> list[tuple[float, float]]:
    """Empirical Kaplan-Meier survival curve.

    `durations[i]` — days from flag to fix (observed) or to window end
    (censored); `observed[i]` tells which. Returns [(t, S(t))] steps starting
    at (0.0, 1.0); S drops only at observed event times. Empty input → [].
    """
    pairs = sorted(zip(durations, observed), key=lambda p: p[0])
    if not pairs:
        return []
    curve: list[tuple[float, float]] = [(0.0, 1.0)]
    surv = 1.0
    i = 0
    n = len(pairs)
    while i < n:
        t = pairs[i][0]
        at_risk = sum(1 for d, _ in pairs if d >= t)
        died = 0
        j = i
        while j < n and pairs[j][0] == t:
            if pairs[j][1]:
                died += 1
            j += 1
        if died and at_risk:
            surv *= 1.0 - died / at_risk
            curve.append((float(t), float(surv)))
        i = j
    return curve


def survival_stats(
    durations: list[float],
    observed: list[bool],
    marks: tuple[float, ...] = (30.0, 60.0, 90.0, 180.0),
) -> dict:
    """Summary for the report: median survival, share fixed by each mark.

    Median is the first event time with S <= 0.5 (None when survival never
    drops that low — honest for heavily censored data). Shares are read off
    the KM curve (1 - S at each mark), not raw counts.
    """
    curve = kaplan_meier(durations, observed)
    median: float | None = None
    for t, s in curve:
        if s <= 0.5:
            median = t
            break
    fixed_by: dict[float, float] = {}
    for m in marks:
        s_at = 1.0
        for t, s in curve:
            if t <= m:
                s_at = s
            else:
                break
        fixed_by[m] = round(1.0 - s_at, 4)
    return {
        "n_events": int(sum(1 for o in observed if o)),
        "n_censored": int(sum(1 for o in observed if not o)),
        "median_days": median,
        "fixed_by_mark": fixed_by,
        "curve": curve,
    }
