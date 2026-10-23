"""FD-29 DLC: Kaplan-Meier survival math on synthetic data (no torch/git)."""

from __future__ import annotations

from src.registry.survival import kaplan_meier, survival_stats


def test_km_two_events() -> None:
    curve = kaplan_meier([5.0, 10.0], [True, True])
    assert curve[0] == (0.0, 1.0)
    assert curve[1] == (5.0, 0.5)
    assert curve[2] == (10.0, 0.0)


def test_km_censored_skips_drop_but_counts_at_risk() -> None:
    curve = kaplan_meier([5.0, 8.0, 10.0], [True, False, True])
    assert curve[0] == (0.0, 1.0)
    assert curve[1][0] == 5.0 and abs(curve[1][1] - 2 / 3) < 1e-9
    assert curve[-1] == (10.0, 0.0)
    assert len(curve) == 3  # censored t=8 adds no step


def test_km_all_censored_flat() -> None:
    assert kaplan_meier([30.0, 90.0], [False, False]) == [(0.0, 1.0)]
    assert kaplan_meier([], []) == []


def test_survival_stats_median_and_marks() -> None:
    stats = survival_stats([5.0, 8.0, 10.0], [True, False, True])
    assert stats["n_events"] == 2
    assert stats["n_censored"] == 1
    assert stats["median_days"] == 10.0  # first t with S <= 0.5
    assert stats["fixed_by_mark"][30.0] == 1.0
    assert stats["fixed_by_mark"][60.0] == 1.0


def test_survival_stats_heavy_censoring_no_median() -> None:
    stats = survival_stats([100.0, 200.0], [False, False])
    assert stats["median_days"] is None
    assert stats["fixed_by_mark"][90.0] == 0.0
