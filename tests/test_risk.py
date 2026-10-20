from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.risk import (
    RiskConfig,
    accumulate_next_risk,
    accumulate_risk,
    analyze_repo_fan_in,
    compute_impact,
    compute_urgency,
    count_symbol_calls,
    is_public_symbol,
    load_risk_config,
    parse_fan_in_from_source,
    reset_risk,
)


class TestRiskAccumulation:
    def test_empty_probabilities(self) -> None:
        assert accumulate_risk([]) == 0.0

    def test_single_probability(self) -> None:
        assert accumulate_risk([0.4]) == pytest.approx(0.4)
        assert accumulate_risk([1.0]) == pytest.approx(1.0)
        assert accumulate_risk([0.0]) == pytest.approx(0.0)

    def test_multiple_independent_probabilities(self) -> None:
        # R = 1 - (1 - 0.5) * (1 - 0.5) = 0.75
        assert accumulate_risk([0.5, 0.5]) == pytest.approx(0.75)

        # R = 1 - (1 - 0.2) * (1 - 0.3) * (1 - 0.4) = 1 - 0.8 * 0.7 * 0.6 = 1 - 0.336 = 0.664
        assert accumulate_risk([0.2, 0.3, 0.4]) == pytest.approx(0.664)

    def test_certain_drift_yields_one(self) -> None:
        assert accumulate_risk([0.2, 1.0, 0.4]) == pytest.approx(1.0)

    def test_clamping_out_of_bounds_inputs(self) -> None:
        assert accumulate_risk([-0.5]) == pytest.approx(0.0)
        assert accumulate_risk([1.5]) == pytest.approx(1.0)
        assert accumulate_risk([0.3, -0.2, 1.2]) == pytest.approx(1.0)

    def test_sequential_accumulation_matches_batch(self) -> None:
        probs = [0.25, 0.4, 0.6]
        batch_res = accumulate_risk(probs)

        current = reset_risk()
        assert current == 0.0
        for p in probs:
            current = accumulate_next_risk(current, p)

        assert current == pytest.approx(batch_res)

    def test_reset_risk(self) -> None:
        assert reset_risk() == 0.0


class TestImpactHeuristic:
    def test_zero_fan_in(self) -> None:
        assert compute_impact(0, is_public=True) == 0.0
        assert compute_impact(0, is_public=False) == 0.0

    def test_negative_fan_in(self) -> None:
        assert compute_impact(-5, is_public=True) == 0.0

    def test_fan_in_cap_reached(self) -> None:
        # At fan_in = fan_in_cap (20), log1p(20) / log1p(20) = 1.0
        assert compute_impact(20, is_public=True, fan_in_cap=20) == pytest.approx(1.0)

    def test_fan_in_cap_exceeded(self) -> None:
        assert compute_impact(100, is_public=True, fan_in_cap=20) == pytest.approx(1.0)

    def test_private_factor_discount(self) -> None:
        # At fan_in = 20, impact is 1.0 * private_factor (0.5) = 0.5
        assert compute_impact(
            20, is_public=False, fan_in_cap=20, private_factor=0.5
        ) == pytest.approx(0.5)

    def test_monotonicity(self) -> None:
        impact1 = compute_impact(2, is_public=True)
        impact2 = compute_impact(5, is_public=True)
        impact3 = compute_impact(15, is_public=True)
        assert 0.0 < impact1 < impact2 < impact3 < 1.0


class TestPublicity:
    def test_public_names(self) -> None:
        assert is_public_symbol("fetch") is True
        assert is_public_symbol("get_user") is True
        assert is_public_symbol("Module.execute") is True

    def test_private_names(self) -> None:
        assert is_public_symbol("_internal") is False
        assert is_public_symbol("__dunder_or_private") is False
        assert is_public_symbol("Module._private_method") is False
        assert is_public_symbol("") is False
        assert is_public_symbol(None) is False


class TestUrgencyHeuristic:
    def test_zero_risk_yields_zero_urgency(self) -> None:
        assert compute_urgency(0.0, days_since_verified=100) == 0.0

    def test_zero_days_elapsed(self) -> None:
        # urgency = R * type_weight * 1.0
        assert compute_urgency(0.8, doc_type="docstring", days_since_verified=0) == pytest.approx(
            0.8
        )

    def test_negative_days_treated_as_zero(self) -> None:
        assert compute_urgency(0.8, doc_type="docstring", days_since_verified=-10) == pytest.approx(
            0.8
        )

    def test_age_cap_saturation(self) -> None:
        # At 180 days (age_cap_days=180, age_gain=0.5): factor = 1 + 0.5 * 180/180 = 1.5
        urgency_at_cap = compute_urgency(0.6, doc_type="docstring", days_since_verified=180)
        assert urgency_at_cap == pytest.approx(0.6 * 1.0 * 1.5)

        # Beyond 180 days (e.g. 300), factor remains 1.5
        urgency_beyond = compute_urgency(0.6, doc_type="docstring", days_since_verified=300)
        assert urgency_beyond == pytest.approx(urgency_at_cap)

    def test_doc_type_weights(self) -> None:
        r = 0.5
        u_docstring = compute_urgency(r, doc_type="docstring", days_since_verified=0)
        u_readme = compute_urgency(r, doc_type="readme", days_since_verified=0)
        u_tutorial = compute_urgency(r, doc_type="tutorial", days_since_verified=0)
        u_api_ref = compute_urgency(r, doc_type="api_ref", days_since_verified=0)
        u_unknown = compute_urgency(r, doc_type="unknown_section", days_since_verified=0)

        assert u_docstring == pytest.approx(0.5 * 1.0)
        assert u_readme == pytest.approx(0.5 * 0.8)
        assert u_tutorial == pytest.approx(0.5 * 0.7)
        assert u_api_ref == pytest.approx(0.5 * 1.0)
        assert u_unknown == pytest.approx(0.5 * 1.0)


class TestConfigLoading:
    def test_load_default_config_file(self) -> None:
        cfg = load_risk_config()
        assert cfg.fan_in_cap == 20.0
        assert cfg.private_factor == 0.5
        assert cfg.age_cap_days == 180.0
        assert cfg.age_gain == 0.5
        assert cfg.threshold == 0.5
        assert cfg.get_type_weight("docstring") == 1.0
        assert cfg.get_type_weight("readme") == 0.8
        assert cfg.get_type_weight("tutorial") == 0.7

    def test_load_nonexistent_returns_defaults(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        cfg = load_risk_config(missing)
        assert cfg.fan_in_cap == 20.0
        assert cfg.threshold == 0.5

    def test_load_custom_yaml(self, tmp_path: Path) -> None:
        custom_yaml = tmp_path / "custom_risk.yaml"
        custom_yaml.write_text(
            """
type_weight:
  docstring: 1.2
impact:
  fan_in_cap: 35
  private_factor: 0.25
urgency:
  age_cap_days: 90
  age_gain: 0.8
threshold: 0.65
""",
            encoding="utf-8",
        )
        cfg = load_risk_config(custom_yaml)
        assert cfg.fan_in_cap == 35.0
        assert cfg.private_factor == 0.25
        assert cfg.age_cap_days == 90.0
        assert cfg.age_gain == 0.8
        assert cfg.threshold == 0.65
        assert cfg.get_type_weight("docstring") == 1.2


class TestFanInAST:
    def test_parse_fan_in_from_source(self) -> None:
        code = """
def sample():
    fetch("url1")
    fetch("url2")
    client.post("url3")
    nested(helper(10))
    _private_call()
"""
        counts = parse_fan_in_from_source(code)
        assert counts["fetch"] == 2
        assert counts["post"] == 1
        assert counts["nested"] == 1
        assert counts["helper"] == 1
        assert counts["_private_call"] == 1
        assert counts["nonexistent"] == 0

    def test_parse_syntax_error_source_returns_empty(self) -> None:
        bad_code = "def invalid syntax :::"
        assert parse_fan_in_from_source(bad_code) == {}

    def test_analyze_repo_fan_in(self, tmp_path: Path) -> None:
        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        (pkg_dir / "mod1.py").write_text("fetch('a'); compute()", encoding="utf-8")
        (pkg_dir / "mod2.py").write_text("fetch('b'); client.fetch('c')", encoding="utf-8")

        # Ignore folder test
        venv_dir = tmp_path / ".venv"
        venv_dir.mkdir()
        (venv_dir / "ignored.py").write_text("fetch('x'); fetch('y')", encoding="utf-8")

        # Syntax error file shouldn't crash
        (pkg_dir / "broken.py").write_text("def broken(:", encoding="utf-8")

        counts = analyze_repo_fan_in(tmp_path)
        # mod1 (1) + mod2 (2) = 3 (ignored excluded)
        assert counts.get("fetch") == 3
        assert counts.get("compute") == 1

        # Target symbol filter
        filtered = analyze_repo_fan_in(tmp_path, target_symbols=["fetch", "absent"])
        assert filtered == {"fetch": 3, "absent": 0}

        # count_symbol_calls
        assert count_symbol_calls(tmp_path, "fetch") == 3
        assert count_symbol_calls(tmp_path, "absent") == 0
