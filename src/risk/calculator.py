from __future__ import annotations

from collections.abc import Iterable
import math

from src.risk.config import RiskConfig, load_risk_config


def clamp(val: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp value into [low, high]."""
    return max(low, min(high, val))


def reset_risk() -> float:
    """Return initial risk score after verification (0.0)."""
    return 0.0


def accumulate_next_risk(current_risk: float, p_new: float) -> float:
    """Accumulate single new drift probability into current risk (R12).

    Formula: R_new = 1 - (1 - R_current) * (1 - p_new)
    """
    r_clamped = clamp(float(current_risk), 0.0, 1.0)
    p_clamped = clamp(float(p_new), 0.0, 1.0)
    accumulated = 1.0 - (1.0 - r_clamped) * (1.0 - p_clamped)
    return clamp(accumulated, 0.0, 1.0)


def accumulate_risk(probabilities: Iterable[float]) -> float:
    """Accumulate risk across multiple drift events since last verification (R12).

    Formula: R = 1 - Prod(1 - p_i)
    """
    prod_complement = 1.0
    for p in probabilities:
        p_clamped = clamp(float(p), 0.0, 1.0)
        prod_complement *= 1.0 - p_clamped
        if prod_complement == 0.0:
            break
    return clamp(1.0 - prod_complement, 0.0, 1.0)


def is_public_symbol(symbol: str | None) -> bool:
    """Determine whether a symbol is public by name.

    Per project convention, names starting with '_' are considered private/internal.
    """
    if not symbol:
        return False
    # If symbol includes class/method qualification, check the final target name
    leaf = symbol.rsplit(".", 1)[-1].strip()
    if not leaf:
        return False
    return not leaf.startswith("_")


def compute_impact(
    fan_in: int | float,
    is_public: bool = True,
    fan_in_cap: float = 20.0,
    private_factor: float = 0.5,
) -> float:
    """Compute impact score based on call count and publicity (R12).

    Formula:
      min(1.0, log1p(fan_in) / log1p(fan_in_cap)) * (1.0 if is_public else private_factor)
    """
    safe_fan_in = max(0.0, float(fan_in))
    safe_cap = max(1.0, float(fan_in_cap))

    log_ratio = math.log1p(safe_fan_in) / math.log1p(safe_cap)
    normalized_fan_in = clamp(log_ratio, 0.0, 1.0)

    visibility_factor = 1.0 if is_public else clamp(float(private_factor), 0.0, 1.0)
    impact = normalized_fan_in * visibility_factor
    return clamp(impact, 0.0, 1.0)


def compute_urgency(
    risk: float,
    doc_type: str = "docstring",
    days_since_verified: float = 0.0,
    config: RiskConfig | None = None,
    type_weight: float | None = None,
    age_gain: float | None = None,
    age_cap_days: float | None = None,
) -> float:
    """Compute urgency score based on accumulated risk, doc type, and elapsed time (R12).

    Formula:
      urgency = R * type_weight * (1 + age_gain * min(days, age_cap) / age_cap)
    """
    r_clamped = clamp(float(risk), 0.0, 1.0)
    if r_clamped == 0.0:
        return 0.0

    cfg = config or load_risk_config()

    actual_weight = type_weight if type_weight is not None else cfg.get_type_weight(doc_type)
    actual_age_gain = age_gain if age_gain is not None else cfg.age_gain
    actual_age_cap = age_cap_days if age_cap_days is not None else cfg.age_cap_days

    safe_days = max(0.0, float(days_since_verified))
    safe_cap = max(1.0, float(actual_age_cap))

    time_factor = 1.0 + float(actual_age_gain) * (min(safe_days, safe_cap) / safe_cap)
    urgency = r_clamped * float(actual_weight) * time_factor
    return max(0.0, urgency)
