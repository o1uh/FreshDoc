from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import os
from pathlib import Path
import warnings

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "risk.yaml"

DEFAULT_TYPE_WEIGHTS: dict[str, float] = {
    "docstring": 1.0,
    "api_ref": 1.0,
    "tutorial": 0.7,
    "readme": 0.8,
}


@dataclass(frozen=True)
class RiskConfig:
    """Configuration for risk, impact, and urgency heuristics (R12)."""

    type_weight: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TYPE_WEIGHTS))
    fan_in_cap: float = 20.0
    private_factor: float = 0.5
    age_cap_days: float = 180.0
    age_gain: float = 0.5
    threshold: float = 0.5

    def get_type_weight(self, doc_type: str) -> float:
        """Return weight for document type; defaults to docstring weight or 1.0."""
        return self.type_weight.get(doc_type, self.type_weight.get("docstring", 1.0))


@lru_cache(maxsize=32)
def load_risk_config(config_path: Path | str | None = None) -> RiskConfig:
    """Load RiskConfig from YAML file with fallback to defaults.

    Results are cached per resolved path (lru_cache): replay FD-29 calls
    compute_urgency thousands of times without re-reading YAML on each call.
    After editing the YAML on disk call ``load_risk_config.cache_clear()``.
    Corrupted YAML warns and falls back to defaults (never silent).
    """
    path_to_use: Path | None = None
    if config_path is not None:
        path_to_use = Path(config_path)
    else:
        env_val = os.environ.get("FRESHDOC_RISK_CONFIG")
        if env_val:
            path_to_use = Path(env_val)
        elif DEFAULT_CONFIG_PATH.exists():
            path_to_use = DEFAULT_CONFIG_PATH
        elif Path("config/risk.yaml").exists():
            path_to_use = Path("config/risk.yaml")

    if path_to_use is None or not path_to_use.exists():
        return RiskConfig()

    try:
        with open(path_to_use, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:
        warnings.warn(
            f"risk config {path_to_use} unreadable ({exc}); using defaults",
            stacklevel=2,
        )
        return RiskConfig()

    type_weight = dict(raw.get("type_weight") or DEFAULT_TYPE_WEIGHTS)
    impact_raw = raw.get("impact") or {}
    urgency_raw = raw.get("urgency") or {}

    fan_in_cap = float(impact_raw.get("fan_in_cap", 20.0))
    private_factor = float(impact_raw.get("private_factor", 0.5))
    age_cap_days = float(urgency_raw.get("age_cap_days", 180.0))
    age_gain = float(urgency_raw.get("age_gain", 0.5))
    threshold = float(raw.get("threshold", 0.5))

    return RiskConfig(
        type_weight=type_weight,
        fan_in_cap=fan_in_cap,
        private_factor=private_factor,
        age_cap_days=age_cap_days,
        age_gain=age_gain,
        threshold=threshold,
    )
