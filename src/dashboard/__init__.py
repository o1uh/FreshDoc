"""FreshDoc Dashboard package (FD-30 / R12)."""

from src.dashboard.core import (
    execute_verify,
    get_available_branches,
    get_available_commits,
    get_available_projects,
    load_freshness_data,
    load_section_history,
)

__all__ = [
    "execute_verify",
    "get_available_branches",
    "get_available_commits",
    "get_available_projects",
    "load_freshness_data",
    "load_section_history",
]
