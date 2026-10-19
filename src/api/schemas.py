from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class StateItem(BaseModel):
    """A single documentation section state item."""

    section_id: str
    symbol: str | None = None
    risk: float | None = None
    p_stale: float | None = None
    urgency: float | None = None
    impact: float | None = None
    drift_type: str | None = None
    evidence: Any | None = None
    verified_commit: str | None = None
    doc_path: str | None = None
    commit_sha: str | None = None
    build_id: str | None = None
    computed_at: str | None = None
    event: str | None = None
    seq: int | None = None


class StatusResponse(BaseModel):
    """Response model for GET /status matching R12 example."""

    repo: str
    branch: str
    commit: str | None = None
    items: list[StateItem] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    """Response model for GET /sections/history."""

    section_id: str
    repo: str | None = None
    branch: str | None = None
    items: list[StateItem] = Field(default_factory=list)


class VerifyRequest(BaseModel):
    """Optional payload body for POST /verify."""

    repo: str | None = None
    branch: str | None = None
    section_id: str | None = None
    commit: str | None = None
    verified_by: str | None = None


class VerifyResponse(BaseModel):
    """Response model for POST /verify."""

    status: str = "verified"
    record: StateItem
