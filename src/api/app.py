from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from fastapi import Body, Depends, FastAPI, HTTPException, Query, status

from src.api.schemas import (
    HistoryResponse,
    StateItem,
    StatusResponse,
    VerifyRequest,
    VerifyResponse,
)
from src.registry.storage import history, init_db, latest, verify

DEFAULT_DB_PATH = Path("data/registry.db")


def get_db_path() -> Path:
    """Resolve database path from environment or default."""
    env_path = os.environ.get("FRESHDOC_DB_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_DB_PATH


def create_app(db_path: Path | str | None = None) -> FastAPI:
    """Create and configure FreshDoc FastAPI application.

    Schema is created on startup (lifespan), not on import, so importing
    this module has no filesystem side effects.
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = getattr(app.state, "db_path", None)
        if db is not None and str(db) != ":memory:":
            init_db(db)
        yield

    app = FastAPI(
        title="FreshDoc Registry API",
        description="Documentation drift, state registry, and verification API (R12)",
        version="1.0.0",
        lifespan=lifespan,
    )

    resolved_db_path = Path(db_path) if db_path else get_db_path()
    app.state.db_path = resolved_db_path

    def db_dependency() -> Path | str:
        return getattr(app.state, "db_path", resolved_db_path)

    @app.get("/health", tags=["System"])
    def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    @app.get(
        "/status",
        response_model=StatusResponse,
        tags=["Registry"],
        summary="Get latest documentation state by branch or commit",
    )
    def get_status(
        repo: Annotated[str, Query(description="Repository name or slug, e.g. owner/repo")],
        branch: Annotated[str, Query(description="Branch name, e.g. main")],
        path: Annotated[str | None, Query(description="Optional prefix/filter by doc_path")] = None,
        commit: Annotated[str | None, Query(description="Optional commit SHA cutoff")] = None,
        min_risk: Annotated[float | None, Query(description="Minimum risk score filter")] = None,
        current_db: Path | str = Depends(db_dependency),
    ) -> StatusResponse:
        """Return latest state of each section matching criteria."""
        rows = latest(
            current_db,
            repo=repo,
            branch=branch,
            commit=commit,
            path=path,
            min_risk=min_risk,
        )
        return StatusResponse(
            repo=repo,
            branch=branch,
            commit=commit,
            items=[StateItem(**r) for r in rows],
        )

    @app.get(
        "/sections/history",
        response_model=HistoryResponse,
        tags=["Registry"],
        summary="Get chronological history of a documentation section",
    )
    def get_section_history(
        section_id: Annotated[str, Query(description="Identifier e.g. path.py::func#docstring")],
        repo: Annotated[str | None, Query(description="Optional repository filter")] = None,
        branch: Annotated[str | None, Query(description="Optional branch filter")] = None,
        current_db: Path | str = Depends(db_dependency),
    ) -> HistoryResponse:
        """Return full chronological audit trail of changes for section_id."""
        rows = history(current_db, section_id=section_id, repo=repo, branch=branch)
        return HistoryResponse(
            section_id=section_id,
            repo=repo,
            branch=branch,
            items=[StateItem(**r) for r in rows],
        )

    @app.post(
        "/verify",
        response_model=VerifyResponse,
        status_code=status.HTTP_200_OK,
        tags=["Registry"],
        summary="Verify documentation section and reset its risk to 0",
    )
    def verify_section(
        repo: Annotated[str | None, Query(description="Repository name")] = None,
        branch: Annotated[str | None, Query(description="Branch name")] = None,
        section_id: Annotated[str | None, Query(description="Section identifier")] = None,
        commit: Annotated[str | None, Query(description="Commit SHA being verified")] = None,
        body: Annotated[VerifyRequest | None, Body()] = None,
        current_db: Path | str = Depends(db_dependency),
    ) -> VerifyResponse:
        """Record manual or automated verification of a documentation section."""
        actual_repo = repo or (body.repo if body else None)
        actual_branch = branch or (body.branch if body else None)
        actual_section_id = section_id or (body.section_id if body else None)
        actual_commit = commit or (body.commit if body else None)
        verified_by = body.verified_by if body else None

        missing = []
        if not actual_repo:
            missing.append("repo")
        if not actual_branch:
            missing.append("branch")
        if not actual_section_id:
            missing.append("section_id")
        if not actual_commit:
            missing.append("commit")

        if missing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required parameter(s): {', '.join(missing)}",
            )

        assert actual_repo is not None
        assert actual_branch is not None
        assert actual_section_id is not None
        assert actual_commit is not None

        record = verify(
            current_db,
            repo=actual_repo,
            branch=actual_branch,
            section_id=actual_section_id,
            commit=actual_commit,
            verified_by=verified_by,
        )
        return VerifyResponse(status="verified", record=StateItem(**record))

    return app


app = create_app()
