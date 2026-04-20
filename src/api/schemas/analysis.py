from __future__ import annotations

from fastapi import HTTPException, Query
from pydantic import BaseModel


class CredibleSetsResponse(BaseModel):
    variants: list[dict]


class CredibleSetsQueryParams(BaseModel):
    project_id: str
    credible_set_id: str


def get_credible_sets_params(
    project_id: str | None = Query(None),
    credible_set_id: str | None = Query(None),
) -> CredibleSetsQueryParams:
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id is required")
    if not credible_set_id:
        raise HTTPException(status_code=400, detail="Credible_set_id is required")
    return CredibleSetsQueryParams(
        project_id=project_id, credible_set_id=credible_set_id
    )
