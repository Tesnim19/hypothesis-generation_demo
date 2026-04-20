from __future__ import annotations

from fastapi import HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field


class CredibleSetVariantRow(BaseModel):
    """One variant row from fine-mapping output (column names vary by pipeline)."""

    model_config = ConfigDict(extra="allow")

    variant: str | int | None = None
    rs_id: str | None = None
    beta: float | int | None = None
    chromosome: str | int | float | None = None
    log_pvalue: float | int | None = None
    position: int | float | None = None
    ref_allele: str | None = None
    minor_allele: str | None = None
    ref_allele_freq: float | int | None = None
    posterior_prob: float | int | None = None
    cs: float | int | None = Field(default=None, description="Credible set index when present")
    region_id: str | None = None
    region_chr: str | int | None = None
    region_center: int | float | None = None
    converged: bool | None = None
    credible_set: float | int | None = None


class CredibleSetsResponse(BaseModel):
    variants: list[CredibleSetVariantRow]


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
