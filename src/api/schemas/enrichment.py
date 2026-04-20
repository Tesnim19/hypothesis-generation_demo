from __future__ import annotations

from fastapi import Query
from pydantic import BaseModel, Field, field_validator


class EnrichmentsListResponse(BaseModel):
    enrichments: list[dict]


class EnrichPostBody(BaseModel):
    """Optional JSON body fields (also accepted as query params)."""

    variant: str | None = None
    project_id: str | None = None
    tissue_name: str | None = None
    seed: int = 42

    @field_validator("seed", mode="before")
    @classmethod
    def coerce_seed(cls, v: object) -> int:
        if v is None or v == "":
            return 42
        try:
            return int(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise ValueError("seed must be an integer")


class EnrichPostAcceptedResponse(BaseModel):
    hypothesis_id: str
    project_id: str


class EnrichQueryParams(BaseModel):
    id: str | None = None
    project_id: str | None = None


def get_enrich_query_params(
    id: str | None = Query(None),
    project_id: str | None = Query(None),
) -> EnrichQueryParams:
    return EnrichQueryParams(id=id, project_id=project_id)
