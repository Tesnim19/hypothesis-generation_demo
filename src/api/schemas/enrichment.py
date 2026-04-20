from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import Query
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer


class EnrichmentListItem(BaseModel):
    """One enrichment document as returned from the enrich collection."""

    model_config = ConfigDict(populate_by_name=True)

    id: str | None = None
    mongo_id: str | None = Field(
        default=None,
        validation_alias="_id",
        description="MongoDB document id (serialized as _id in JSON)",
    )
    user_id: str | None = None
    project_id: str | None = None
    variant: str | None = None
    phenotype: str | None = None
    causal_gene: str | None = None
    GO_terms: Any | None = None
    causal_graph: Any | None = None
    created_at: datetime | str | None = None

    @model_serializer(mode="wrap")
    def _to_json(self, handler):  # type: ignore[no-untyped-def]
        data = handler(self)
        mid = data.pop("mongo_id", None)
        if mid is not None:
            data["_id"] = mid
        return data


class EnrichmentsListResponse(BaseModel):
    enrichments: list[EnrichmentListItem]


class EnrichPostBody(BaseModel):
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
