from __future__ import annotations

from fastapi import Query
from pydantic import BaseModel, ConfigDict, Field


class PhenotypeListParams(BaseModel):
    id: str | None = Field(default=None, description="Single phenotype id")
    search: str | None = Field(default=None)
    limit: int | None = Field(default=None)
    skip: int = Field(default=0)


def get_phenotype_list_params(
    id: str | None = Query(None),
    search: str | None = Query(None),
    limit: int | None = Query(None),
    skip: int = Query(0),
) -> PhenotypeListParams:
    return PhenotypeListParams(id=id, search=search, limit=limit, skip=skip)


class PhenotypeBulkItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = ""
    name: str = ""


class PhenotypeBulkResponse(BaseModel):
    message: str
    inserted_count: int
    skipped_count: int
    total_provided: int


class PhenotypeSingleWrapResponse(BaseModel):
    phenotype: dict


class PhenotypeListResponse(BaseModel):
    phenotypes: list
    total_count: int
    skip: int
    limit: int
    has_more: bool
    next_skip: int | None = None
    search_term: str | None = None
