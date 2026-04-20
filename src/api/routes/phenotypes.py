from __future__ import annotations

from typing import Union

from fastapi import APIRouter, Body, Depends, HTTPException
from loguru import logger
from pydantic import ValidationError

from src.api.dependencies import _deps
from src.api.schemas.phenotypes import (
    PhenotypeBulkItem,
    PhenotypeBulkResponse,
    PhenotypeListParams,
    PhenotypeListResponse,
    PhenotypeSingleWrapResponse,
    get_phenotype_list_params,
)
from src.utils import serialize_datetime_fields

router = APIRouter()


@router.get(
    "/phenotypes",
    response_model=Union[PhenotypeSingleWrapResponse, PhenotypeListResponse],
    response_model_exclude_none=True,
)
async def get_phenotypes(
    params: PhenotypeListParams = Depends(get_phenotype_list_params),
):
    phenotypes = _deps["phenotypes"]
    id_ = params.id
    search = params.search
    limit = params.limit
    skip = params.skip

    try:
        if id_:
            phenotype = phenotypes.get_phenotypes(phenotype_id=id_)
            if not phenotype:
                raise HTTPException(status_code=404, detail="Phenotype not found")
            wrapped = serialize_datetime_fields({"phenotype": phenotype})
            return PhenotypeSingleWrapResponse.model_validate(wrapped)

        lim = limit if limit is not None else 100

        all_phenotypes = phenotypes.get_phenotypes(
            limit=lim, skip=skip, search_term=search
        )
        total_count = phenotypes.count_phenotypes(search_term=search)

        response: dict = {
            "phenotypes": all_phenotypes,
            "total_count": total_count,
            "skip": skip,
            "limit": lim,
            "has_more": (skip + len(all_phenotypes)) < total_count,
            "next_skip": (
                skip + len(all_phenotypes)
                if (skip + len(all_phenotypes)) < total_count
                else None
            ),
        }
        if search:
            response["search_term"] = search

        return PhenotypeListResponse.model_validate(serialize_datetime_fields(response))

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error getting phenotypes: {exc}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get phenotypes: {exc}"
        )


@router.post("/phenotypes", status_code=201, response_model=PhenotypeBulkResponse)
async def post_phenotypes(data: list = Body(...)):
    phenotypes = _deps["phenotypes"]
    try:
        if not isinstance(data, list):
            raise HTTPException(
                status_code=400, detail="Expected JSON array of phenotypes"
            )

        phenotypes_data = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                row = PhenotypeBulkItem.model_validate(item)
            except ValidationError:
                logger.warning(f"Skipping invalid phenotype entry: {item}")
                continue
            phenotype = {"id": row.id, "phenotype_name": row.name}
            if phenotype["id"] and phenotype["phenotype_name"]:
                phenotypes_data.append(phenotype)
            else:
                logger.warning(f"Skipping invalid phenotype entry: {item}")

        if not phenotypes_data:
            raise HTTPException(
                status_code=400, detail="No valid phenotypes found in JSON data"
            )

        result = phenotypes.bulk_create_phenotypes(phenotypes_data)
        return PhenotypeBulkResponse(
            message="Phenotypes loaded successfully",
            inserted_count=result["inserted_count"],
            skipped_count=result["skipped_count"],
            total_provided=len(phenotypes_data),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error loading phenotypes: {exc}")
        raise HTTPException(
            status_code=500, detail=f"Failed to load phenotypes: {exc}"
        )
