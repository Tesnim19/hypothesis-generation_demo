from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from src.api.dependencies import _deps
from src.api.auth import get_current_user_id
from src.api.schemas.analysis import (
    CredibleSetsResponse,
    get_credible_sets_params,
    CredibleSetsQueryParams,
)
from src.utils import convert_variants_to_object_array, serialize_datetime_fields

router = APIRouter()


@router.get("/credible-sets", response_model=CredibleSetsResponse)
async def get_credible_sets(
    params: CredibleSetsQueryParams = Depends(get_credible_sets_params),
    current_user_id: str = Depends(get_current_user_id),
):
    analysis = _deps["analysis"]
    project_id = params.project_id
    credible_set_id = params.credible_set_id

    try:
        credible_set = analysis.get_credible_set_by_id(
            current_user_id, project_id, credible_set_id
        )
        if not credible_set:
            raise HTTPException(
                status_code=404, detail="No credible set found with this ID"
            )

        variants_data = credible_set.get("variants_data", {})
        if not variants_data:
            raise HTTPException(
                status_code=404,
                detail="No variants data found for this credible set",
            )

        variants = variants_data.get("data", {})
        variants_array = convert_variants_to_object_array(variants)
        variants_array = serialize_datetime_fields(variants_array)
        return CredibleSetsResponse(variants=variants_array)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching credible set: {exc}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch credible set: {exc}"
        )
