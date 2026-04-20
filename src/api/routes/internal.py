from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException
from loguru import logger
from pydantic import ValidationError

from src.api.auth import verify_service_token
from src.api.schemas.internal import (
    HealthResponse,
    InternalTaskUpdatePayload,
    InternalTaskUpdateResponse,
)
from src.socketio_instance import sio

router = APIRouter()


@router.post("/internal/task-update", status_code=200)
async def internal_task_update(
    payload: dict = Body(...),
    _: None = Depends(verify_service_token),
) -> InternalTaskUpdateResponse:
    """Receive a Prefect task-update POST and broadcast to Socket.IO room."""
    try:
        validated = InternalTaskUpdatePayload.model_validate(payload)
    except ValidationError as exc:
        errs = exc.errors()
        err0 = errs[0] if errs else {}
        ctx_err = err0.get("ctx", {}).get("error")
        if ctx_err is not None:
            detail = str(ctx_err)
        else:
            detail = err0.get("msg", "Invalid payload")
        raise HTTPException(status_code=400, detail=detail) from exc

    rest = validated.model_dump(exclude={"target_room", "event"})
    await sio.emit(validated.event, rest, room=validated.target_room)
    logger.info(f"[HTTP bridge] Broadcast {validated.event} to room '{validated.target_room}'")
    return InternalTaskUpdateResponse(
        status="broadcasted", room=validated.target_room, event=validated.event
    )


@router.get("/health")
async def health_check() -> HealthResponse:
    return HealthResponse(status="healthy")
