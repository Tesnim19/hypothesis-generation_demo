from __future__ import annotations

from fastapi import APIRouter, Depends
from loguru import logger

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
    validated: InternalTaskUpdatePayload,
    _: None = Depends(verify_service_token),
) -> InternalTaskUpdateResponse:
    """Receive a Prefect task-update POST and broadcast to Socket.IO room."""
    rest = validated.model_dump(exclude={"target_room", "event"})
    await sio.emit(validated.event, rest, room=validated.target_room)
    logger.info(f"[HTTP bridge] Broadcast {validated.event} to room '{validated.target_room}'")
    return InternalTaskUpdateResponse(
        status="broadcasted", room=validated.target_room, event=validated.event
    )


@router.get("/health")
async def health_check() -> HealthResponse:
    return HealthResponse(status="healthy")
