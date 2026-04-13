from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException
from loguru import logger

from api.auth import verify_service_token
from src.socketio_instance import sio

router = APIRouter()


@router.post("/internal/task-update", status_code=200)
async def internal_task_update(
    payload: dict = Body(...),
    _: None = Depends(verify_service_token),
) -> dict:
    """Receive a Prefect task-update POST and broadcast to Socket.IO room."""
    target_room = payload.pop("target_room", None)
    if not target_room:
        raise HTTPException(status_code=400, detail="target_room is required")

    event_name = payload.pop("event", "task_update")
    try:
        await sio.emit(event_name, payload, room=target_room)
    except Exception as exc:
        logger.exception(
            f"[HTTP bridge] Socket.IO emit failed for event={event_name!r} room={target_room!r}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Socket.IO broadcast failed: {exc}",
        ) from exc
    logger.info(f"[HTTP bridge] Broadcast {event_name} to room '{target_room}'")
    return {"status": "broadcasted", "room": target_room, "event": event_name}


@router.get("/health")
async def health_check():
    return {"status": "healthy"}
