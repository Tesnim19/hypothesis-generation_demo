from fastapi import APIRouter, Body, Depends, HTTPException
from loguru import logger

from app.api.dependencies import verify_service_token
from app.core.socketio_instance import sio

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

    await sio.emit("task_update", payload, room=target_room)
    logger.info(f"[HTTP bridge] Broadcast task_update to room '{target_room}'")
    return {"status": "broadcasted", "room": target_room}
