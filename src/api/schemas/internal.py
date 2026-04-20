from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InternalTaskUpdatePayload(BaseModel):
    """Prefect → HTTP bridge: broadcast payload to a Socket.IO room."""

    model_config = ConfigDict(extra="allow")

    target_room: str = Field(..., min_length=1, description="Socket.IO room name")
    event: str = Field(default="task_update")


class InternalTaskUpdateResponse(BaseModel):
    status: str
    room: str
    event: str


class HealthResponse(BaseModel):
    status: str
