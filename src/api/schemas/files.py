from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class UserFileItem(BaseModel):
    id: str | None = None
    display_name: str | None = None
    filename: str | None = None
    file_size: int = 0
    file_size_mb: float
    record_count: int | None = None
    upload_date: str | datetime | None = None
    source: str = Field(default="user_upload")


class UserFilesResponse(BaseModel):
    files: list[UserFileItem]
    total_files: int
