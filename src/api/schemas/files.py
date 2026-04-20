from __future__ import annotations

from pydantic import BaseModel


class UserFilesResponse(BaseModel):
    files: list[dict]
    total_files: int
