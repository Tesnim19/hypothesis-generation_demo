from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HypothesisGraphResponse(BaseModel):
    id: str
    summary: Any | None = None
    graph: Any | None = None


class HypothesisChatResponse(BaseModel):
    response: Any | None = None


class BulkDeleteHypothesesRequest(BaseModel):
    hypothesis_ids: list[str] = Field(..., min_length=1)
