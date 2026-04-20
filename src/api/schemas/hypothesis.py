from __future__ import annotations

from typing import Any

from pydantic import BaseModel, model_validator


class HypothesisGraphResponse(BaseModel):
    id: str
    summary: Any | None = None
    graph: Any | None = None


class HypothesisChatResponse(BaseModel):
    response: Any | None = None


class BulkDeleteHypothesesRequest(BaseModel):
    hypothesis_ids: list | None = None

    @model_validator(mode="after")
    def hypothesis_ids_present(self) -> BulkDeleteHypothesesRequest:
        v = self.hypothesis_ids
        if v is None:
            raise ValueError("hypothesis_ids is required in request body")
        if not isinstance(v, list):
            raise ValueError("hypothesis_ids must be a list")
        if not v:
            raise ValueError("hypothesis_ids list cannot be empty")
        return self


class HypothesisChatForm(BaseModel):
    query: str | None = None
    hypothesis_id: str | None = None

