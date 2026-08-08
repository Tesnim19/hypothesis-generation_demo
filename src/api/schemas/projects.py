from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProjectsListResponse(BaseModel):
    projects: list[dict[str, Any]]


class ProjectDeleteMessage(BaseModel):
    message: str


class BulkDeleteProjectsRequest(BaseModel):
    project_ids: list[str] = Field(..., min_length=1)


class BulkDeleteProjectsOkResponse(BaseModel):
    message: str
    deleted_count: int
    total_requested: int


class BulkDeleteProjectsPartialResponse(BaseModel):
    message: str
    deleted_count: int
    total_requested: int
    errors: Any | None = None


class AnalysisPipelineStartResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    project_id: str
    file_id: str
    message: str
    sample_size: int | None = None
    sample_size_source: str | None = None
    sample_size_message: str | None = None
    sample_size_is_user_provided: bool | None = None
    sample_size_editable: bool | None = None
    sample_size_prefill: int | None = None
