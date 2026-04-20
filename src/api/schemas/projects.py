from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ProjectSummary(BaseModel):
    id: str
    name: str
    phenotype: str = ""
    created_at: datetime | str | None = None
    gwas_file: str
    gwas_records_count: int | None = None
    status: str
    population: str | None = None
    ref_genome: str | None = None
    total_credible_sets_count: int = 0
    total_variants_count: int = 0
    hypothesis_count: int = 0


class ProjectsListResponse(BaseModel):
    projects: list[ProjectSummary]


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
    status: str
    project_id: str
    file_id: str
    message: str


class AnalysisPipelineFormFields(BaseModel):

    project_name: str | None = None
    phenotype: str | None = None
    population: Literal["EUR", "AFR", "AMR", "EAS", "SAS"] = Field(default="EUR")
    ref_genome: str | None = None

    max_workers: int = Field(default=3, ge=1, le=16)
    is_uploaded: bool = Field(default=False)
    maf_threshold: float = Field(default=0.01, ge=0.001, le=0.5)
    seed: int = Field(default=42, ge=1, le=999999)
    window: int = Field(default=2000, le=10000)
    L: int = Field(default=-1)
    coverage: float = Field(default=0.95, ge=0.5, le=0.999)
    min_abs_corr: float = Field(default=0.5, ge=0.5, le=1.0)
    batch_size: int = Field(default=5, ge=1, le=20)
    sample_size: int = Field(default=10000)

    @field_validator("L")
    @classmethod
    def validate_l(cls, v: int) -> int:
        if v == -1 or 1 <= v <= 50:
            return v
        raise ValueError("L must be -1 (auto) or between 1-50")


def _form_scalar(form: Any, key: str, default: Any = None) -> Any:
    v = form.get(key)
    if v is None:
        return default
    if hasattr(v, "read") and hasattr(v, "filename"):
        return default
    return v


def _as_int(form: Any, key: str, default: int) -> int:
    v = _form_scalar(form, key, None)
    if v is None or v == "":
        return default
    try:
        return int(str(v))
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {key}") from exc


def _as_float(form: Any, key: str, default: float) -> float:
    v = _form_scalar(form, key, None)
    if v is None or v == "":
        return default
    try:
        return float(str(v))
    except ValueError as exc:
        raise ValueError(f"Invalid float for {key}") from exc


def parse_analysis_pipeline_form_fields(form: Any) -> AnalysisPipelineFormFields:
    uploaded_raw = _form_scalar(form, "is_uploaded", "false")
    is_uploaded = (
        str(uploaded_raw).lower() == "true" if uploaded_raw not in (None, "") else False
    )

    data = {
        "project_name": _form_scalar(form, "project_name"),
        "phenotype": _form_scalar(form, "phenotype"),
        "population": _form_scalar(form, "population") or "EUR",
        "ref_genome": _form_scalar(form, "ref_genome"),
        "max_workers": _as_int(form, "max_workers", 3),
        "is_uploaded": is_uploaded,
        "maf_threshold": _as_float(form, "maf_threshold", 0.01),
        "seed": _as_int(form, "seed", 42),
        "window": _as_int(form, "window", 2000),
        "L": _as_int(form, "L", -1),
        "coverage": _as_float(form, "coverage", 0.95),
        "min_abs_corr": _as_float(form, "min_abs_corr", 0.5),
        "batch_size": _as_int(form, "batch_size", 5),
        "sample_size": _as_int(form, "sample_size", 10000),
    }
    return AnalysisPipelineFormFields.model_validate(data)
