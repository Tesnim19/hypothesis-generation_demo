from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class ProjectsListResponse(BaseModel):
    projects: list[dict]


class ProjectDeleteMessage(BaseModel):
    message: str


class BulkDeleteProjectsRequest(BaseModel):
    project_ids: list | None = None

    @model_validator(mode="after")
    def project_ids_present(self) -> BulkDeleteProjectsRequest:
        v = self.project_ids
        if v is None:
            raise ValueError("project_ids is required in request body")
        if not isinstance(v, list):
            raise ValueError("project_ids must be a list")
        if not v:
            raise ValueError("project_ids list cannot be empty")
        return self


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
    """Scalar multipart fields for POST /analysis-pipeline (excludes gwas_file part)."""

    project_name: str | None = None
    phenotype: str | None = None
    population: str = Field(default="EUR")
    ref_genome: str | None = None

    max_workers: int = Field(default=3)
    is_uploaded: bool = Field(default=False)
    maf_threshold: float = Field(default=0.01)
    seed: int = Field(default=42)
    window: int = Field(default=2000)
    L: int = Field(default=-1)
    coverage: float = Field(default=0.95)
    min_abs_corr: float = Field(default=0.5)
    batch_size: int = Field(default=5)
    sample_size: int = Field(default=10000)

    @model_validator(mode="after")
    def validate_ranges_match_legacy_api(self) -> AnalysisPipelineFormFields:
        if self.population not in ("EUR", "AFR", "AMR", "EAS", "SAS"):
            raise ValueError("Population must be one of: EUR, AFR, AMR, EAS, SAS")
        if not (1 <= self.max_workers <= 16):
            raise ValueError("Max workers must be between 1-16")
        if not (0.001 <= self.maf_threshold <= 0.5):
            raise ValueError("MAF threshold must be between 0.001-0.5")
        if not (1 <= self.seed <= 999999):
            raise ValueError("Seed must be between 1-999999")
        if self.window > 10000:
            raise ValueError("Fine-mapping window shouldn't be greater than 10000 kb")
        if self.L != -1 and not (1 <= self.L <= 50):
            raise ValueError("L must be -1 (auto) or between 1-50")
        if not (0.5 <= self.coverage <= 0.999):
            raise ValueError("Coverage must be between 0.5-0.999")
        if not (0.5 <= self.min_abs_corr <= 1.0):
            raise ValueError("Min absolute correlation must be between 0.5-1.0")
        if not (1 <= self.batch_size <= 20):
            raise ValueError("Batch size must be between 1-20")
        return self


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
    """Read multipart form scalars; same field names as today (including gwas_file handled in route)."""
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
