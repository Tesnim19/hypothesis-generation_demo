from __future__ import annotations

from pydantic import BaseModel, Field


class GWASFileItem(BaseModel):
    id: str | None = None
    phenotype: str = ""
    phenotype_code: str | None = None
    filename: str | None = None
    sex: str | None = None
    source: str | None = None
    downloaded: bool = False
    download_count: int = 0
    url: str
    showcase_link: str = ""
    sample_size: int | None = None
    genome_build: str | None = None
    file_size_mb: float | None = None


class GwasFilesListResponse(BaseModel):
    gwas_files: list[GWASFileItem]
    total_files: int
    returned: int
    skip: int
    limit: int


class GwasDownloadUrlResponse(BaseModel):
    download_url: str
    cached: bool
