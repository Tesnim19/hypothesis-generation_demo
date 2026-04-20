from __future__ import annotations

from pydantic import BaseModel


class GwasFilesListResponse(BaseModel):
    gwas_files: list[dict]
    total_files: int
    returned: int
    skip: int
    limit: int


class GwasDownloadUrlResponse(BaseModel):
    download_url: str
    cached: bool
