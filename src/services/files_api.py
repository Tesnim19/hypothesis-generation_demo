"""Client for the platform file service (``/v1/files``)."""
from __future__ import annotations

import os

import requests

TIMEOUT = 30
HEAD_BYTES = 1024 * 1024


class FilesApiError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _base_url() -> str:
    return os.getenv("FILES_API_URL", "").rstrip("/")


def enabled() -> bool:
    return bool(_base_url() and os.getenv("FILES_API_SERVICE_TOKEN"))


def _service_headers(user_id: str) -> dict:
    # Placeholder until the file service defines service-to-service auth.
    return {
        "Authorization": f"Bearer {os.environ['FILES_API_SERVICE_TOKEN']}",
        "X-On-Behalf-Of": user_id,
    }


def _call(method: str, path: str, headers: dict, **kwargs) -> requests.Response:
    if not enabled():
        raise FilesApiError(503, "File service is not configured")
    try:
        resp = requests.request(
            method, f"{_base_url()}{path}", headers=headers, timeout=TIMEOUT, **kwargs
        )
    except requests.RequestException as exc:
        raise FilesApiError(502, f"File service unreachable: {exc}")
    if resp.status_code == 404:
        raise FilesApiError(404, "File not found")
    if resp.status_code >= 400:
        raise FilesApiError(502, f"File service error {resp.status_code}")
    return resp


def get_file(file_id: str, authorization: str) -> dict:
    return _call("GET", f"/v1/files/{file_id}", {"Authorization": authorization}).json()


def read_head(file_id: str, authorization: str) -> bytes:
    headers = {"Authorization": authorization, "Range": f"bytes=0-{HEAD_BYTES - 1}"}
    return _call("GET", f"/v1/files/{file_id}/content", headers).content


def download_url(file_id: str, user_id: str) -> str:
    resp = _call("POST", f"/v1/files/{file_id}/download-url", _service_headers(user_id))
    return resp.json()["url"]


def register_file(file_id: str, authorization: str, user_id: str, files) -> dict:
    """Check a file-service file is usable and return its local file_metadata doc."""
    from src.utils import allowed_file

    meta = get_file(file_id, authorization)
    if meta.get("state") != "ready":
        raise FilesApiError(409, f"File is not ready (state: {meta.get('state')})")
    name = meta.get("name") or ""
    if not allowed_file(name):
        raise FilesApiError(
            400, "Invalid file format. Supported: .tsv, .txt, .csv, .gz, .bgz"
        )

    existing = files.find_by_external_id(user_id, file_id)
    if existing:
        return existing
    doc = {
        "user_id": user_id,
        "filename": name,
        "original_filename": name,
        "file_path": "",
        "file_type": "gwas",
        "file_size": meta.get("size", 0),
        "source": "files_api",
        "external_file_id": file_id,
    }
    doc["_id"] = files.create_file_metadata(**doc)
    return doc
