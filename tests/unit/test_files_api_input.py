import asyncio
import gzip
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests
from bson import ObjectId
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import dependencies as deps
from src.api.auth import get_current_user_email, get_current_user_id
from src.api.routes import gwas as gwas_routes
from src.api.routes import projects as project_routes
from src import run_deployment
from src.services import files_api
from src.tasks import project as project_tasks

FILE_ID = "6f1c2a7e-0b9d-4c55-9a3e-2f0f8f6c1d11"


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("FILES_API_URL", "http://files.test")
    monkeypatch.setenv("FILES_API_SERVICE_TOKEN", "svc")


def _respond(monkeypatch, status=200, json=None, content=b""):
    calls = []

    def fake_request(method, url, headers=None, **kwargs):
        calls.append((method, url, headers))
        return SimpleNamespace(status_code=status, json=lambda: json, content=content)

    monkeypatch.setattr(files_api.requests, "request", fake_request)
    return calls


def test_disabled_returns_503(monkeypatch):
    monkeypatch.delenv("FILES_API_URL", raising=False)
    with pytest.raises(files_api.FilesApiError) as exc:
        files_api.get_file(FILE_ID, "Bearer u")
    assert exc.value.status_code == 503


@pytest.mark.parametrize(
    ("status", "expected"), [(401, 401), (403, 404), (404, 404), (409, 502), (500, 502)]
)
def test_service_errors_map_to_status(enabled, monkeypatch, status, expected):
    _respond(monkeypatch, status=status)
    with pytest.raises(files_api.FilesApiError) as exc:
        files_api.get_file(FILE_ID, "Bearer u")
    assert exc.value.status_code == expected


def test_unreachable_service_returns_502(enabled, monkeypatch):
    def boom(*args, **kwargs):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(files_api.requests, "request", boom)
    with pytest.raises(files_api.FilesApiError) as exc:
        files_api.get_file(FILE_ID, "Bearer u")
    assert exc.value.status_code == 502


def test_download_url_uses_service_credential(enabled, monkeypatch):
    calls = _respond(monkeypatch, json={"url": "http://minio.test/obj?sig"})
    assert files_api.download_url(FILE_ID, "user-1") == "http://minio.test/obj?sig"
    method, url, headers = calls[0]
    assert (method, url) == ("POST", f"http://files.test/v1/files/{FILE_ID}/download-url")
    assert headers["Authorization"] == "Bearer svc"


@pytest.mark.parametrize(
    ("meta", "status"),
    [
        ({"name": "a.tsv.gz", "state": "importing"}, 409),
        ({"name": "a.pdf", "state": "ready"}, 400),
    ],
)
def test_register_rejects_unusable_files(enabled, monkeypatch, meta, status):
    _respond(monkeypatch, json=meta)
    with pytest.raises(files_api.FilesApiError) as exc:
        files_api.register_file(FILE_ID, "Bearer u", "user-1", MagicMock())
    assert exc.value.status_code == status


def test_register_creates_metadata_once(enabled, monkeypatch):
    _respond(monkeypatch, json={"name": "a.tsv.gz", "state": "ready", "size": 10})
    files = MagicMock()
    files.find_by_external_id.return_value = None
    files.create_file_metadata.return_value = "doc-1"

    doc = files_api.register_file(FILE_ID, "Bearer u", "user-1", files)
    assert doc["_id"] == "doc-1"
    kwargs = files.create_file_metadata.call_args.kwargs
    assert kwargs["external_file_id"] == FILE_ID and kwargs["source"] == "files_api"

    files.find_by_external_id.return_value = {"_id": "doc-1"}
    files.create_file_metadata.reset_mock()
    assert files_api.register_file(FILE_ID, "Bearer u", "user-1", files)["_id"] == "doc-1"
    files.create_file_metadata.assert_not_called()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    app = FastAPI()
    app.include_router(project_routes.router)
    projects = MagicMock()
    projects.create_project.return_value = "project-1"
    app.dependency_overrides.update({
        get_current_user_id: lambda: "user-1",
        get_current_user_email: lambda: None,
        deps.get_project_handler: lambda: projects,
        deps.get_file_handler: lambda: MagicMock(),
        deps.get_config: lambda: MagicMock(),
        deps.get_storage: lambda: None,
        deps.get_gwas_library_handler: lambda: MagicMock(),
    })
    invoke = MagicMock()
    monkeypatch.setattr(project_routes, "invoke_analysis_pipeline_deployment", invoke)
    return TestClient(app), invoke


FORM = {"project_name": "p", "phenotype": "height"}


def test_route_rejects_file_id_with_gwas_file(client):
    http, _ = client
    resp = http.post("/analysis-pipeline", data={**FORM, "file_id": FILE_ID, "gwas_file": "lib-1"})
    assert resp.status_code == 400


def test_route_starts_pipeline_from_file_id(client, monkeypatch):
    http, invoke = client
    register = MagicMock(return_value={"_id": "doc-1", "filename": "a.tsv.gz", "file_size": 10})
    monkeypatch.setattr(files_api, "register_file", register)

    resp = http.post(
        "/analysis-pipeline",
        data={**FORM, "file_id": FILE_ID},
        headers={"Authorization": "Bearer u"},
    )
    assert resp.status_code == 202, resp.text
    assert register.call_args.args[:3] == (FILE_ID, "Bearer u", "user-1")
    kwargs = invoke.call_args.kwargs
    assert kwargs["file_external_id"] == FILE_ID
    assert kwargs["file_metadata_id"] == "doc-1"


def test_worker_streams_download_link(monkeypatch, tmp_path):
    files = MagicMock()
    files.get_file_metadata.return_value = {"filename": "height.tsv"}
    monkeypatch.setattr(project_tasks, "get_deps", lambda: {"files": files, "projects": MagicMock()})
    monkeypatch.setattr(project_tasks, "emit_analysis_update", lambda *a: None)
    monkeypatch.setattr(files_api, "download_url", lambda fid, uid: "http://minio.test/x?sig")

    def fake_download(dest_dir, storage, source_download_url=None):
        path = tmp_path / "x"
        path.write_text("SNP\tP\nrs1\t0.1\n")
        return str(path)

    monkeypatch.setattr(project_tasks, "download_gwas_source", fake_download)

    path = project_tasks.prepare_gwas_file_task.fn(
        "user-1", "project-1", "doc-1", "", external_file_id=FILE_ID, output_dir=str(tmp_path)
    )
    assert path == str(tmp_path / "height.tsv")
    files.update_file_metadata.assert_called_once()


@pytest.mark.parametrize(
    ("file_id", "local_doc", "requested"),
    [
        (FILE_ID, None, FILE_ID),
        (str(ObjectId()), {"external_file_id": FILE_ID}, FILE_ID),
    ],
)
def test_preview_reads_file_head(enabled, monkeypatch, file_id, local_doc, requested):
    head = gzip.compress(b"SNP\tP\tN\nrs1\t0.1\t1000\n")
    read = MagicMock(return_value=head)
    monkeypatch.setattr(files_api, "read_head", read)
    files = MagicMock()
    files.get_file_metadata.return_value = local_doc
    library = MagicMock()
    library.get_gwas_entry.return_value = None

    resp = asyncio.run(gwas_routes.get_sample_size_info(
        SimpleNamespace(headers={"authorization": "Bearer u"}),
        file_id, gwas_source="upload", current_user_id="user-1",
        gwas_library=library, files=files,
    ))
    assert read.call_args.args == (requested, "Bearer u")
    assert resp.sample_size_source == "file_has_n"


@pytest.mark.parametrize("external_id", [None, FILE_ID])
def test_external_id_sent_only_when_set(monkeypatch, external_id):
    run = MagicMock()
    monkeypatch.setattr(run_deployment, "run_deployment", run)
    run_deployment.invoke_analysis_pipeline_deployment("user-1", "project-1", file_external_id=external_id)
    params = run.call_args.kwargs["parameters"]
    assert params.get("file_external_id") == external_id
    assert ("file_external_id" in params) == bool(external_id)
