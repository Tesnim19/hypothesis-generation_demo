"""OpenAPI metadata, tag definitions, and security scheme configuration."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from src.api.openapi_websocket import build_websocket_api_docs

DEFAULT_PUBLIC_HOST = "dev.rejuve.bio"
DEFAULT_PUBLIC_PORT = "5008"
DEFAULT_PUBLIC_SCHEME = "https"

OPENAPI_TAGS: list[dict[str, str]] = [
    {
        "name": "health",
        "description": "Liveness and readiness checks.",
    },
    {
        "name": "projects",
        "description": (
            "Project CRUD, bulk delete, and GWAS analysis pipeline kickoff "
            "(harmonization, fine-mapping)."
        ),
    },
    {
        "name": "hypothesis",
        "description": (
            "Hypothesis generation from enrichment results, retrieval, deletion, "
            "and LLM chat over a hypothesis."
        ),
    },
    {
        "name": "enrichment",
        "description": (
            "Gene-set enrichment analysis for a project locus. Long-running jobs "
            "return 202; subscribe via Socket.IO for progress."
        ),
    },
    {
        "name": "analysis",
        "description": "Post-GWAS analysis outputs such as credible sets.",
    },
    {
        "name": "gwas_library",
        "description": (
            "Public GWAS catalog (sources, file listing, download). "
            "Sample-size metadata requires authentication."
        ),
    },
    {
        "name": "phenotypes",
        "description": "Phenotype lookup and creation (public endpoints).",
    },
    {
        "name": "user_files",
        "description": "User-uploaded GWAS and related files.",
    },
]

_AUTH_AND_PUBLIC_DOCS = """\
## Authentication

Most endpoints require a **JWT Bearer token**. Click **Authorize** above and enter:

```
<your-jwt-token>
```

Or send the header:

```
Authorization: Bearer <JWT>
```

The token must be signed with HS256 using `JWT_SECRET` and include a `user_id` claim.
Optional claims: `email`.

**Public endpoints** (no token):

| Method | Path |
|--------|------|
| GET | `/health` |
| GET | `/gwas-files/sources` |
| GET | `/gwas-files` |
| GET | `/gwas-files/download/{file_id}` |
| GET | `/phenotypes` |
| POST | `/phenotypes` |
"""

_INTERACTIVE_DOCS = """\
## Interactive documentation

This service exposes Swagger directly on its API port (same pattern as the annotation
service at `:5005/docs`).

| Resource | Dev URL |
|----------|---------|
| Swagger UI | `https://dev.rejuve.bio:5008/docs` |
| ReDoc | `https://dev.rejuve.bio:5008/redoc` |
| OpenAPI schema | `https://dev.rejuve.bio:5008/openapi.json` |

Use the **Servers** dropdown above when trying requests from another host or port.
"""

# Operations that do not require JWT (method lowercase, OpenAPI path template).
PUBLIC_OPERATIONS: set[tuple[str, str]] = {
    ("get", "/health"),
    ("get", "/gwas-files/sources"),
    ("get", "/gwas-files"),
    ("get", "/gwas-files/download/{file_id}"),
    ("get", "/phenotypes"),
    ("post", "/phenotypes"),
}


def get_api_servers() -> list[dict[str, str]]:
    """OpenAPI server list for Swagger Try-it-out (direct API port, not nginx)."""
    explicit = os.getenv("OPENAPI_SERVER_URL", "").strip().rstrip("/")
    if explicit:
        primary = explicit
    else:
        host = os.getenv("OPENAPI_SERVER_HOST", DEFAULT_PUBLIC_HOST)
        port = os.getenv("OPENAPI_SERVER_PORT", DEFAULT_PUBLIC_PORT).strip()
        scheme = os.getenv("OPENAPI_SERVER_SCHEME", DEFAULT_PUBLIC_SCHEME)
        primary = f"{scheme}://{host}:{port}" if port else f"{scheme}://{host}"

    servers: list[dict[str, str]] = [
        {
            "url": primary,
            "description": "Hypothesis Generation API (host port → container 5000)",
        },
    ]

    local = os.getenv("OPENAPI_LOCAL_SERVER_URL", "").strip().rstrip("/")
    if local and local != primary:
        servers.append({"url": local, "description": "Local development override"})

    return servers


def _http_origin_from_server_url(url: str) -> str:
    return url.rstrip("/")


def _ws_origin_from_http(url: str) -> str:
    url = url.rstrip("/")
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :]
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :]
    return url


def build_api_description() -> str:
    """Compose the full OpenAPI description (auth + WebSocket + doc links)."""
    http_origin = _http_origin_from_server_url(get_api_servers()[0]["url"])
    ws_origin = _ws_origin_from_http(http_origin)
    return (
        "Hypothesis Generation API for GWAS-driven gene hypothesis and enrichment workflows.\n\n"
        "Integrates enrichment analysis, knowledge-graph reasoning (Prolog), and LLM summarization "
        "to produce hypothesis graphs and narratives from genetic association data.\n\n"
        + _AUTH_AND_PUBLIC_DOCS
        + "\n"
        + build_websocket_api_docs(ws_origin)
        + "\n"
        + _INTERACTIVE_DOCS
    )


def configure_openapi(app: FastAPI) -> None:
    """Register a custom OpenAPI generator with Bearer JWT security metadata."""

    def custom_openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema

        schema = get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            description=app.description,
            routes=app.routes,
            tags=app.openapi_tags,
            servers=app.servers,
        )

        components = schema.setdefault("components", {})
        components.setdefault("securitySchemes", {})["BearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": (
                "JWT access token with a required `user_id` claim. "
                "Signed with HS256 using the server `JWT_SECRET`."
            ),
        }

        for path, path_item in schema.get("paths", {}).items():
            for method, operation in path_item.items():
                if method.startswith("x-") or method not in {
                    "get",
                    "post",
                    "put",
                    "patch",
                    "delete",
                    "head",
                    "options",
                }:
                    continue
                if (method, path) not in PUBLIC_OPERATIONS:
                    operation["security"] = [{"BearerAuth": []}]

        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi  # type: ignore[method-assign]
