"""Auth and dependency injection (was auth.py + deps from api.py)."""
from __future__ import annotations

import logging
import os
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

JWT_SECRET = os.getenv("JWT_SECRET")
_bearer = HTTPBearer()

# Injected at startup by main.py
_deps: dict[str, Any] = {}


def set_deps(deps: dict[str, Any]) -> None:
    global _deps
    _deps.update(deps)


def get_deps() -> dict[str, Any]:
    return _deps


def _decode(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])


async def get_current_user_id(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    try:
        data = _decode(creds.credentials)
        return str(data["user_id"])
    except Exception as exc:
        logging.error(f"Token decode error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token is invalid!",
        )


async def verify_service_token(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
) -> None:
    try:
        data = _decode(creds.credentials)
        if data.get("service") != "prefect":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Service token required",
            )
    except HTTPException:
        raise
    except Exception as exc:
        logging.error(f"Service token error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid service token",
        )


def extract_token_from_environ(environ: dict) -> str | None:
    auth_header = environ.get("HTTP_AUTHORIZATION", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    query_string = environ.get("QUERY_STRING", "")
    for param in query_string.split("&"):
        if param.startswith("token="):
            return param[6:]
    return None
