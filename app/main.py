"""FastAPI + Socket.IO application entry point (was app.py at project root)."""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager

import socketio as python_socketio
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core import setup_logging, sio
from app.core.deps import create_dependencies
from app.core.config import get_settings, Settings
from app.api import router


def create_app(config: Settings) -> python_socketio.ASGIApp:
    load_dotenv()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # instantiate dependencies and store in app state for access in routes
        create_dependencies(config)
        logger.info("Application dependencies initialized")
        yield
        logger.info("Application shutting down")

    fastapi_app = FastAPI(
        title="Hypothesis Generation API",
        version="0.1.0",
        lifespan=lifespan,
    )

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    fastapi_app.include_router(router)

    combined_app = python_socketio.ASGIApp(sio, fastapi_app)
    return combined_app


def main() -> None:
    config = get_settings()

    if not all([config.ensembl_hgnc_map, config.hgnc_ensembl_map, config.go_map]):
        raise ValueError(
            "Missing required configuration: ensembl_hgnc_map, hgnc_ensembl_map, go_map"
        )

    setup_logging(log_level="INFO")
    logger.info(f"Starting FastAPI + Socket.IO on {config.host}:{config.port}")

    app = create_app(config)

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
