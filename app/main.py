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

from app.core.config import Config, create_dependencies
from app.core.logging import setup_logging
from app.core.socket import sio
from app.core.status_tracker import StatusTracker


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FastAPI + Socket.IO Server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--embedding-model", type=str, default="w601sxs/b1ade-embed-kd")
    parser.add_argument("--swipl-host", type=str, default="localhost")
    parser.add_argument("--swipl-port", type=int, default=4242)
    parser.add_argument("--ensembl-hgnc-map", type=str, required=True)
    parser.add_argument("--hgnc-ensembl-map", type=str, required=True)
    parser.add_argument("--go-map", type=str, required=True)
    return parser.parse_args()


def create_app(config: Config) -> python_socketio.ASGIApp:
    load_dotenv()

    from app.api import init_deps, router

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        deps = create_dependencies(config)
        status_tracker_instance = StatusTracker()
        status_tracker_instance.initialize(deps["tasks"])
        init_deps(deps)
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
    args = parse_arguments()
    config = Config.from_args(args)

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
