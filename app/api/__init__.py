"""API routes and Socket.IO handlers."""
from app.api.dependencies import set_deps
from app.api.routes import router

# Register Socket.IO event handlers (import side-effect)
from app.api import socket_handlers  # noqa: F401


def init_deps(deps: dict) -> None:
    """Called at startup to inject shared dependencies."""
    set_deps(deps)


__all__ = ["router", "init_deps"]
