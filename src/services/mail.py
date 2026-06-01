from __future__ import annotations

import os

from fastapi_mail import ConnectionConfig, FastMail, MessageSchema
from loguru import logger


def _build_fast_mail() -> FastMail | None:
    """Build a FastMail instance from environment variables. Returns None if MAIL_SERVER is not set."""
    mail_server = os.getenv("MAIL_SERVER", "")
    if not mail_server:
        return None
    cfg = ConnectionConfig(
        MAIL_USERNAME=os.getenv("MAIL_USERNAME", ""),
        MAIL_PASSWORD=os.getenv("MAIL_PASSWORD", ""),
        MAIL_FROM=os.getenv("MAIL_FROM", ""),
        MAIL_PORT=int(os.getenv("MAIL_PORT", "587")),
        MAIL_SERVER=mail_server,
        MAIL_STARTTLS=os.getenv("MAIL_TLS", "true").lower() == "true",
        MAIL_SSL_TLS=os.getenv("MAIL_SSL", "false").lower() == "true",
        USE_CREDENTIALS=True,
    )
    return FastMail(cfg)


# Module-level instance — set by init_mail() in the API process.
# Worker processes that never call init_mail() fall back to _build_fast_mail().
_fast_mail: FastMail | None = None


def init_mail(
    mail_username: str,
    mail_password: str,
    mail_from: str,
    mail_server: str,
    mail_port: int = 587,
    mail_tls: bool = True,
    mail_ssl: bool = False,
) -> None:
    global _fast_mail
    if not mail_server:
        logger.warning("init_mail called with empty MAIL_SERVER — email disabled")
        return
    cfg = ConnectionConfig(
        MAIL_USERNAME=mail_username,
        MAIL_PASSWORD=mail_password,
        MAIL_FROM=mail_from,
        MAIL_PORT=mail_port,
        MAIL_SERVER=mail_server,
        MAIL_STARTTLS=mail_tls,
        MAIL_SSL_TLS=mail_ssl,
        USE_CREDENTIALS=True,
    )
    _fast_mail = FastMail(cfg)
    logger.info("Mail service initialized")


async def send_email(subject: str, recipients: list[str], body: str) -> None:
    """
    Send a plain-text email.

    Safe to call from both the API process (where init_mail was called)
    and from Prefect/Dask worker processes (where it was not — falls back
    to building FastMail from env vars on demand).
    """
    if not recipients:
        logger.warning("send_email called with empty recipients list — skipping")
        return

    fm = _fast_mail or _build_fast_mail()
    if fm is None:
        logger.warning("MAIL_SERVER not configured — skipping email: %s", subject)
        return

    try:
        msg = MessageSchema(
            subject=subject,
            recipients=recipients,
            body=body,
            subtype="plain",
        )
        await fm.send_message(msg)
        logger.info("Email sent to {} | subject: {}", recipients, subject)
    except Exception as exc:
        logger.error("Failed to send email to {}: {}", recipients, exc)