"""structlog setup: JSON logs with secret redaction (spec 14)."""

import logging
import logging.handlers
import re
from collections.abc import MutableMapping
from typing import Any

import structlog

from tradepartner.paths import logs_dir

_SECRET_KEY = re.compile(
    r"(api[_-]?key|secret|token|password|passwd|totp|authorization|request[_-]?token)", re.I
)
REDACTED = "***REDACTED***"


def redact_secrets(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    for key in list(event_dict):
        if _SECRET_KEY.search(key):
            event_dict[key] = REDACTED
    return event_dict


def configure_logging(to_file: bool = True, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if to_file:
        logs_dir().mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.TimedRotatingFileHandler(
                logs_dir() / "tradepartner.log", when="midnight", backupCount=30, utc=True
            )
        )
    logging.basicConfig(format="%(message)s", level=level, handlers=handlers, force=True)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redact_secrets,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )
