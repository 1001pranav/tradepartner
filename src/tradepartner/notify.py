"""Notifications: macOS notification plus optional Telegram (spec 4, 14). Never logs secrets."""

import subprocess
from collections.abc import Callable
from typing import Protocol

import httpx
import keyring
import structlog

log = structlog.get_logger()
TELEGRAM_SERVICE = "tradepartner"


class Notifier(Protocol):
    def __call__(self, title: str, message: str) -> None: ...


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def macos_notify(title: str, message: str, runner: Callable[..., object] = subprocess.run) -> None:
    script = f'display notification "{_escape(message)}" with title "{_escape(title)}"'
    try:
        runner(["osascript", "-e", script], check=False, timeout=5)
    except (OSError, subprocess.SubprocessError):
        log.warning("notify_failed", channel="macos")


def telegram_notify(title: str, message: str, client: httpx.Client | None = None) -> bool:
    """Optional. Token and chat id live in the Keychain; returns False if not configured."""
    token = keyring.get_password(TELEGRAM_SERVICE, "telegram_token")
    chat = keyring.get_password(TELEGRAM_SERVICE, "telegram_chat_id")
    if not (token and chat):
        return False
    c = client or httpx.Client(timeout=10)
    try:
        c.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": f"{title}\n{message}"},
        )
    except httpx.HTTPError:
        log.warning("notify_failed", channel="telegram")
        return False
    return True


def notify(title: str, message: str) -> None:
    log.info("notification", title=title, message=message)
    macos_notify(title, message)
    telegram_notify(title, message)
