"""Kite login flow (spec 2.1, 14). The login is always manual; nothing is automated.

The api key, secret and daily access token live in the macOS Keychain via ``keyring``.
A tiny HTTP server bound to 127.0.0.1 receives the redirect carrying ``request_token``.
"""

import contextlib
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Protocol
from urllib.parse import parse_qs, urlparse

import keyring

from tradepartner.calendar import IST, utc_now

SERVICE = "tradepartner"
_KEYS = ("api_key", "api_secret", "access_token", "token_date")


class LoginError(Exception):
    pass


class KiteSessionClient(Protocol):
    def login_url(self) -> str: ...
    def generate_session(self, request_token: str, api_secret: str) -> dict[str, Any]: ...


def save_credentials(api_key: str, api_secret: str) -> None:
    keyring.set_password(SERVICE, "api_key", api_key)
    keyring.set_password(SERVICE, "api_secret", api_secret)


def load_credentials() -> tuple[str, str] | None:
    key = keyring.get_password(SERVICE, "api_key")
    secret = keyring.get_password(SERVICE, "api_secret")
    if key and secret:
        return key, secret
    return None


def save_access_token(token: str, on: date | None = None) -> None:
    keyring.set_password(SERVICE, "access_token", token)
    keyring.set_password(
        SERVICE, "token_date", (on or utc_now().astimezone(IST).date()).isoformat()
    )


def load_access_token(today: date | None = None) -> str | None:
    """Returns today's token, or None if missing or stale. Kite tokens expire daily."""
    today = today or utc_now().astimezone(IST).date()
    token = keyring.get_password(SERVICE, "access_token")
    stored = keyring.get_password(SERVICE, "token_date")
    if token and stored == today.isoformat():
        return token
    return None


def clear_access_token() -> None:
    for key in ("access_token", "token_date"):
        with contextlib.suppress(keyring.errors.PasswordDeleteError):
            keyring.delete_password(SERVICE, key)


def wait_for_request_token(port: int, timeout_sec: float = 300.0) -> str:
    """Serve one redirect on 127.0.0.1 and return the request_token."""
    result: dict[str, str] = {}
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            qs = parse_qs(urlparse(self.path).query)
            if qs.get("status", ["success"])[0] == "success" and "request_token" in qs:
                result["token"] = qs["request_token"][0]
                body = b"Login complete. You can close this tab."
                self.send_response(200)
            else:
                body = b"Login failed."
                self.send_response(400)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body)
            done.set()

        def log_message(self, format: str, *args: object) -> None:
            return  # never log URLs: they carry the request token

    server = HTTPServer(("127.0.0.1", port), Handler)
    server.timeout = 0.2
    deadline = timeout_sec
    try:
        waited = 0.0
        while not done.is_set() and waited < deadline:
            server.handle_request()
            waited += 0.2
    finally:
        server.server_close()
    if "token" not in result:
        raise LoginError("did not receive a request_token before timeout")
    return result["token"]


def complete_login(client: KiteSessionClient, api_secret: str, request_token: str) -> str:
    data = client.generate_session(request_token, api_secret)
    token = data.get("access_token")
    if not isinstance(token, str) or not token:
        raise LoginError("Kite did not return an access_token")
    save_access_token(token)
    return token
