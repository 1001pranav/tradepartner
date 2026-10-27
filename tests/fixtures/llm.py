"""Mock Ollama (no network): scripted replies served through httpx.MockTransport."""

import json
from collections.abc import Callable

import httpx

from tradepartner.llm.client import OllamaClient

GOOD = {
    "symbol": "INFY", "relevant": True, "relevance": 0.9, "event_type": "earnings",
    "sentiment": 0.6, "materiality": 0.8, "expected_horizon": "days", "confidence": 0.85,
    "summary": "Infosys reported quarterly profit above market expectations.",
}  # fmt: skip


def chat_reply(content: str | dict[str, object]) -> httpx.Response:
    text = content if isinstance(content, str) else json.dumps(content)
    return httpx.Response(200, json={"message": {"role": "assistant", "content": text}})


class FakeOllama:
    """``script`` entries are dicts/str (a chat reply), int (an HTTP error status) or a callable."""

    def __init__(self, script: list[object] | None = None, default: object | None = None) -> None:
        self.script = list(script or [])
        self.default = GOOD if default is None else default
        self.requests: list[dict[str, object]] = []
        self.digest = "sha256:abc123"
        self.up = True

    def handler(self, request: httpx.Request) -> httpx.Response:
        if not self.up:
            raise httpx.ConnectError("down")
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "m", "digest": self.digest}]})
        body = json.loads(request.content)
        self.requests.append(body)
        item = self.script.pop(0) if self.script else self.default
        if isinstance(item, int):
            return httpx.Response(item, json={"error": "boom"})
        if callable(item):
            return chat_reply(item(body))  # type: ignore[arg-type]
        return chat_reply(item)  # type: ignore[arg-type]

    def client(
        self, model: str = "m", retries: int = 1, sleep: Callable[[float], None] = lambda s: None
    ) -> OllamaClient:
        http = httpx.Client(base_url="http://test", transport=httpx.MockTransport(self.handler))
        return OllamaClient(model, client=http, retries=retries, sleep=sleep)
