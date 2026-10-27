"""Ollama wrapper: JSON-schema constrained output, temperature 0, fixed seed, retries and
timeouts (spec 4, 10.2). The LLM is an analyst only; it never decides prices or whether to trade.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
import structlog
from pydantic import BaseModel, ValidationError

log = structlog.get_logger()
M = TypeVar("M", bound=BaseModel)

DEFAULT_URL = "http://127.0.0.1:11434"
SEED = 42


class LLMUnavailableError(Exception):
    """Ollama cannot be reached or kept failing."""


@dataclass(frozen=True)
class LLMResult:
    parsed: BaseModel | None
    raw: str
    valid: bool
    error: str | None = None


class OllamaClient:
    def __init__(
        self,
        model: str,
        num_ctx: int = 4096,
        keep_alive: str = "5m",
        base_url: str = DEFAULT_URL,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
        retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.retries = retries
        self._sleep = sleep
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout)

    # ---- plumbing ----

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = self._client.post(path, json=payload)
                r.raise_for_status()
                return r.json()  # type: ignore[no-any-return]
            except (httpx.HTTPError, ValueError) as exc:
                last = exc
                log.warning("ollama_retry", attempt=attempt, error=type(exc).__name__)
                if attempt < self.retries:
                    self._sleep(2.0**attempt)
        raise LLMUnavailableError(f"Ollama request failed: {last}") from last

    def health(self) -> bool:
        try:
            return self._client.get("/api/tags").status_code == 200
        except httpx.HTTPError:
            return False

    def model_digest(self) -> str | None:
        """Digest of the loaded model, stored with every score (spec 6.2)."""
        try:
            r = self._client.get("/api/tags")
            r.raise_for_status()
            for m in r.json().get("models", []):
                if m.get("name") == self.model or m.get("model") == self.model:
                    return str(m.get("digest"))
        except (httpx.HTTPError, ValueError):
            return None
        return None

    # ---- structured generation ----

    def generate(self, prompt: str, schema_model: type[M]) -> LLMResult:
        """One call, schema-constrained, validated. Invalid output is reported, not raised."""
        body = {
            "model": self.model,
            "stream": False,
            "format": schema_model.model_json_schema(),
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0, "seed": SEED, "num_ctx": self.num_ctx},
            "keep_alive": self.keep_alive,
        }
        data = self._post("/api/chat", body)
        raw = str(data.get("message", {}).get("content", ""))
        try:
            return LLMResult(schema_model.model_validate_json(raw), raw, True)
        except ValidationError as exc:
            return LLMResult(
                None, raw, False, str(exc.errors()[0]["msg"]) if exc.errors() else "invalid"
            )

    def generate_with_retry(self, prompt: str, schema_model: type[M]) -> LLMResult:
        """Spec 10.2: one retry on invalid output, then discard (the caller logs it)."""
        first = self.generate(prompt, schema_model)
        if first.valid:
            return first
        log.info("llm_invalid_output_retry", error=first.error)
        return self.generate(prompt, schema_model)
