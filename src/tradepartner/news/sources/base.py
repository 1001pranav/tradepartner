"""News source protocol: ``fetch(since) -> list[RawArticle]`` (spec 10.1)."""

import html
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class RawArticle:
    source: str
    url: str
    title: str
    body: str = ""
    published_at: datetime | None = None
    lang: str | None = None
    symbol_hint: str | None = None  # exchange announcements carry their own symbol
    isin_hint: str | None = None


class NewsSource(Protocol):
    name: str

    def fetch(self, since: datetime) -> list[RawArticle]: ...


def clean_text(text: str | None) -> str:
    """Strip markup and collapse whitespace."""
    if not text:
        return ""
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", text))).strip()
