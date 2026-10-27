"""Save a command's printed result as an HTML report so it also shows up in the UI."""

from datetime import datetime
from html import escape
from pathlib import Path

from tradepartner.paths import reports_dir


def save_text_report(
    name: str, title: str, lines: list[str], directory: Path | None = None
) -> Path:
    out = directory or reports_dir()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}-{datetime.now():%Y%m%d-%H%M%S}.html"
    body = escape("\n".join(lines))
    path.write_text(
        f"<!doctype html><meta charset=utf-8><title>{escape(title)}</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:1000px}"
        "pre{background:#f6f6f6;padding:1rem;border-radius:6px;font-size:13px;overflow-x:auto}</style>"
        f"<h1>{escape(title)}</h1><pre>{body}</pre>"
    )
    return path
