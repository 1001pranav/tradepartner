"""Safe editing of config files from the UI.

Only a whitelist of files can be edited. ``risk.yaml`` (limits, capital) and ``app.yaml`` (broker,
execution mode, UI binding) are deliberately NOT editable here: loosening risk or switching to the
live broker is done by hand in the file, as the spec requires. Every save is validated, backed up
and written to the audit log.
"""

import csv
import difflib
import io
import shutil
import tempfile
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import yaml
from sqlalchemy.orm import Session

from tradepartner.db import audit
from tradepartner.paths import backups_dir, config_dir
from tradepartner.settings import ConfigError, StrategyConfig, load_settings

READ_ONLY = {
    "risk.yaml": "Risk limits and capital: edit by hand (loosening needs your explicit decision).",
    "app.yaml": "Broker, execution mode, UI binding and LLM: edit by hand.",
}


class ConfigEditError(ValueError):
    pass


def _csv_check(required: set[str], symbol_col: str | None = "symbol") -> Callable[[str], list[str]]:
    def check(text: str) -> list[str]:
        lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
        if not lines:
            return ["file is empty"]
        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        if set(reader.fieldnames or []) != required:
            return [f"columns must be exactly: {', '.join(sorted(required))}"]
        errs = []
        for n, row in enumerate(reader, start=2):
            if symbol_col and not (row.get(symbol_col) or "").strip():
                errs.append(f"row {n}: missing {symbol_col}")
        return errs

    return check


def _yaml_load(text: str) -> object:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigEditError(f"invalid YAML: {exc}") from exc


def _check_strategy(stem: str) -> Callable[[str], list[str]]:
    def check(text: str) -> list[str]:
        data = _yaml_load(text)
        if not isinstance(data, dict):
            return ["must be a mapping"]
        if data.get("name") != stem:
            return [f"name must stay '{stem}'"]
        try:
            StrategyConfig(**data)
        except Exception as exc:
            return [str(exc)]
        errs = []
        if "top_k" in data and not (isinstance(data["top_k"], int) and 1 <= data["top_k"] <= 20):
            errs.append("top_k must be an integer from 1 to 20")
        if "n_trials" in data and not (
            isinstance(data["n_trials"], int) and 0 <= data["n_trials"] <= 50
        ):
            errs.append("n_trials must be an integer from 0 to 50 (hard cap)")
        for k in ("min_prob",):
            if k in data and not (isinstance(data[k], (int, float)) and 0 <= data[k] <= 1):
                errs.append(f"{k} must be between 0 and 1")
        return errs

    return check


def _check_whole(name: str) -> Callable[[str], list[str]]:
    """Validate by loading the full settings from a temp copy with this file replaced."""

    def check(text: str) -> list[str]:
        _yaml_load(text)
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(config_dir(), tmp, dirs_exist_ok=True)
            (Path(tmp) / name).write_text(text)
            try:
                load_settings(Path(tmp))
            except ConfigError as exc:
                return [str(exc)]
        return []

    return check


def _check_holidays(text: str) -> list[str]:
    data = _yaml_load(text)
    if not isinstance(data, dict) or not {"holidays", "extra_trading_days"} <= set(data):
        return ["needs 'holidays' and 'extra_trading_days' lists"]
    errs = []
    for key in ("holidays", "extra_trading_days"):
        for item in data[key] or []:
            try:
                d = item if isinstance(item, date) else date.fromisoformat(str(item))
            except ValueError:
                errs.append(f"{key}: {item!r} is not YYYY-MM-DD")
                continue
            if key == "holidays" and d.weekday() >= 5:
                errs.append(f"holidays: {d} is a weekend (weekends are automatic)")
    return errs


def editable() -> dict[str, Callable[[str], list[str]]]:
    files: dict[str, Callable[[str], list[str]]] = {
        "costs.yaml": _check_whole("costs.yaml"),
        "tax.yaml": _check_whole("tax.yaml"),
        "holidays.yaml": _check_holidays,
        "universe/nifty100.csv": _csv_check({"symbol"}),
        "universe/excluded.csv": _csv_check({"symbol", "reason"}),
        "universe/sectors.csv": _csv_check({"symbol", "sector"}),
        "universe/aliases.csv": _csv_check({"alias", "symbol"}, "alias"),
        "asm_gsm.csv": _csv_check({"symbol"}),
    }
    for p in sorted((config_dir() / "strategies").glob("*.yaml")):
        files[f"strategies/{p.name}"] = _check_strategy(p.stem)
    return files


def _resolve(name: str) -> Path:
    if name not in editable():
        raise ConfigEditError(f"{name} is not editable from the UI")
    path = (config_dir() / name).resolve()
    if config_dir().resolve() not in path.parents:
        raise ConfigEditError("path escapes the config directory")
    return path


def read(name: str) -> str:
    return _resolve(name).read_text()


def validate(name: str, text: str) -> list[str]:
    _resolve(name)
    try:
        return editable()[name](text)
    except ConfigEditError as exc:
        return [str(exc)]


def diff(old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(old.splitlines(True), new.splitlines(True), "current", "edited")
    )


def save(name: str, text: str, session: Session | None = None, now: datetime | None = None) -> Path:
    """Validate, back up the old file, write the new one, audit it. Raises on any problem."""
    errors = validate(name, text)
    if errors:
        raise ConfigEditError("; ".join(errors))
    path = _resolve(name)
    when = now or datetime.now(UTC)
    old = path.read_text()
    if old == text:
        return path
    backup = backups_dir() / "config-edits" / f"{when:%Y%m%d-%H%M%S}-{name.replace('/', '_')}"
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text(old)
    path.write_text(text if text.endswith("\n") else text + "\n")
    if session is not None:
        audit.log(session, "user", "config_edit", {"file": name, "backup": str(backup)}, when)
        session.commit()
    return path


def add_holiday(day: date, session: Session | None = None, now: datetime | None = None) -> Path:
    """Insert one market-holiday date (kept sorted, comments preserved) via the validated save."""
    text = read("holidays.yaml")
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip() == "holidays:"), None)
    if start is None:
        raise ConfigEditError("holidays.yaml has no 'holidays:' list")
    end = start + 1
    while end < len(lines) and lines[end].startswith("  - "):
        end += 1
    existing = [ln[4:].strip() for ln in lines[start + 1 : end]]
    if day.isoformat() in existing:
        return _resolve("holidays.yaml")
    pos = start + 1 + sum(1 for d in existing if d < day.isoformat())
    lines.insert(pos, f"  - {day.isoformat()}")
    return save("holidays.yaml", "\n".join(lines) + "\n", session, now)


def exclude_symbol(
    symbol: str, reason: str, session: Session | None = None, now: datetime | None = None
) -> Path:
    """Leave a symbol out of backtests/training for a data reason (never because of returns)."""
    symbol, reason = symbol.strip().upper(), reason.strip().replace(",", ";").replace("\n", " ")
    if not symbol or not reason:
        raise ConfigEditError("symbol and reason are both required")
    text = read("universe/excluded.csv")
    if any(ln.split(",")[0] == symbol for ln in text.splitlines()):
        return _resolve("universe/excluded.csv")
    return save("universe/excluded.csv", text.rstrip("\n") + f"\n{symbol},{reason}\n", session, now)
