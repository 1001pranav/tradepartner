"""Background jobs started from the UI. A fixed whitelist of research commands only: training,
replay, paper simulation, backtest and data tasks. Nothing here can place an order or touch the
broker: those commands are simply not in the list, and arguments are validated, never free text.
"""

import json
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from tradepartner.paths import home_dir

ML_STRATEGIES = ("swing_v1", "positional_v1")
BASELINES = ("baseline_momentum20", "baseline_momentum12_1")
ACTIONS = {
    "train": "Train a model (candidate only, nothing approved)",
    "replay": "Replay: hide the future, predict, reveal",
    "paper-sim": "Historical paper trading (simulated)",
    "backtest": "Backtest a baseline rule",
    "news-fetch": "News: fetch latest articles (RSS, NSE, BSE)",
    "news-score": "News: score articles with the local LLM (heavy; do not run with training)",
    "explain-jumps": "Explain price jumps from web news (local LLM; heavy)",
    "data-quality": "Data quality check",
    "data-backfill": "Backfill / resume candle download (read-only market data)",
}
_PY = "from tradepartner.cli import app; app()"


class JobError(ValueError):
    pass


def jobs_dir() -> Path:
    return home_dir() / "logs" / "ui-jobs"


def _iso(value: object, field: str) -> str:
    try:
        return date.fromisoformat(str(value)).isoformat()
    except ValueError as exc:
        raise JobError(f"{field} must be YYYY-MM-DD") from exc


def _int(value: object, field: str, lo: int, hi: int) -> str:
    try:
        n = int(str(value))
    except ValueError as exc:
        raise JobError(f"{field} must be a number") from exc
    if not lo <= n <= hi:
        raise JobError(f"{field} must be between {lo} and {hi}")
    return str(n)


def _choice(value: object, field: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise JobError(f"{field} must be one of {', '.join(allowed)}")
    return str(value)


def build_args(action: str, p: dict[str, Any]) -> list[str]:
    """Validated CLI arguments for a whitelisted action."""
    if action == "train":
        args = [
            "train",
            "--strategy",
            _choice(p.get("strategy"), "strategy", ML_STRATEGIES),
            "--news" if p.get("news") else "--no-news",
        ]
        if p.get("trials") not in (None, ""):
            args += ["--trials", _int(p["trials"], "trials", 0, 50)]
        return args
    if action == "replay":
        return [
            "replay",
            "--as-of",
            _iso(p.get("as_of"), "as_of"),
            "--days",
            _int(p.get("days", 5), "days", 1, 60),
            "--top",
            _int(p.get("top", 5), "top", 1, 20),
        ]
    if action == "paper-sim":
        args = [
            "paper-sim",
            "--strategy",
            _choice(p.get("strategy"), "strategy", ML_STRATEGIES),
            "--start",
            _iso(p.get("start"), "start"),
            "--capital-inr",
            _int(p.get("capital_inr", 1_000_000), "capital", 10_000, 100_000_000),
        ]
        if p.get("end"):
            args += ["--end", _iso(p["end"], "end")]
        return args
    if action == "backtest":
        return [
            "backtest",
            "--strategy",
            _choice(p.get("strategy"), "strategy", BASELINES),
            "--capital-inr",
            _int(p.get("capital_inr", 1_000_000), "capital", 10_000, 100_000_000),
        ]
    if action == "news-fetch":
        return ["news", "fetch", "--hours", _int(p.get("hours", 48), "hours", 1, 720)]
    if action == "news-score":
        return [
            "news",
            "score",
            "--limit",
            _int(p.get("limit", 20), "limit", 1, 500),
            "--max-minutes",
            _int(p.get("max_minutes", 10), "max_minutes", 1, 120),
        ]
    if action == "explain-jumps":
        args = ["data", "explain-jumps", "--limit", _int(p.get("limit", 20), "limit", 1, 500)]
        if p.get("symbol"):
            if not re.fullmatch(r"[A-Z0-9&-]{1,20}", str(p["symbol"])):
                raise JobError("symbol must be an NSE symbol")
            args += ["--symbol", str(p["symbol"])]
        return args
    if action == "data-quality":
        return ["data", "quality"]
    if action == "data-backfill":
        return [
            "data",
            "backfill",
            "--universe",
            "nifty100",
            "--years",
            _int(p.get("years", 10), "years", 1, 20),
        ]
    raise JobError(f"unknown action {action!r}")


@dataclass(frozen=True)
class UiJob:
    id: str
    action: str
    args: list[str]
    pid: int
    started: float
    log: str

    @property
    def log_path(self) -> Path:
        return Path(self.log)

    def exit_code(self) -> int | None:
        """None while running; the process exit code once the wrapper wrote it."""
        if not self.log_path.exists():
            return None
        m = re.findall(r"^\[exit (\d+)\]$", self.log_path.read_text(errors="replace"), re.M)
        return int(m[-1]) if m else None

    @property
    def status(self) -> str:
        code = self.exit_code()
        return "running" if code is None else ("done" if code == 0 else f"failed ({code})")


def launch(
    action: str, params: dict[str, Any], popen: Callable[..., Any] = subprocess.Popen
) -> UiJob:
    args = build_args(action, params)
    jobs_dir().mkdir(parents=True, exist_ok=True)
    job_id = f"{int(time.time())}-{action}"
    log = jobs_dir() / f"{job_id}.log"
    cmd = " ".join(shlex.quote(a) for a in [sys.executable, "-c", _PY, *args])
    wrapper = f'{cmd} > {shlex.quote(str(log))} 2>&1; echo "[exit $?]" >> {shlex.quote(str(log))}'
    proc = popen(["/bin/sh", "-c", wrapper], start_new_session=True)
    job = UiJob(job_id, action, args, int(getattr(proc, "pid", 0)), time.time(), str(log))
    (jobs_dir() / f"{job_id}.json").write_text(json.dumps(job.__dict__))
    return job


def list_jobs(limit: int = 15) -> list[UiJob]:
    if not jobs_dir().exists():
        return []
    files = sorted(jobs_dir().glob("*.json"), reverse=True)[:limit]
    return [UiJob(**json.loads(f.read_text())) for f in files]


def tail(job: UiJob, lines: int = 40) -> str:
    if not job.log_path.exists():
        return ""
    text = [
        ln
        for ln in job.log_path.read_text(errors="replace").splitlines()
        if "Context impl" not in ln and "non-transactional" not in ln
    ]
    return "\n".join(text[-lines:])
