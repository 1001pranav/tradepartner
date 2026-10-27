"""APScheduler wiring for the jobs in ``jobs.py`` (spec 12): IST, trading days only."""

from typing import Any

import structlog
from apscheduler.schedulers.base import BaseScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from tradepartner import jobs
from tradepartner.calendar import IST
from tradepartner.jobs import JobContext, JobSpec

log = structlog.get_logger()
WEEKDAYS = "mon-fri"

# name -> CronTrigger kwargs. The trading-day guard lives in ``jobs.run_job`` (holidays).
TRIGGERS: dict[str, dict[str, Any]] = {
    "health_check": {"hour": 7, "minute": 45, "day_of_week": WEEKDAYS},
    "refresh_reference": {"hour": 8, "minute": 0, "day_of_week": WEEKDAYS},
    "results_calendar": {"hour": 8, "minute": 5, "day_of_week": WEEKDAYS},
    "news_fetch": {"hour": "6-22", "minute": "*/15", "day_of_week": WEEKDAYS},
    "news_score": {"hour": 8, "minute": 10, "day_of_week": WEEKDAYS},
    "generate_signals": {"hour": 8, "minute": 50, "day_of_week": WEEKDAYS},
    "notify_proposals": {"hour": 9, "minute": 0, "day_of_week": WEEKDAYS},
    "token_check": {"hour": 9, "minute": 10, "day_of_week": WEEKDAYS},
    "poll_quotes": {"hour": "9-15", "minute": "*", "second": "*/10", "day_of_week": WEEKDAYS},
    "submit_entries": {"hour": 9, "minute": 20, "day_of_week": WEEKDAYS},
    "track_limits": {"hour": "9-15", "minute": "*", "second": "*/30", "day_of_week": WEEKDAYS},
    "poll_fills": {"hour": "9-15", "minute": "*", "second": "15,45", "day_of_week": WEEKDAYS},
    "protective_check": {"hour": "9-15", "minute": "*/5", "second": 5, "day_of_week": WEEKDAYS},
    "intraday_scan": {"hour": "9-14", "minute": "*/5", "second": 20, "day_of_week": WEEKDAYS},
    "mis_protect": {"hour": "9-15", "minute": "*", "second": "0,30", "day_of_week": WEEKDAYS},
    "expire_entries": {"hour": 15, "minute": 0, "day_of_week": WEEKDAYS},
    "square_off_mis": {"hour": 15, "minute": 10, "day_of_week": WEEKDAYS},
    "sync_ledger": {"hour": 15, "minute": 45, "day_of_week": WEEKDAYS},
    "eod_candles": {"hour": 16, "minute": 30, "day_of_week": WEEKDAYS},
    "drift_check": {"hour": 16, "minute": 45, "day_of_week": WEEKDAYS},
    "daily_report": {"hour": 17, "minute": 0, "day_of_week": WEEKDAYS},
    "backup": {"hour": 17, "minute": 5, "day_of_week": WEEKDAYS},
    "weekly_retrain": {"hour": 10, "minute": 0, "day_of_week": "sat"},
}


def validate() -> None:
    """Every job has a trigger and vice versa; no catch-up job may submit orders."""
    names = {j.name for j in jobs.JOBS}
    if names != set(TRIGGERS):
        raise RuntimeError(f"jobs and triggers differ: {names ^ set(TRIGGERS)}")
    if any(j.name == "submit_entries" and j.catch_up for j in jobs.JOBS):
        raise RuntimeError("submit_entries must never be caught up (no stale entry orders)")


def build_scheduler(ctx: JobContext, scheduler: BaseScheduler | None = None) -> BaseScheduler:
    validate()
    sched = scheduler or BlockingScheduler(timezone=IST)
    for spec in jobs.JOBS:
        sched.add_job(
            _runner(ctx, spec), CronTrigger(timezone=IST, **TRIGGERS[spec.name]), id=spec.name,
            name=spec.name, max_instances=1, coalesce=True, misfire_grace_time=300,
        )  # fmt: skip
    return sched


def _runner(ctx: JobContext, spec: JobSpec) -> Any:
    def go() -> None:
        jobs.run_job(ctx, spec)

    go.__name__ = spec.name
    return go


def start(ctx: JobContext) -> None:  # pragma: no cover - blocks forever
    """Run catch-up for jobs missed while asleep, then block on the scheduler."""
    ran = jobs.catch_up(ctx, jobs.JOBS)
    if ran:
        log.info("catch_up_ran", jobs=ran)
    build_scheduler(ctx).start()
