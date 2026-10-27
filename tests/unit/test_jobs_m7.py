"""Scheduled jobs (spec 12): every job runs offline with injected fakes."""

import threading
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from tests.fixtures.pipeline import NOW0, TrainedArtifact, add_model, make_env, train_once
from tradepartner import jobs, scheduler
from tradepartner.data import kite_auth
from tradepartner.db.models import (
    AuditLog,
    JobRun,
    Model,
    Order,
    OrderProposal,
    ResultDate,
    RiskEvent,
    Signal,
)
from tradepartner.modes.state import GLOBAL, strategy_scope
from tradepartner.news.sources.base import RawArticle
from tradepartner.risk import kill
from tradepartner.signals import proposals

BY = jobs.BY_NAME


@pytest.fixture(scope="module")
def art(tmp_path_factory: pytest.TempPathFactory) -> TrainedArtifact:
    return train_once(tmp_path_factory.mktemp("trained"))


def run(env, name: str) -> str:  # type: ignore[no-untyped-def]
    return jobs.run_job(env.ctx, BY[name])


def runs(db: Session, status: str | None = None) -> list[JobRun]:
    q = select(JobRun)
    if status:
        q = q.where(JobRun.status == status)
    return list(db.scalars(q))


def go_manual(env) -> None:  # type: ignore[no-untyped-def]
    with env.ctx.session_factory() as s:
        rt = env.ctx.runtime(s)
        rt.controller.set_mode(GLOBAL, "MANUAL", "user", "t", env.ctx.clock())
        rt.controller.set_mode(strategy_scope("swing_v1"), "MANUAL", "user", "t", env.ctx.clock())


# ---------------- runner ----------------


def test_run_job_records_ok_skip_and_error(db: Session, tmp_path: Path) -> None:
    env = make_env(db, tmp_path)
    boom = jobs.JobSpec("boom", lambda c: 1 / 0, "never")
    ok = jobs.JobSpec("fine", lambda c: "done", "never")
    assert jobs.run_job(env.ctx, ok) == "ok" and jobs.run_job(env.ctx, boom) == "error"
    assert env.notes[-1][0] == "TradePartner job failed" and "boom" in env.notes[-1][1]
    saturday = datetime.combine(
        NOW0.date() + timedelta(days=(5 - NOW0.weekday()) % 7 or 7),
        datetime.min.time(),
        tzinfo=NOW0.tzinfo,
    ).replace(hour=9)
    env.clock_now = saturday
    assert jobs.run_job(env.ctx, ok) == "skipped"  # not a trading day
    assert jobs.run_job(env.ctx, jobs.JobSpec("weekly_retrain", lambda c: "x", "Sat")) == "ok"
    statuses = {(r.job, r.status) for r in runs(db)}
    assert {
        ("fine", "ok"),
        ("boom", "error"),
        ("fine", "skipped"),
        ("weekly_retrain", "ok"),
    } <= statuses
    assert any(r.error and "ZeroDivision" in r.error for r in runs(db, "error"))


def test_market_hours_jobs_do_not_run_outside_the_session_and_leave_no_rows(
    db: Session, tmp_path: Path
) -> None:
    env = make_env(db, tmp_path)
    env.set_clock(8, 0)
    assert jobs.run_job(env.ctx, BY["track_limits"]) == "skipped" and runs(db) == []
    env.set_clock(9, 14)
    assert jobs.run_job(env.ctx, BY["poll_quotes"]) == "skipped"
    env.set_clock(15, 30)
    assert jobs.run_job(env.ctx, BY["poll_quotes"]) == "skipped"
    assert jobs.in_market_hours(NOW0.replace(hour=9, minute=15))


def test_heavy_jobs_never_run_concurrently(db: Session, tmp_path: Path) -> None:
    env = make_env(db, tmp_path)
    assert env.ctx.heavy.acquire()
    try:
        assert (
            jobs.run_job(env.ctx, BY["news_score"]) == "skipped"
        )  # the LLM/training mutex is held
    finally:
        env.ctx.heavy.release()
    skipped = runs(db, "skipped")
    assert skipped and "heavy job" in (skipped[0].error or "")
    assert not env.ctx.heavy.locked()  # an ok/failed job always releases the lock
    jobs.run_job(env.ctx, BY["news_score"])
    assert not env.ctx.heavy.locked() and isinstance(env.ctx.heavy, type(threading.Lock()))


def test_catch_up_runs_missed_idempotent_jobs_but_never_submits_entries(
    db: Session, tmp_path: Path
) -> None:
    env = make_env(db, tmp_path)
    env.set_clock(10, 0)  # the laptop woke up at 10:00
    ran = jobs.catch_up(env.ctx, jobs.JOBS)
    assert "health_check" in ran and "generate_signals" in ran and "submit_entries" not in ran
    assert (
        "sync_ledger" not in ran and "daily_report" not in ran
    )  # their time (15:45, 17:00) has not come
    assert jobs.catch_up(env.ctx, jobs.JOBS) == [
        j for j in ran if j not in jobs.jobs_done_today(env.ctx)
    ]
    env.set_clock(18, 0)
    later = jobs.catch_up(env.ctx, jobs.JOBS)
    assert {"sync_ledger", "eod_candles", "daily_report"} <= set(later)
    assert (
        not BY["submit_entries"].catch_up and BY["news_score"].heavy and BY["weekly_retrain"].heavy
    )


# ---------------- individual jobs ----------------


def test_health_check_reports_problems(db: Session, tmp_path: Path) -> None:
    env = make_env(db, tmp_path)
    assert run(env, "health_check") == "ok"
    assert "Kite login needed" in env.notes[-1][1]  # no token in the (memory) Keychain
    kite_auth.save_access_token("t", NOW0.date())
    env.ctx.internet_check = lambda: False
    env.ollama.up = False
    run(env, "health_check")
    assert "no internet" in env.notes[-1][1] and "Ollama is not running" in env.notes[-1][1]
    assert "Kite login" not in env.notes[-1][1]


def test_refresh_reference_and_results_calendar(
    db: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = make_env(db, tmp_path)
    assert run(env, "refresh_reference") == "ok"
    from tradepartner.db.models import Instrument

    assert db.scalar(select(func.count()).select_from(Instrument)) == 8
    env.ctx.kite = lambda: None
    run(env, "refresh_reference")
    import httpx

    real = httpx.Client
    cal = [
        {
            "symbol": "SYM01",
            "purpose": "Financial Results",
            "date": (NOW0.date() + timedelta(days=3)).strftime("%d-%b-%Y"),
        }
    ]
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda *a, **k: real(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json=cal if "event" in r.url.path else {})
            )
        ),
    )
    run(env, "results_calendar")
    assert db.scalars(select(ResultDate.symbol)).all() == ["SYM01"]


def test_news_fetch_and_score(db: Session, tmp_path: Path) -> None:
    env = make_env(db, tmp_path)

    class Src:
        name = "t"

        def fetch(self, since: datetime) -> list[RawArticle]:
            return [
                RawArticle(
                    "t",
                    "https://x/1",
                    "SYM01 wins a big order",
                    "body",
                    env.clock_now.astimezone(UTC) - timedelta(hours=1),
                    symbol_hint="SYM01",
                )
            ]

    env.ctx.news_sources = lambda: [Src()]
    assert run(env, "news_fetch") == "ok" and run(env, "news_fetch") == "ok"
    from tests.fixtures.llm import GOOD
    from tradepartner.db.models import NewsRaw, NewsScore

    assert db.scalar(select(func.count()).select_from(NewsRaw)) == 1
    env.ollama.default = {**GOOD, "symbol": "SYM01"}
    env.set_clock(8, 10)
    assert run(env, "news_score") == "ok"
    assert db.scalar(select(func.count()).select_from(NewsScore)) == 1
    env.ollama.up = False
    assert "Ollama down" in (jobs.job_news_score(env.ctx) or "")


def test_generate_signals_job_needs_fresh_data_then_creates_signals(
    db: Session, tmp_path: Path, art: TrainedArtifact
) -> None:
    env = make_env(db, tmp_path)
    add_model(db, art, "approved")
    add_model(db, art, "shadow")
    env.set_clock(8, 50, NOW0.date() + timedelta(days=7))  # data is a week stale
    assert "not fresh" in jobs.job_generate_signals(env.ctx) and env.notes
    env.set_clock(8, 50)
    detail = jobs.job_generate_signals(env.ctx)
    assert "swing_v1/approved" in detail and "swing_v1/shadow" in detail
    sigs = db.scalars(select(Signal)).all()
    assert sigs and {s.model_id for s in sigs} == {m.id for m in db.scalars(select(Model))}
    assert any(s.explanation for s in sigs) and any(
        s.explanation is None for s in sigs
    )  # shadow: none
    assert (
        jobs.job_notify_proposals(env.ctx).endswith("signals")
        and "signals ready" in env.notes[-1][1]
    )
    again = jobs.job_generate_signals(env.ctx)
    assert "0 signals" in again  # idempotent


def test_generate_signals_without_models_or_data(db: Session, tmp_path: Path) -> None:
    env = make_env(db, tmp_path)
    assert jobs.job_generate_signals(env.ctx) == "no approved or shadow models"


def test_full_manual_morning_flow(db: Session, tmp_path: Path, art: TrainedArtifact) -> None:
    env = make_env(db, tmp_path)
    add_model(db, art, "approved")
    go_manual(env)
    env.set_clock(8, 50)
    assert run(env, "generate_signals") == "ok"
    env.set_clock(9, 20)
    assert run(env, "submit_entries") == "ok"
    assert env.kite.quote_calls >= 1  # fresh quotes before routing
    shown = db.scalars(select(OrderProposal).where(OrderProposal.risk_decision == "approved")).all()
    assert (
        shown and db.scalar(select(func.count()).select_from(Order)) == 0
    )  # MANUAL places nothing
    assert "proposals actionable" in env.notes[-1][1]
    assert any(proposals.current_status(db, s.id) == "shown" for s in db.scalars(select(Signal)))


def test_submit_entries_needs_a_broker_connection(db: Session, tmp_path: Path) -> None:
    env = make_env(db, tmp_path)
    env.ctx.kite = lambda: None
    assert jobs.job_submit_entries(env.ctx) == "no broker connection: nothing submitted"
    assert jobs.job_poll_quotes(env.ctx) == "no Kite login"


def test_stale_entries_are_never_submitted_after_the_signal_age_limit(
    db: Session, tmp_path: Path, art: TrainedArtifact
) -> None:
    env = make_env(db, tmp_path)
    add_model(db, art, "approved")
    go_manual(env)
    env.set_clock(8, 50)
    run(env, "generate_signals")
    env.set_clock(15, 0)  # the laptop was asleep until 15:00: > 3 h later
    run(env, "submit_entries")
    assert db.scalar(select(func.count()).select_from(OrderProposal)) == 0


def test_track_limits_halts_on_a_loss_breach_and_honours_the_kill_file(
    db: Session, tmp_path: Path, art: TrainedArtifact
) -> None:
    from tradepartner.execution.paper import PaperBroker

    env = make_env(
        db, tmp_path, broker_factory=lambda s, q: PaperBroker(s, q.get, 10.0, env.ctx.clock)
    )
    add_model(db, art, "approved")
    with env.ctx.session_factory() as s:
        rt = env.ctx.runtime(s)
        rt.controller.set_mode(GLOBAL, "MANUAL", "user", "t", env.ctx.clock())
    env.set_clock(9, 30)
    assert run(env, "track_limits") == "ok"
    assert (
        db.scalar(select(func.count()).select_from(RiskEvent).where(RiskEvent.severity == "halt"))
        == 0
    )
    kill.arm()
    run(env, "track_limits")
    db.expire_all()
    from tradepartner.modes.state import ModeReader

    assert ModeReader(db).current(GLOBAL).mode == "OFF"
    assert (
        db.scalar(
            select(func.count()).select_from(RiskEvent).where(RiskEvent.rule == "kill_switch")
        )
        == 1
    )
    kill.clear()


def test_expire_entries_and_square_off(db: Session, tmp_path: Path, art: TrainedArtifact) -> None:
    from tradepartner.execution.paper import PaperBroker

    env = make_env(
        db, tmp_path, broker_factory=lambda s, q: PaperBroker(s, q.get, 10.0, env.ctx.clock)
    )
    add_model(db, art, "approved")
    go_manual(env)
    env.set_clock(8, 50)
    run(env, "generate_signals")
    env.set_clock(9, 20)
    run(env, "submit_entries")
    env.set_clock(15, 0)
    assert "expired" in (jobs.job_expire_entries(env.ctx) or "")
    env.set_clock(15, 10)
    assert jobs.job_square_off_mis(env.ctx).startswith("squared off 0")  # no MIS positions


def test_sync_ledger_eod_and_reports(db: Session, tmp_path: Path, art: TrainedArtifact) -> None:
    env = make_env(db, tmp_path)
    add_model(db, art, "approved")
    env.set_clock(15, 45)
    assert run(env, "sync_ledger") == "ok"
    env.ctx.kite = lambda: None
    assert "broker sync skipped" in jobs.job_sync_ledger(env.ctx)
    env.ctx.kite = lambda: env.kite
    assert run(env, "eod_candles") == "ok"
    assert run(env, "daily_report") == "ok"
    from tradepartner.paths import reports_dir

    assert list(reports_dir().glob("daily-*.html"))
    assert "TradePartner daily report" in env.notes[-1][0]
    env.ctx.kite = lambda: None
    assert jobs.job_eod_candles(env.ctx) == "no Kite login"


def test_token_check_lowers_auto_when_there_is_no_token(
    db: Session, tmp_path: Path, art: TrainedArtifact
) -> None:
    env = make_env(db, tmp_path)
    with env.ctx.session_factory() as s:
        rt = env.ctx.runtime(s)
        rt.controller.set_mode(GLOBAL, "AUTO", "user", "t", env.ctx.clock())
    assert jobs.job_token_check(env.ctx).startswith("no token")
    with env.ctx.session_factory() as s:
        assert (
            env.ctx.runtime(s).controller.current(GLOBAL).mode == "MANUAL"
        )  # AUTO never survives without a token
    kite_auth.save_access_token("t", NOW0.date())
    assert jobs.job_token_check(env.ctx) == "token ok"


def test_backup_and_drift_jobs(db: Session, tmp_path: Path, art: TrainedArtifact) -> None:
    env = make_env(db, tmp_path)
    out = jobs.job_backup(env.ctx)
    assert (
        "backup written" in out or out == ""
    )  # sqlite3 may be missing; the script itself still ran
    assert jobs.job_drift_check(env.ctx) == "no approved models"
    m = add_model(db, art, "approved")
    m.metrics_json = {**m.metrics_json, "feature_bins": {}, "train_brier": None}
    db.commit()
    assert "swing_v1: ok" in jobs.job_drift_check(env.ctx)
    m.metrics_json = {
        **m.metrics_json,
        "feature_bins": {
            "ret_20": {"edges": [float("-inf"), 0.0, float("inf")], "expected": [0.999, 0.001]}
        },
    }
    db.commit()
    with env.ctx.session_factory() as s:
        rt = env.ctx.runtime(s)
        rt.controller.set_mode(GLOBAL, "AUTO", "user", "t", env.ctx.clock())
        rt.controller.unlock_auto(
            "swing_v1",
            __import__("tradepartner.modes.controller", fromlist=["x"]).UnlockEvidence(
                True, 40, date(2025, 1, 1), True
            ),
            "swing_v1",
            env.ctx.clock(),
        )
        rt.controller.set_mode(strategy_scope("swing_v1"), "AUTO", "user", "t", env.ctx.clock())
    assert "DRIFT" in jobs.job_drift_check(env.ctx) and "drift" in env.notes[-1][0].lower()
    with env.ctx.session_factory() as s:
        assert env.ctx.runtime(s).controller.current(strategy_scope("swing_v1")).mode == "MANUAL"


def test_weekly_retrain_registers_shadow_candidates_and_never_approves(
    db: Session, tmp_path: Path
) -> None:
    env = make_env(db, tmp_path)
    env.set_clock(10, 0, NOW0.date() + timedelta(days=(5 - NOW0.weekday()) % 7 or 7))  # Saturday
    assert jobs.run_job(env.ctx, BY["weekly_retrain"]) == "ok"
    models = db.scalars(select(Model)).all()
    assert any(m.strategy == "swing_v1" and m.status == "shadow" for m in models)
    assert all(m.status != "approved" and m.approved_by is None for m in models)
    assert (
        db.scalar(
            select(func.count()).select_from(AuditLog).where(AuditLog.action == "model_shadow")
        )
        >= 1
    )
    assert "weekly retrain" in env.notes[-1][0]
    empty = make_env(db, tmp_path / "x").ctx
    empty.store = type(empty.store)(tmp_path / "empty")
    assert jobs.job_weekly_retrain(empty) == "no market data"


# ---------------- scheduler wiring ----------------


def test_every_job_has_a_trigger_and_submit_entries_cannot_be_caught_up() -> None:
    scheduler.validate()
    assert set(scheduler.TRIGGERS) == set(BY)
    original = jobs.JOBS[:]
    try:
        jobs.JOBS.append(jobs.JobSpec("extra", lambda c: None, "x"))
        with pytest.raises(RuntimeError, match="differ"):
            scheduler.validate()
        jobs.JOBS.pop()
        jobs.JOBS[jobs.JOBS.index(BY["submit_entries"])] = jobs.JobSpec(
            "submit_entries", lambda c: None, "09:20", catch_up=True
        )
        with pytest.raises(RuntimeError, match="never be caught up"):
            scheduler.validate()
    finally:
        jobs.JOBS[:] = original


def test_scheduler_registers_jobs_at_the_spec_times(db: Session, tmp_path: Path) -> None:
    env = make_env(db, tmp_path)
    sched = scheduler.build_scheduler(env.ctx, BackgroundScheduler(timezone=scheduler.IST))
    ids = {j.id for j in sched.get_jobs()}
    assert ids == set(BY)
    sched.start(paused=True)
    try:
        nxt = {
            j.id: j.trigger.get_next_fire_time(
                None, datetime.combine(NOW0.date(), datetime.min.time(), tzinfo=scheduler.IST)
            )
            for j in sched.get_jobs()
        }
        job = next(j for j in sched.get_jobs() if j.id == "health_check")
    finally:
        sched.shutdown(wait=False)
    assert (nxt["health_check"].hour, nxt["health_check"].minute) == (7, 45)
    assert (nxt["generate_signals"].hour, nxt["generate_signals"].minute) == (8, 50)
    assert (nxt["submit_entries"].hour, nxt["submit_entries"].minute) == (9, 20)
    assert (nxt["sync_ledger"].hour, nxt["sync_ledger"].minute) == (15, 45)
    assert nxt["weekly_retrain"].weekday() == 5 and nxt["weekly_retrain"].hour == 10
    assert nxt["poll_quotes"].second % 10 == 0 and nxt["news_fetch"].minute % 15 == 0
    # the scheduled callable runs the job through the guarded runner
    job.func()
    assert any(r.job == "health_check" for r in runs(db))
