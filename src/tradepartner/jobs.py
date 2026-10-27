"""Scheduled jobs (spec 12). Every external dependency (Kite, Ollama, clock, notifier) is
injected through ``JobContext`` so each job is testable offline. Each job opens its own DB
session because APScheduler runs jobs on worker threads.
"""

import shutil
import subprocess
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.backtest import loader
from tradepartner.calendar import IST, TradingCalendar
from tradepartner.data import candles, instruments, kite_auth, quality
from tradepartner.data import meta as meta_mod
from tradepartner.data import universe as universe_mod
from tradepartner.data.corporate_actions import Action, load_actions
from tradepartner.data.live import Bar, BarBuilder, bars_to_frame
from tradepartner.data.quotes import QuoteBook, poll
from tradepartner.db.models import JobRun, Model, OrderProposal, ResultDate
from tradepartner.execution import protective
from tradepartner.execution.broker import Broker
from tradepartner.execution.killswitch import Pending
from tradepartner.execution.mis_protect import place_mis_stops
from tradepartner.execution.protective import ProtectiveManager
from tradepartner.features import news as news_features
from tradepartner.ledger import rebuild, reconcile
from tradepartner.ledger import sync as ledger_sync
from tradepartner.llm.client import OllamaClient
from tradepartner.models import drift, pipeline, registry
from tradepartner.news import ingest as ingest_mod
from tradepartner.news.analyst import NewsAnalyst
from tradepartner.news.entity_link import AliasTable
from tradepartner.news.sources.base import NewsSource
from tradepartner.paths import backups_dir, home_dir, reports_dir
from tradepartner.reports import daily as daily_report
from tradepartner.risk import kill
from tradepartner.runtime import Runtime, build_runtime
from tradepartner.settings import Settings
from tradepartner.signals import explain, generate, outcomes, proposals, shadow
from tradepartner.signals.intraday import generate_intraday_signals
from tradepartner.strategies import intraday as intraday_mod
from tradepartner.strategies.factory import CLASSES, build_model_strategy

log = structlog.get_logger()
ML_STRATEGIES = tuple(CLASSES)
MARKET_OPEN, MARKET_CLOSE = (9, 15), (15, 30)


@dataclass
class JobContext:
    settings: Settings
    session_factory: Callable[[], Session]
    store: candles.CandleStore
    quotes: QuoteBook
    kite: Callable[[], Any | None]  # a logged-in Kite client, or None
    llm: Callable[[], OllamaClient]
    clock: Callable[[], datetime]
    notifier: Callable[[str, str], None]
    calendar: TradingCalendar
    news_sources: Callable[[], list[NewsSource]] = lambda: []
    broker_factory: Callable[[Session, QuoteBook], Broker] | None = None
    internet_check: Callable[[], bool] = lambda: True
    heavy: threading.Lock = field(default_factory=threading.Lock)
    kill_pending: list[Pending] = field(default_factory=list)
    bars: BarBuilder = field(default_factory=BarBuilder)
    new_bars: list[Bar] = field(default_factory=list)
    cache: dict[str, Any] = field(default_factory=dict)

    def runtime(self, session: Session) -> Runtime:
        today = self.clock().astimezone(IST).date()
        syms = universe_mod.members_as_of(session, self.settings.app.universe, today)
        broker = self.broker_factory(session, self.quotes) if self.broker_factory else None
        return build_runtime(
            session,
            self.settings,
            broker=broker,
            quotes=self.quotes,
            meta_provider=lambda: meta_mod.build_meta(session, self.store, syms, today),
            kill_pending=self.kill_pending,
            clock=self.clock,
            notifier=self.notifier,
            calendar=self.calendar,
        )

    def today(self) -> date:
        return self.clock().astimezone(IST).date()


def in_market_hours(now: datetime) -> bool:
    t = now.astimezone(IST)
    return MARKET_OPEN <= (t.hour, t.minute) < MARKET_CLOSE


@dataclass(frozen=True)
class JobSpec:
    name: str
    fn: Callable[[JobContext], str | None]
    schedule: str  # human-readable cron description
    heavy: bool = False
    catch_up: bool = False  # safe to run late on wake
    market_hours: bool = False  # only while the session is open
    daily_at: tuple[int, int] | None = None  # for catch-up ordering


# ---------------------------------------------------------------- runner


def run_job(ctx: JobContext, spec: JobSpec) -> str:
    """Run one job with a ``job_runs`` row, trading-day guard, heavy-job lock and error capture."""
    now = ctx.clock()
    if not ctx.calendar.is_trading_day(ctx.today()) and spec.name != "weekly_retrain":
        return _record(ctx, spec, now, "skipped", "not a trading day")
    if spec.market_hours and not in_market_hours(now):
        return "skipped"  # high-frequency jobs do not log a row per tick
    lock = ctx.heavy if spec.heavy else None
    if lock is not None and not lock.acquire(blocking=False):
        return _record(ctx, spec, now, "skipped", "another heavy job is running")
    try:
        detail = spec.fn(ctx)
        return _record(ctx, spec, now, "ok", detail)
    except Exception as exc:  # a failed job must never take the scheduler down
        log.error("job_failed", job=spec.name, error=str(exc), trace=traceback.format_exc(limit=3))
        ctx.notifier("TradePartner job failed", f"{spec.name}: {exc}")
        return _record(ctx, spec, now, "error", f"{type(exc).__name__}: {exc}")
    finally:
        if lock is not None:
            lock.release()


def _record(
    ctx: JobContext, spec: JobSpec, started: datetime, status: str, detail: str | None
) -> str:
    with ctx.session_factory() as s:
        s.add(
            JobRun(
                job=spec.name,
                started_at=started,
                ended_at=ctx.clock(),
                status=status,
                error=detail if status in ("error", "skipped") else None,
            )
        )
        s.commit()
    return status


def jobs_done_today(ctx: JobContext) -> set[str]:
    with ctx.session_factory() as s:
        return {
            r.job
            for r in s.scalars(select(JobRun).where(JobRun.status == "ok"))
            if r.started_at.astimezone(IST).date() == ctx.today()
        }


def catch_up(ctx: JobContext, specs: list[JobSpec]) -> list[str]:
    """After sleep or a crash: run idempotent daily jobs whose time has passed and that did not
    run today. Order-submitting jobs are never in this list (stale entries are never sent)."""
    now = ctx.clock().astimezone(IST)
    done = jobs_done_today(ctx)
    ran = []
    for spec in sorted(
        (s for s in specs if s.catch_up and s.daily_at), key=lambda s: s.daily_at or (0, 0)
    ):
        hh, mm = spec.daily_at or (0, 0)
        if (now.hour, now.minute) >= (hh, mm) and spec.name not in done:
            run_job(ctx, spec)
            ran.append(spec.name)
    return ran


# ---------------------------------------------------------------- jobs


def job_health_check(ctx: JobContext) -> str:
    problems: list[str] = []
    free_gb = shutil.disk_usage(home_dir() if home_dir().exists() else Path.home()).free / 1e9
    if free_gb < 2:
        problems.append(f"low disk: {free_gb:.1f} GB free")
    with ctx.session_factory() as s:
        s.execute(select(1))
    if not ctx.llm().health():
        problems.append("Ollama is not running")
    if not ctx.internet_check():
        problems.append("no internet")
    if kite_auth.load_access_token(ctx.today()) is None:
        problems.append("Kite login needed: run `tradepartner login`")
    msg = "; ".join(problems) or "all systems OK"
    ctx.notifier("TradePartner health", msg)
    return msg


def job_refresh_reference(ctx: JobContext) -> str:
    kite = ctx.kite()
    notes = []
    with ctx.session_factory() as s:
        if kite is not None:
            notes.append(f"{instruments.refresh_instruments(s, kite, ctx.clock())} instruments")
        else:
            notes.append("no Kite login: instruments not refreshed")
        names = universe_mod.members_as_of(s, ctx.settings.app.universe, ctx.today())
        notes.append(f"{len(names)} universe members")
    return "; ".join(notes)


def job_results_calendar(ctx: JobContext) -> str:
    from tradepartner.news.sources.nse import NseAnnouncements

    src = NseAnnouncements(acquire=candles.RateLimiter(0.5).acquire)
    added = 0
    with ctx.session_factory() as s:
        for sym, when, purpose in src.fetch_result_calendar(
            ctx.today(), ctx.today() + timedelta(days=14)
        ):
            if (
                s.scalar(
                    select(ResultDate.id).where(
                        ResultDate.symbol == sym, ResultDate.result_date == when
                    )
                )
                is None
            ):
                s.add(ResultDate(symbol=sym, result_date=when, purpose=purpose, source="nse"))
                added += 1
        s.commit()
    return f"{added} result dates"


def job_news_fetch(ctx: JobContext) -> str:
    now = ctx.clock()
    stored = 0
    with ctx.session_factory() as s:
        aliases = AliasTable.from_session(s)
        for src in ctx.news_sources():
            stored += ingest_mod.ingest(
                s, src.fetch(now - timedelta(hours=26)), now, aliases
            ).stored
    return f"{stored} new articles"


def job_news_score(ctx: JobContext) -> str:
    """08:00-08:50 batch scoring window (heavy). Exchange announcements and universe first."""
    now = ctx.clock().astimezone(IST)
    deadline = now.replace(hour=8, minute=50, second=0, microsecond=0)
    budget = max(60.0, (deadline - now).total_seconds()) if now < deadline else 3000.0
    llm = ctx.llm()
    if not llm.health():
        return "Ollama down: scoring skipped"
    with ctx.session_factory() as s:
        names = frozenset(universe_mod.members_as_of(s, ctx.settings.app.universe, ctx.today()))
        run = NewsAnalyst(s, llm, clock=ctx.clock).run(
            limit=600, priority_symbols=names, max_seconds=budget
        )
    return f"scored {run.scored}, discarded {run.discarded}, {run.per_hour:.0f}/hour"


def _view(ctx: JobContext, s: Session) -> tuple[Any, date]:
    today = ctx.today()
    names = universe_mod.members_as_of(s, ctx.settings.app.universe, today)
    acts: dict[str, list[Action]] = {}
    for a in load_actions(s):
        acts.setdefault(a.symbol, []).append(a)
    view = loader.load_view(ctx.store, names, acts)
    as_of = candles.last_completed_session(ctx.clock(), ctx.calendar)
    return view, as_of


def job_generate_signals(ctx: JobContext) -> str:
    """08:50: features as of the previous close, strategies, signals and explanations."""
    notes: list[str] = []
    with ctx.session_factory() as s:
        view, as_of = _view(ctx, s)
        if not view.symbols or as_of not in view.trading_days():
            msg = f"market data is not fresh for {as_of}: signals skipped"
            ctx.notifier("TradePartner", msg)
            return msg
        llm = ctx.llm()
        explainer = explain.Explainer(llm if llm.health() else None)
        now = ctx.clock()

        def live_news(day: date, symbols: list[str]) -> Any:
            return news_features.news_features(s, symbols, now, ctx.settings.app.llm, mode="live")

        for strategy in ML_STRATEGIES:
            for status in ("approved", "shadow"):
                for model in registry.by_status(s, strategy, status):
                    strat = build_model_strategy(model, ctx.settings, live_news)
                    sigs = generate.generate_signals(
                        s,
                        model,
                        strat,
                        view,
                        as_of,
                        now,
                        explainer,
                        lambda sym, t: generate.recent_news_lines(s, sym, t),
                    )
                    notes.append(f"{strategy}/{status}: {len(sigs)} signals")
    return "; ".join(notes) or "no approved or shadow models"


def job_notify_proposals(ctx: JobContext) -> str:
    n = 0
    with ctx.session_factory() as s:
        for strategy in ML_STRATEGIES:
            n += len(proposals.pending_signals(s, strategy, ctx.clock(), timedelta(hours=18)))
    ctx.notifier("TradePartner", f"{n} signals ready; they become actionable at 09:20")
    return f"{n} signals"


def _symbols_of_interest(s: Session, rt: Runtime, ctx: JobContext) -> list[str]:
    syms = {p.symbol for p in rt.context(None).positions}
    for strategy in ML_STRATEGIES:
        for sig in proposals.pending_signals(s, strategy, ctx.clock(), timedelta(hours=18)):
            syms.add(sig.symbol)
    return sorted(syms)


def job_poll_quotes(ctx: JobContext) -> str:
    kite = ctx.kite()
    if kite is None:
        return "no Kite login"
    with ctx.session_factory() as s:
        rt = ctx.runtime(s)
        symbols = _symbols_of_interest(s, rt, ctx)
    n = (
        poll(kite, symbols, ctx.quotes, ctx.clock(), candles.RateLimiter(1.0).acquire)
        if symbols
        else 0
    )
    return f"{n} quotes"


def _intraday_enabled(ctx: JobContext) -> bool:
    cfg = ctx.settings.strategies.get(intraday_mod.NAME)
    return bool(cfg and cfg.enabled)


def job_intraday_scan(ctx: JobContext) -> str:
    """Every 5 minutes: persist finished bars, scan for ORB/VWAP setups, route fresh signals.
    Paper-only until the intraday gate passes (the risk engine enforces it for the live broker)."""
    if not _intraday_enabled(ctx):
        return "intraday_v1 is disabled in config/strategies/intraday_v1.yaml"
    now = ctx.clock()
    done = ctx.new_bars + ctx.bars.flush(now)
    ctx.new_bars.clear()
    for sym in sorted({b.symbol for b in done}):
        ctx.store.write("5minute", sym, bars_to_frame([b for b in done if b.symbol == sym]))
    with ctx.session_factory() as s:
        models = [
            m for st in ("approved", "shadow") for m in registry.by_status(s, intraday_mod.NAME, st)
        ]
        if not models:
            return "no approved or shadow intraday model"
        names = universe_mod.members_as_of(s, ctx.settings.app.universe, ctx.today())
        bars = {sym: ctx.store.read("5minute", sym).tail(400) for sym in names}
        bars = {k: v for k, v in bars.items() if len(v)}
        if "intraday_blocked" not in ctx.cache or ctx.cache.get("intraday_day") != ctx.today():
            feats = news_features.news_features(s, names, now, ctx.settings.app.llm, mode="live")
            bad = feats.index[(feats["sent_wmean_7d"] <= -0.3) & (feats["n_rel_1d"] >= 1)]
            ctx.cache.update(intraday_blocked=frozenset(bad), intraday_day=ctx.today())
        blocked: frozenset[str] = ctx.cache["intraday_blocked"]
        cfg = ctx.settings.strategies[intraday_mod.NAME].model_dump()
        made = 0
        for model in models:
            scanner = intraday_mod.IntradayScanner(
                registry.load_booster(model), list(model.metrics_json["feature_columns"]), cfg
            )
            made += len(generate_intraday_signals(s, model, scanner, bars, now, blocked))
        rt = ctx.runtime(s)
        r = proposals.submit_pending(s, rt.router, intraday_mod.NAME, now, timedelta(minutes=10))
    return f"{made} signals; shown {r.shown}, placed {r.placed}, rejected {r.rejected}"


def job_mis_protect(ctx: JobContext) -> str:
    """Every 30 s: every bot MIS position must have its exchange SL order."""
    with ctx.session_factory() as s:
        placed = place_mis_stops(ctx.runtime(s), ctx.clock())
    return f"stops placed for {placed}" if placed else "ok"


def job_submit_entries(ctx: JobContext) -> str:
    """09:20: AUTO places risk-approved entries; MANUAL makes them actionable. Never catch-up."""
    if ctx.kite() is None and ctx.broker_factory is None:
        return "no broker connection: nothing submitted"
    job_poll_quotes(ctx)
    total = proposals.SubmitSummary()
    with ctx.session_factory() as s:
        rt = ctx.runtime(s)
        for strategy in ML_STRATEGIES:
            r = proposals.submit_pending(s, rt.router, strategy, ctx.clock())
            total.shown += r.shown
            total.placed += r.placed
            total.rejected += r.rejected
    ctx.notifier(
        "TradePartner",
        f"{total.shown} proposals actionable, {total.placed} placed, {total.rejected} rejected",
    )
    return f"shown {total.shown}, placed {total.placed}, rejected {total.rejected}"


def job_track_limits(ctx: JobContext) -> str:
    """Every 30 s: loss limits -> halts, kill file, kill escalation."""
    with ctx.session_factory() as s:
        rt = ctx.runtime(s)
        notes = []
        if kill.kill_active() and rt.controller.current("global").mode != "OFF":
            rt.kill.trigger(ctx.clock(), "KILL file present")
            notes.append("kill switch triggered")
        escalated = rt.kill.tick(ctx.clock())
        if escalated:
            notes.append(f"escalated {escalated}")
        switched = rt.enforce_limits(list(ML_STRATEGIES))
        if switched:
            ctx.notifier("TradePartner HALT", f"Switched OFF: {', '.join(switched)}")
            notes.append(f"halted {switched}")
    return "; ".join(notes) or "ok"


def job_expire_entries(ctx: JobContext) -> str:
    """15:00: no new entries; expire unfilled entry orders and their signals."""
    cancelled = 0
    with ctx.session_factory() as s:
        rt = ctx.runtime(s)
        for o in rt.broker.open_orders():
            if o.tag and o.tag.startswith("tp-") and o.side == "BUY":
                rt.broker.cancel_order(o.broker_order_id)
                cancelled += 1
        expired = sum(proposals.expire_unfilled(s, st, ctx.clock()) for st in ML_STRATEGIES)
    return f"cancelled {cancelled} entry orders, expired {expired} signals"


EXIT_DISCOUNT_BPS = 50  # protective exits sell this far below LTP to cross the spread


def _exit_limit_paise(ltp_paise: int, discount_bps: int = EXIT_DISCOUNT_BPS) -> int:
    """Integer-paise limit price below LTP, on the 5-paise tick (rule 6: no float money)."""
    return ltp_paise * (10_000 - discount_bps) // 10_000 // 5 * 5


def _protective_exit(
    ctx: JobContext,
    rt: Runtime,
    symbol: str,
    product: str,
    qty: int,
    strategy: str,
    reason: str,
    ltp_paise: int,
    discount_bps: int = EXIT_DISCOUNT_BPS,
) -> bool:
    """Sell a bot position through the router. Any working SELL for the symbol is cancelled
    first, and the key carries a 5-minute bucket, so an unfilled exit is re-priced and retried
    instead of being rejected as a duplicate."""
    from tradepartner.risk.types import Proposal

    for o in rt.broker.open_orders():
        if o.symbol == symbol and o.side == "SELL" and o.product == product:
            rt.broker.cancel_order(o.broker_order_id)
    now = ctx.clock()
    p = Proposal(
        strategy, symbol, "SELL", product, "LIMIT", "EXIT", now,
        f"prot-exit-{symbol}-{product}-{now:%Y%m%d%H}{now.minute // 5:02d}-{qty}",
        limit_price_paise=_exit_limit_paise(ltp_paise, discount_bps),
        exit_qty=qty, exit_reason=reason, protective=True,
    )  # fmt: skip
    return rt.router.submit(p, now).status == "placed"


def job_square_off_mis(ctx: JobContext) -> str:
    """15:10: square off bot-created MIS positions before the broker's auto square-off."""
    done, alerts = 0, []
    with ctx.session_factory() as s:
        rt = ctx.runtime(s)
        c = rt.context(None)
        for pos in c.positions:
            if pos.product != "MIS" or pos.qty <= 0:
                continue
            if not pos.bot_created:
                alerts.append(pos.symbol)
                continue
            q = c.quotes.get(pos.symbol)
            if q is None:
                alerts.append(pos.symbol)
                continue
            if _protective_exit(
                ctx, rt, pos.symbol, "MIS", pos.qty, pos.strategy or "mis_squareoff",
                "time_stop", q.ltp_paise, 20,
            ):  # fmt: skip
                done += 1
    if alerts:
        ctx.notifier("TradePartner", f"MIS positions need manual square-off: {', '.join(alerts)}")
    return f"squared off {done}; manual attention: {alerts}"


def job_sync_ledger(ctx: JobContext) -> str:
    """15:45: orders/trades/positions/holdings, rebuild, reconcile, reveal signal outcomes."""
    kite = ctx.kite()
    notes = []
    with ctx.session_factory() as s:
        if kite is not None:
            res = ledger_sync.sync(
                s,
                kite,
                ctx.today(),
                str(kite.profile().get("user_id", "kite")),
                ctx.settings.risk.per_trade.limit_price_band_pct,
            )
            notes.append(f"orders+{res.orders} fills+{res.fills}")
        else:
            notes.append("no Kite login: broker sync skipped")
        rebuilt = rebuild.rebuild(s, ctx.settings.costs, ctx.settings.tax)
        notes.append(f"{rebuilt.realized} realized rows")
        for alert in rebuilt.alerts:
            ctx.notifier("TradePartner ledger alert", alert)
        if kite is not None:
            mism = reconcile.reconcile_positions(s, ctx.today())
            if mism:
                ctx.notifier(
                    "TradePartner reconciliation", f"{len(mism)} mismatches: AUTO entries blocked"
                )
            notes.append(f"{len(mism)} position mismatches")
        view, _ = _view(ctx, s)
        if view.symbols:
            notes.append(
                f"{outcomes.reveal_outcomes(s, view, ctx.settings, ctx.clock())} outcomes revealed"
            )
    return "; ".join(notes)


def job_eod_candles(ctx: JobContext) -> str:
    kite = ctx.kite()
    if kite is None:
        return "no Kite login"
    with ctx.session_factory() as s:
        targets = universe_mod.tradable_members(s, ctx.settings.app.universe, ctx.today())
        acts: dict[str, list[Action]] = {}
        for a in load_actions(s):
            acts.setdefault(a.symbol, []).append(a)
    from tradepartner.data.kite_client import INDEX_TOKENS

    targets.update(INDEX_TOKENS)
    res = candles.backfill(
        kite,
        ctx.store,
        targets,
        years=10,
        now=ctx.clock(),
        calendar=ctx.calendar,
        limiter=candles.RateLimiter(2.0),
    )
    issues = quality.report(ctx.store, sorted(targets), ctx.calendar, acts)
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        ctx.notifier(
            "TradePartner data quality",
            f"{len(errors)} data errors, e.g. {errors[0].symbol}: {errors[0].kind}",
        )
    return f"{res.rows} rows, {len(res.failed)} failed, {len(errors)} quality errors"


def job_daily_report(ctx: JobContext) -> str:
    with ctx.session_factory() as s:
        rep = daily_report.build(s, ctx.today())
    path = reports_dir()
    path.mkdir(parents=True, exist_ok=True)
    (path / f"daily-{ctx.today()}.html").write_text(daily_report.render_html(rep))
    text = daily_report.render_text(rep)
    ctx.notifier("TradePartner daily report", text)
    return text


def job_backup(ctx: JobContext) -> str:
    script = Path(__file__).resolve().parents[2] / "ops" / "backup.sh"
    if not script.exists():
        return "backup script not found"
    out = subprocess.run(
        [str(script), str(home_dir())], capture_output=True, text=True, check=False, timeout=300
    )
    backups_dir().mkdir(parents=True, exist_ok=True)
    return (out.stdout or out.stderr).strip()[:200]


def job_drift_check(ctx: JobContext) -> str:
    """Daily drift monitors on approved models; AUTO -> MANUAL on a breach (spec 10.6)."""
    import numpy as np

    notes = []
    with ctx.session_factory() as s:
        rt = ctx.runtime(s)
        view, _ = _view(ctx, s)
        for strategy in ML_STRATEGIES:
            model = registry.current_approved(s, strategy)
            if model is None or not view.symbols:
                continue
            strat = build_model_strategy(model, ctx.settings)
            days = view.trading_days()[-30:]
            recent = strat._matrix(view)
            recent = recent[recent.index.get_level_values("date").isin(days)]
            oc = outcomes.outcome_frame(s, strategy, model.id)
            probs = oc["prob"].fillna(0.5).to_numpy(dtype=float) if len(oc) else np.array([])
            labels_ = oc["label"].fillna(0).to_numpy(dtype=float) if len(oc) else np.array([])
            rep = drift.check_drift(
                model.metrics_json.get("feature_bins", {}),
                recent,
                model.metrics_json.get("train_brier"),
                probs,
                labels_,
            )
            if rep.flagged:
                lowered = rt.downgrade_on_drift(strategy, rep.reasons)
                ctx.notifier(
                    "TradePartner drift",
                    f"{strategy}: {'; '.join(rep.reasons)}"
                    + (" (set to MANUAL)" if lowered else ""),
                )
            notes.append(f"{strategy}: {'DRIFT' if rep.flagged else 'ok'}")
    return "; ".join(notes) or "no approved models"


def job_weekly_retrain(ctx: JobContext) -> str:
    """Saturday: retrain each ML strategy -> new candidate -> shadow; evaluate shadow candidates."""
    notes = []
    with ctx.session_factory() as s:
        view, _ = _view(ctx, s)
        if not view.symbols or view.index is None:
            return "no market data"
        panel = news_features.news_panel(s, view.symbols, view.trading_days(), ctx.settings.app.llm)
        now = ctx.clock()
        for strategy in ML_STRATEGIES:
            out = pipeline.train_strategy(
                s, strategy, view, ctx.settings, now, news_panel=panel, reports_root=reports_dir()
            )
            if out.model_id is not None:
                registry.start_shadow(s, out.model_id, now)
            notes.append(f"{strategy}: candidate {out.model_id} ({out.chosen})")
        for strategy in ML_STRATEGIES:
            approved = registry.current_approved(s, strategy)
            for cand in registry.by_status(s, strategy, "shadow"):
                shadow.shadow_comparison(s, cand, approved, now)
    ctx.notifier("TradePartner weekly retrain", "; ".join(notes))
    return "; ".join(notes)


def _protective(ctx: JobContext, s: Session) -> tuple[Runtime, ProtectiveManager]:
    rt = ctx.runtime(s)
    pm = ProtectiveManager(
        s, rt.broker, lambda sym: (q.ltp_paise if (q := ctx.quotes.get(sym)) else None),
        ctx.notifier, ctx.clock,
    )  # fmt: skip
    return rt, pm


def job_protective_check(ctx: JobContext) -> str:
    """Every 5 minutes: every open bot CNC position must have a live GTT OCO at the broker."""
    if ctx.settings.app.broker != "kite":
        return "paper broker: GTT is simulated, nothing to verify"
    with ctx.session_factory() as s:
        rt, pm = _protective(ctx, s)
        # Keeping a bot position's stop alive is protective, so it runs in every mode: a halt,
        # a kill or a restart must not leave a partial fill unresized or a GTT missing.
        report = pm.check(True)
        for it in protective.intents(s):
            q = ctx.quotes.get(it.symbol)
            if report.get(it.symbol) in protective.EXIT_STATUSES and q is not None:
                placed = _protective_exit(
                    ctx,
                    rt,
                    it.symbol,
                    "CNC",
                    it.qty,
                    it.strategy or "protective",
                    "stop",
                    q.ltp_paise,
                )
                report[it.symbol] += " -> exit placed" if placed else " -> EXIT NOT PLACED"
    bad = {k: v for k, v in report.items() if v not in ("OK", "CREATED", "RESIZED")}
    return (
        f"{len(report)} positions checked; problems: {bad}"
        if bad
        else f"{len(report)} positions OK"
    )


def job_poll_fills(ctx: JobContext) -> str:
    """Every 30 s: pull fills so a new entry gets its GTT immediately (AUTO) or an alert."""
    kite = ctx.kite()
    if kite is None or ctx.settings.app.broker != "kite":
        return "skipped"
    with ctx.session_factory() as s:
        res = ledger_sync.sync(s, kite, ctx.today(), str(kite.profile().get("user_id", "kite")),
                               ctx.settings.risk.per_trade.limit_price_band_pct)  # fmt: skip
        if res.fills:
            rebuild.rebuild(s, ctx.settings.costs, ctx.settings.tax)
            job_protective_check(ctx)
    return f"fills+{res.fills}"


def job_token_check(ctx: JobContext) -> str:
    """09:10: no valid token -> notify and never stay in AUTO (never a headless login)."""
    if kite_auth.load_access_token(ctx.today()) is not None:
        return "token ok"
    with ctx.session_factory() as s:
        lowered = ctx.runtime(s).no_token_safeguard()
    ctx.notifier(
        "TradePartner", "Kite token missing or expired: run `tradepartner login`. AUTO disabled."
    )
    return f"no token; lowered {lowered}"


JOBS: list[JobSpec] = [
    JobSpec("health_check", job_health_check, "07:45", catch_up=True, daily_at=(7, 45)),
    JobSpec("refresh_reference", job_refresh_reference, "08:00", catch_up=True, daily_at=(8, 0)),
    JobSpec("results_calendar", job_results_calendar, "08:05", catch_up=True, daily_at=(8, 5)),
    JobSpec("news_fetch", job_news_fetch, "every 15 min", catch_up=True, daily_at=(6, 0)),
    JobSpec("news_score", job_news_score, "08:10", heavy=True, catch_up=True, daily_at=(8, 10)),
    JobSpec(
        "generate_signals",
        job_generate_signals,
        "08:50",
        heavy=True,
        catch_up=True,
        daily_at=(8, 50),
    ),
    JobSpec("notify_proposals", job_notify_proposals, "09:00"),
    JobSpec("token_check", job_token_check, "09:10"),
    JobSpec("poll_quotes", job_poll_quotes, "every 10 s, 09:15-15:30", market_hours=True),
    JobSpec("submit_entries", job_submit_entries, "09:20"),  # deliberately NOT catch-up
    JobSpec("track_limits", job_track_limits, "every 30 s, 09:15-15:30", market_hours=True),
    JobSpec("poll_fills", job_poll_fills, "every 30 s, 09:15-15:30", market_hours=True),
    JobSpec(
        "protective_check", job_protective_check, "every 5 min, 09:15-15:30", market_hours=True
    ),
    JobSpec("intraday_scan", job_intraday_scan, "every 5 min, 09:35-14:30", market_hours=True),
    JobSpec("mis_protect", job_mis_protect, "every 30 s, 09:15-15:30", market_hours=True),
    JobSpec("expire_entries", job_expire_entries, "15:00"),
    JobSpec("square_off_mis", job_square_off_mis, "15:10"),
    JobSpec("sync_ledger", job_sync_ledger, "15:45", catch_up=True, daily_at=(15, 45)),
    JobSpec("eod_candles", job_eod_candles, "16:30", catch_up=True, daily_at=(16, 30)),
    JobSpec("drift_check", job_drift_check, "16:45", catch_up=True, daily_at=(16, 45)),
    JobSpec("daily_report", job_daily_report, "17:00", catch_up=True, daily_at=(17, 0)),
    JobSpec("backup", job_backup, "17:05", catch_up=True, daily_at=(17, 5)),
    JobSpec("weekly_retrain", job_weekly_retrain, "Sat 10:00", heavy=True),
]
BY_NAME = {j.name: j for j in JOBS}


def last_proposal_time(session: Session) -> datetime | None:
    p = session.scalar(select(OrderProposal).order_by(OrderProposal.id.desc()).limit(1))
    return p.created_at if p else None


def model_summary(session: Session) -> list[tuple[str, str, int]]:
    return [(m.strategy, m.status, m.id) for m in session.scalars(select(Model).order_by(Model.id))]
