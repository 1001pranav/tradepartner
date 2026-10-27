"""Typer CLI skeleton (spec section 5). Commands beyond M0 are stubs for their milestones."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import typer
from kiteconnect import KiteConnect
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from tradepartner import jobs as jobs_mod
from tradepartner import live_test, scheduler
from tradepartner.appctx import build_job_context, kite_or_none
from tradepartner.calendar import IST, TradingCalendar, utc_now
from tradepartner.data import (
    candles,
    corporate_actions,
    instruments,
    kite_auth,
    kite_client,
    quality,
)
from tradepartner.data import universe as universe_mod
from tradepartner.data.quotes import poll
from tradepartner.db import audit
from tradepartner.db.migrate import upgrade
from tradepartner.db.models import AuditLog, Instrument, Model, NewsRaw, NewsScore, ResultDate
from tradepartner.db.session import make_engine, make_session_factory
from tradepartner.jobs import JobContext
from tradepartner.ledger import importers, rebuild, reconcile
from tradepartner.ledger import sync as ledger_sync
from tradepartner.llm.client import OllamaClient
from tradepartner.logging import configure_logging
from tradepartner.models import approval, registry
from tradepartner.modes.controller import ModeController, ModeError
from tradepartner.modes.evidence import collect_unlock_evidence
from tradepartner.modes.state import GLOBAL, strategy_scope
from tradepartner.news import ingest as ingest_mod
from tradepartner.news.analyst import NewsAnalyst
from tradepartner.news.entity_link import AliasTable, Ambiguous
from tradepartner.news.sources.base import NewsSource
from tradepartner.news.sources.bse import BseAnnouncements
from tradepartner.news.sources.gdelt import GdeltSource
from tradepartner.news.sources.nse import NseAnnouncements
from tradepartner.news.sources.rss import RssSource
from tradepartner.paths import db_path, home_dir, kill_file, reports_dir
from tradepartner.reports import tax_export
from tradepartner.reports.text import save_text_report
from tradepartner.risk import kill as kill_mod
from tradepartner.runtime import build_runtime
from tradepartner.settings import (
    ConfigError,
    Settings,
    StrategyConfig,
    load_settings,
    require_ready,
)

app = typer.Typer(help="TradePartner: local AI-assisted trading partner for NSE cash equities.")
db_app = typer.Typer(help="Database commands.")
data_app = typer.Typer(help="Market data commands (M1).")
ledger_app = typer.Typer(help="Ledger commands (M2).")
report_app = typer.Typer(help="Reports (M2/M7).")
news_app = typer.Typer(help="News ingestion and the LLM analyst (M5).")
app.add_typer(db_app, name="db")
app.add_typer(data_app, name="data")
app.add_typer(ledger_app, name="ledger")
app.add_typer(report_app, name="report")
app.add_typer(news_app, name="news")


def _startup() -> Settings:
    """Load config and enforce the startup gate. Exits non-zero if unsafe."""
    try:
        settings = load_settings()
        require_ready(settings)
    except ConfigError as exc:
        typer.secho(f"Refusing to start: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    return settings


def _startup_config() -> Settings:
    try:
        return load_settings()
    except ConfigError as exc:
        typer.secho(f"Config error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


def _session() -> Session:
    home_dir().mkdir(parents=True, exist_ok=True)
    engine = make_engine()
    upgrade(engine.url.render_as_string(hide_password=False))
    return make_session_factory(engine)()


def _kite() -> KiteConnect:
    try:
        return kite_client.get_kite()
    except kite_client.NotLoggedInError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


def _not_yet(milestone: str) -> None:
    typer.echo(f"Not implemented yet (planned for {milestone}).")
    raise typer.Exit(code=1)


@app.callback()
def main() -> None:
    configure_logging(to_file=False)


@app.command()
def check() -> None:
    """Validate all config files and the startup gate."""
    _startup()
    typer.echo("Config OK.")


@app.command()
def login() -> None:
    """Kite login via a local redirect server (token to Keychain). Always manual."""
    import webbrowser

    settings = _startup_config()
    creds = kite_auth.load_credentials()
    if creds is None:
        key = typer.prompt("Kite api key")
        secret = typer.prompt("Kite api secret", hide_input=True)
        kite_auth.save_credentials(key, secret)
        creds = (key, secret)
    kite = kite_client.login_client()
    typer.echo(
        f"Opening the Kite login page. Redirect URL must be http://127.0.0.1:"
        f"{settings.app.kite.redirect_port}/"
    )
    webbrowser.open(kite.login_url())
    try:
        token = kite_auth.wait_for_request_token(settings.app.kite.redirect_port)
        kite_auth.complete_login(kite, creds[1], token)
    except kite_auth.LoginError as exc:
        typer.secho(f"Login failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("Logged in. Token stored in Keychain (valid until tomorrow morning).")


def _job_context(settings: Settings) -> JobContext:
    return build_job_context(settings, lambda: _llm(settings))


@app.command()
def run() -> None:
    """Start the scheduler (never in AUTO: every mode is clamped to MANUAL at start)."""
    settings = _startup()
    ctx = _job_context(settings)
    with ctx.session_factory() as session:
        clamped = ModeController(session).startup(utc_now())
    if clamped:
        typer.secho(
            f"Startup clamped AUTO -> MANUAL for {clamped}. Re-enable AUTO yourself.",
            fg=typer.colors.YELLOW,
        )
    scheduler.validate()
    typer.echo("Scheduler starting (IST, trading days). Ctrl-C to stop.")
    scheduler.start(ctx)


job_app = typer.Typer(help="Run or list scheduled jobs (M7).")
app.add_typer(job_app, name="job")


@job_app.command("list")
def job_list() -> None:
    """The daily schedule."""
    for spec in jobs_mod.JOBS:
        flags = ("heavy " if spec.heavy else "") + ("catch-up" if spec.catch_up else "")
        typer.echo(f"{spec.schedule:28s} {spec.name:20s} {flags}")


@job_app.command("run")
def job_run(name: str) -> None:
    """Run one job now (idempotent catch-up, or a manual re-run)."""
    spec = jobs_mod.BY_NAME.get(name)
    if spec is None:
        typer.secho(f"unknown job {name!r}; see `job list`", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    settings = _startup()
    status = jobs_mod.run_job(_job_context(settings), spec)
    typer.echo(f"{name}: {status}")
    if status == "error":
        raise typer.Exit(code=1)


@app.command()
def kill() -> None:
    """KILL SWITCH: halt everything, cancel entries, square off bot MIS positions."""
    kill_mod.arm()  # first and unconditionally: running processes also watch this file
    typer.secho(f"Kill file created at {kill_file()}.", fg=typer.colors.RED)
    try:
        settings = load_settings()
        with _session() as session:
            rt = build_runtime(session, settings)
            report = rt.kill.trigger(utc_now(), "CLI kill")
        typer.echo(
            f"Global mode OFF. Cancelled {len(report.cancelled)} entry orders; "
            f"squared off {report.squared_off or 'nothing'}; skipped {report.skipped or 'none'}."
        )
    except Exception as exc:  # the file is armed regardless; report loudly
        typer.secho(
            f"Kill procedure error (file still armed): {exc}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1) from exc


@app.command()
def mode(
    scope: str = typer.Argument("show", help="'show', 'global' or a strategy name"),
    value: str = typer.Argument("", help="OFF | MANUAL | AUTO"),
    reason: str = typer.Option("cli", help="Why (written to the audit log)"),
) -> None:
    """Show modes, or set one: `tradepartner mode global MANUAL`, `mode swing_v1 AUTO`."""
    with _session() as session:
        ctl = ModeController(session)
        if scope == "show":
            for sc in [GLOBAL, *[strategy_scope(n) for n in sorted(load_settings().strategies)]]:
                row = ctl.current(sc)
                lock = (
                    ""
                    if sc == GLOBAL
                    else (" (AUTO unlocked)" if row.auto_unlocked else " (AUTO locked)")
                )
                typer.echo(f"{sc:24s} {row.mode}{lock}")
            return
        target = GLOBAL if scope == "global" else strategy_scope(scope)
        try:
            ctl.set_mode(target, value.upper(), "user", reason, utc_now())  # type: ignore[arg-type]
        except ModeError as exc:
            typer.secho(f"Refused: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        typer.echo(f"{target} -> {value.upper()}")


@app.command("unlock-auto")
def unlock_auto(strategy: str) -> None:
    """Unlock AUTO for a strategy (needs evidence, and the strategy name typed to confirm)."""
    with _session() as session:
        ctl = ModeController(session)
        ev = collect_unlock_evidence(session, strategy)
        typer.echo(f"Evidence: {ev}")
        confirm = typer.prompt(f"Type the strategy name '{strategy}' to confirm")
        try:
            ctl.unlock_auto(strategy, ev, confirm, utc_now())
        except ModeError as exc:
            typer.secho(f"Refused: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
    typer.echo(f"AUTO unlocked for {strategy}. Enable it with `tradepartner mode {strategy} AUTO`.")


@app.command()
def backtest(
    strategy: str = typer.Option("baseline_momentum20", help="Strategy name"),
    universe: str = typer.Option("nifty100"),
    start: str = typer.Option("", help="YYYY-MM-DD (default: first stored day)"),
    end: str = typer.Option("", help="YYYY-MM-DD (default: last stored day)"),
    slippage_bps: float = typer.Option(-1.0, help="Override config slippage (bps)"),
    capital_inr: int = typer.Option(0, help="Capital for this backtest if config has 0"),
) -> None:
    """Run the backtester (same RiskEngine as live) and write an HTML report."""
    from datetime import date as _date

    from tradepartner.backtest import benchmarks, loader, metrics
    from tradepartner.backtest import report as bt_report
    from tradepartner.backtest.engine import run_backtest
    from tradepartner.strategies import registry

    settings = _startup_config()
    if capital_inr:
        settings = settings.model_copy(
            update={"risk": settings.risk.model_copy(update={"capital_base_inr": capital_inr})}
        )
    if settings.risk.capital_base_inr <= 0:
        typer.secho("Set capital_base_inr (or pass --capital-inr).", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    try:
        strat = registry.get(strategy)
    except KeyError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    store = candles.CandleStore()
    with _session() as session:
        names = universe_mod.all_symbols(session, universe)
        members = universe_mod.membership_fn(session, universe)
        biased = universe_mod.has_survivorship_bias(session, universe)
        acts: dict[str, list[corporate_actions.Action]] = {}
        for a in corporate_actions.load_actions(session):
            acts.setdefault(a.symbol, []).append(a)
    view = loader.load_view(store, names, acts, members=members)
    if not view.symbols:
        typer.secho(
            "No stored candles. Run `tradepartner data backfill` first.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    first, last = loader.first_last(view)
    s_date = _date.fromisoformat(start) if start else first
    e_date = _date.fromisoformat(end) if end else last
    result = run_backtest(
        strat,
        view,
        settings,
        s_date,
        e_date,
        slippage_bps=None if slippage_bps < 0 else slippage_bps,
        survivorship_warning=biased,
    )
    cap = settings.risk.capital_base_paise
    base: dict[str, metrics.Metrics] = {}
    bh = benchmarks.buy_and_hold(view, s_date, e_date, cap)
    if len(bh) > 1:
        base["nifty50_buy_and_hold"] = metrics.compute(bh, [])
    ew = benchmarks.equal_weight_universe(
        view, s_date, e_date, cap, settings.costs.slippage.default_bps
    )
    if len(ew) > 1:
        base["equal_weight_universe"] = metrics.compute(ew, [])
    idx_close = None if view.index is None else view.index.set_index("date")["close"]
    html = bt_report.render(result, base, idx_close)
    path = bt_report.save(html, reports_dir(), f"backtest-{strategy}")
    m = result.metrics
    typer.echo(
        f"{strategy}: CAGR {m.cagr:.1%}  Sharpe {m.sharpe:.2f}  maxDD {m.max_drawdown:.1%}  "
        f"trades {m.n_trades}  cost drag {m.cost_drag:.1%}"
    )
    for w in result.warnings:
        typer.secho(w, fg=typer.colors.YELLOW)
    typer.echo(f"Report: {path}")


@app.command()
def train(
    strategy: str = typer.Option("swing_v1", help="swing_v1 | positional_v1"),
    universe: str = typer.Option("nifty100"),
    trials: int = typer.Option(-1, help="Optuna trials per fold (max 50); default from config"),
    news: bool = typer.Option(True, help="Also train/evaluate the with-news variant if possible"),
    capital_inr: int = typer.Option(0, help="Only used for cost scaling; config value otherwise"),
) -> None:
    """Walk-forward training and evaluation. Registers a CANDIDATE; never approves anything."""
    from tradepartner.backtest import loader
    from tradepartner.features import news as news_features
    from tradepartner.models import pipeline

    settings = _startup_config()
    if trials >= 0:
        cfg = settings.strategies.get(strategy)
        merged = {**(cfg.model_dump() if cfg else {"name": strategy}), "n_trials": min(trials, 50)}
        settings = settings.model_copy(
            update={"strategies": {**settings.strategies, strategy: StrategyConfig(**merged)}}
        )
    store = candles.CandleStore()
    now = utc_now()
    if strategy == "intraday_v1":
        _train_intraday(settings, store, universe, trials, now)
        return
    with _session() as session:
        names = universe_mod.all_symbols(session, universe)
        members = universe_mod.membership_fn(session, universe)
        biased = universe_mod.has_survivorship_bias(session, universe)
        acts: dict[str, list[corporate_actions.Action]] = {}
        for a in corporate_actions.load_actions(session):
            acts.setdefault(a.symbol, []).append(a)
        view = loader.load_view(store, names, acts, members=members)
        if not view.symbols or view.index is None:
            typer.secho(
                "Need stored candles and the NIFTY 50 index: run `data backfill`.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        panel = None
        if news:
            panel = news_features.news_panel(
                session, view.symbols, view.trading_days(), settings.app.llm
            )
        try:
            out = pipeline.train_strategy(
                session,
                strategy,
                view,
                settings,
                now,
                news_panel=panel,
                survivorship_warning=biased,
                reports_root=reports_dir(),
            )
        except registry.ModelError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from exc
    for v in out.variants.values():
        m = v.gate.metrics
        typer.echo(
            f"{v.name}: Sharpe {m['sharpe']:.2f} vs baseline {m['baseline_sharpe']:.2f}, "
            f"trades {m['trades']}, ready for review: {v.gate.ready_for_review}"
            + (f" (fails: {', '.join(v.gate.failed)})" if v.gate.failed else "")
        )
        if v.llm_period:
            typer.echo(
                f"  LLM features were tested only for {v.llm_period[0]} to {v.llm_period[1]}"
            )
    for w in out.warnings:
        typer.secho(w, fg=typer.colors.YELLOW)
    typer.echo(f"chosen: {out.chosen}; candidate model id {out.model_id}. Nothing was approved.")


@app.command()
def replay(
    strategy: str = typer.Option("swing_v1", help="swing_v1 (5-day picks)"),
    as_of: str = typer.Option(
        ..., "--as-of", help="YYYY-MM-DD: train only on data before this day"
    ),
    days: int = typer.Option(5, help="How many trading days to reveal after as_of"),
    universe: str = typer.Option("nifty100"),
    top: int = typer.Option(5, help="How many top picks to show"),
) -> None:
    """Hide the future, predict it, reveal what happened. Registers and trades nothing."""
    from datetime import date as _date

    from tradepartner.backtest import loader
    from tradepartner.models import replay as replay_mod

    settings = _startup_config()
    day = _date.fromisoformat(as_of)
    store = candles.CandleStore()
    with _session() as session:
        names = universe_mod.all_symbols(session, universe)
        members = universe_mod.membership_fn(session, universe)
        acts: dict[str, list[corporate_actions.Action]] = {}
        for a in corporate_actions.load_actions(session):
            acts.setdefault(a.symbol, []).append(a)
    view = loader.load_view(store, names, acts, members=members)
    try:
        res = replay_mod.replay(view, settings, day, days, horizon=5, top_k=top)
    except registry.ModelError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    out_lines: list[str] = []

    def emit(msg: str) -> None:
        out_lines.append(msg)
        typer.echo(msg)

    emit(
        f"Trained on {res.train_rows} rows up to {res.train_last_date} "
        f"(never saw anything after that). Predicting {res.days} trading days after {res.as_of}."
    )
    emit(f"{'stock':12s} {'score':>7s} {'actual':>9s} {'vs Nifty':>9s}")
    for p in res.picks:
        act = "n/a" if p.actual_return is None else f"{p.actual_return:+.1%}"
        vs = "n/a" if p.vs_index is None else f"{p.vs_index:+.1%}"
        emit(f"{p.symbol:12s} {p.score:7.3f} {act:>9s} {vs:>9s}")
    idx = "n/a" if res.index_return is None else f"{res.index_return:+.1%}"
    mean = "n/a" if res.universe_mean is None else f"{res.universe_mean:+.1%}"
    emit(f"Nifty 50 over the same days: {idx}.  Average of all {len(view.symbols)} stocks: {mean}.")
    if res.hit_rate is not None:
        emit(
            f"Picks that beat Nifty: {res.hit_rate:.0%}. "
            "One week proves nothing; repeat over many dates."
        )
    path = save_text_report(f"replay-{strategy}", f"Replay {strategy} as of {day}", out_lines)
    typer.echo(f"Saved report: {path}")


@app.command("paper-sim")
def paper_sim(
    strategy: str = typer.Option("positional_v1", help="swing_v1 | positional_v1"),
    start: str = typer.Option(
        ..., help="YYYY-MM-DD first day to trade (needs ~2 years of data before)"
    ),
    end: str = typer.Option("", help="YYYY-MM-DD (default: last stored day)"),
    capital_inr: int = typer.Option(1_000_000, help="Simulated starting capital"),
    retrain_days: int = typer.Option(21, help="Retrain the model every N trading days"),
    universe: str = typer.Option("nifty100"),
) -> None:
    """HISTORICAL PAPER TRADING: replay the past with a model that retrains only on what it
    could have known, through the real risk engine, costs and slippage. Simulated: no broker."""
    from datetime import date as _date

    from tradepartner.backtest import benchmarks, loader, metrics
    from tradepartner.backtest import report as bt_report
    from tradepartner.backtest.engine import run_backtest
    from tradepartner.models import pipeline
    from tradepartner.strategies.walkforward import WalkForwardModel

    settings = _startup_config()
    settings = settings.model_copy(
        update={"risk": settings.risk.model_copy(update={"capital_base_inr": capital_inr})}
    )
    try:
        p = pipeline.strategy_params(settings, strategy)
    except registry.ModelError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    store = candles.CandleStore()
    with _session() as session:
        names = universe_mod.all_symbols(session, universe)
        members = universe_mod.membership_fn(session, universe)
        biased = universe_mod.has_survivorship_bias(session, universe)
        acts: dict[str, list[corporate_actions.Action]] = {}
        for a in corporate_actions.load_actions(session):
            acts.setdefault(a.symbol, []).append(a)
    view = loader.load_view(store, names, acts, members=members)
    if view.index is None or not view.symbols:
        typer.secho("Need stored candles and the NIFTY 50 index.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    s_date = _date.fromisoformat(start)
    e_date = _date.fromisoformat(end) if end else loader.first_last(view)[1]
    typer.echo(f"Training labels and features for {strategy} (this takes a few minutes)...")
    lab = pipeline._labels(view, p, settings)
    p = {
        **p,
        "min_prob": p.get("min_prob", 0.45),
        "max_stop_pct": 7.0 if strategy == "swing_v1" else 8.0,
    }
    wf = WalkForwardModel(strategy, view, lab, p, p["objective"], retrain_days)
    result = run_backtest(wf, view, settings, s_date, e_date, survivorship_warning=biased)
    cap = settings.risk.capital_base_paise
    base: dict[str, metrics.Metrics] = {}
    bh = benchmarks.buy_and_hold(view, s_date, e_date, cap)
    if len(bh) > 1:
        base["nifty50_buy_and_hold"] = metrics.compute(bh, [])
    ew = benchmarks.equal_weight_universe(
        view, s_date, e_date, cap, settings.costs.slippage.default_bps
    )
    if len(ew) > 1:
        base["equal_weight_universe"] = metrics.compute(ew, [])
    idx_close = view.index.set_index("date")["close"]
    path = bt_report.save(
        bt_report.render(result, base, idx_close, "Historical paper trading"),
        reports_dir(),
        f"paper-sim-{strategy}",
    )
    m = result.metrics
    typer.echo(
        f"{strategy} paper-traded {s_date} to {e_date}: CAGR {m.cagr:.1%}  Sharpe {m.sharpe:.2f}  "
        f"maxDD {m.max_drawdown:.1%}  trades {m.n_trades}  cost drag {m.cost_drag:.1%}"
    )
    for name, bm in base.items():
        typer.echo(
            f"  {name}: CAGR {bm.cagr:.1%}  Sharpe {bm.sharpe:.2f}  maxDD {bm.max_drawdown:.1%}"
        )
    typer.echo(
        f"Model retrained {len(wf.retrains)} times; last used data up to "
        f"{wf.retrains[-1][1] if wf.retrains else 'n/a'}."
    )
    for w in result.warnings:
        typer.secho(w, fg=typer.colors.YELLOW)
    typer.echo(f"Simulated only: no broker, no orders. Report: {path}")


@app.command("ew-sim")
def ew_sim(
    start: str = typer.Option("2017-01-02", help="YYYY-MM-DD"),
    end: str = typer.Option("", help="YYYY-MM-DD (default: last stored day)"),
    capital_inr: int = typer.Option(1_000_000, help="Simulated starting capital"),
    universe: str = typer.Option("nifty100"),
) -> None:
    """Equal-weight portfolio with real costs and Indian capital-gains tax (simulation only)."""
    from datetime import date as _date

    from tradepartner.backtest import benchmarks, loader, metrics
    from tradepartner.backtest.equalweight import run_equal_weight

    settings = _startup_config()
    store = candles.CandleStore()
    with _session() as session:
        names = universe_mod.all_symbols(session, universe)
        members = universe_mod.membership_fn(session, universe)
        biased = universe_mod.has_survivorship_bias(session, universe)
        acts: dict[str, list[corporate_actions.Action]] = {}
        for a in corporate_actions.load_actions(session):
            acts.setdefault(a.symbol, []).append(a)
    view = loader.load_view(store, names, acts, members=members)
    if not view.symbols or view.index is None:
        typer.secho("Need stored candles and the NIFTY 50 index.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    s_date = _date.fromisoformat(start)
    e_date = _date.fromisoformat(end) if end else loader.first_last(view)[1]
    cap = capital_inr * 100
    out_lines: list[str] = []

    def emit(msg: str) -> None:
        out_lines.append(msg)
        typer.echo(msg)

    def emit_warn(msg: str, fg: str = typer.colors.YELLOW) -> None:
        out_lines.append(msg)
        typer.secho(msg, fg=fg)

    head = ("portfolio", "CAGR", "Sharpe", "maxDD", "trades", "costs", "tax", "final INR")
    emit(f"{head[0]:22s} " + " ".join(f"{h:>10s}" for h in head[1:]))

    def row(label: str, m: metrics.Metrics, trades: str, cost: str, tax: str, final: float) -> None:
        cells = [
            f"{m.cagr:.1%}",
            f"{m.sharpe:.2f}",
            f"{m.max_drawdown:.1%}",
            trades,
            cost,
            tax,
            f"{final:,.0f}",
        ]
        emit(f"{label:22s} " + " ".join(f"{c:>10s}" for c in cells))

    for freq in ("monthly", "quarterly", "annual", "never"):
        r = run_equal_weight(view, settings, s_date, e_date, cap, freq)
        costs = f"{(r.charges_paise + r.slippage_paise) / 100:,.0f}"
        row(
            f"equal-weight {freq}",
            r.metrics,
            str(r.trades),
            costs,
            f"{r.tax_paise / 100:,.0f}",
            r.equity.iloc[-1] / 100,
        )
    nf = benchmarks.buy_and_hold(view, s_date, e_date, cap)
    row("Nifty 50 (no tax/cost)", metrics.compute(nf, []), "-", "-", "-", nf.iloc[-1] / 100)
    emit("Costs = Zerodha charges + slippage; tax = STCG/LTCG per config/tax.yaml. Simulated only.")
    if biased:
        emit_warn(
            "WARNING: today's Nifty 100 applied to history (survivorship bias): optimistic.",
            fg=typer.colors.YELLOW,
        )
    path = save_text_report("ew-sim", f"Equal-weight simulation {s_date} to {e_date}", out_lines)
    typer.echo(f"Saved report: {path}")


def _train_intraday(
    settings: Settings, store: candles.CandleStore, universe: str, trials: int, now: datetime
) -> None:
    from tradepartner.models import intraday as intraday_models

    with _session() as session:
        names = universe_mod.members_as_of(session, universe, now.date())
        bars = {s: store.read("5minute", s) for s in names}
        bars = {k: v for k, v in bars.items() if len(v)}
        if not bars:
            typer.secho(
                "No 5-minute bars: run `tradepartner data backfill --interval 5minute`.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        try:
            rep = intraday_models.train_intraday(
                session, bars, settings, now, n_trials=trials if trials >= 0 else 10
            )
        except registry.ModelError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from exc
    gate = cast(dict[str, Any], rep["gate"])
    typer.echo(
        f"intraday_v1: ready for review: {gate['ready_for_review']} (fails: {gate['failed']})"
    )
    typer.echo("Candidate registered. Intraday stays PAPER-ONLY; nothing was approved.")


@app.command()
def sync() -> None:
    """Sync orders, trades, positions and holdings from Kite, rebuild the ledger, reconcile."""
    settings = _startup_config()
    kite = _kite()
    now = utc_now()
    today = now.astimezone(IST).date()
    with _session() as session:
        profile = kite.profile()
        res = ledger_sync.sync(
            session,
            kite,
            today,
            str(profile.get("user_id", "kite")),
            settings.risk.per_trade.limit_price_band_pct,
        )
        rebuilt = rebuild.rebuild(session, settings.costs, settings.tax)
        mism = reconcile.reconcile_positions(session, today)
    typer.echo(
        f"orders+{res.orders} fills+{res.fills} snapshots={res.snapshots} "
        f"linked_proposals={res.linked_proposals} skipped_non_equity={res.skipped_non_equity}"
    )
    for alert in rebuilt.alerts:
        typer.secho(f"ALERT: {alert}", fg=typer.colors.YELLOW)
    for m in mism:
        typer.secho(
            f"MISMATCH {m.symbol}: ledger {m.ledger_qty} vs broker {m.broker_qty} ({m.reason})",
            fg=typer.colors.RED,
        )


@db_app.command("upgrade")
def db_upgrade() -> None:
    """Create or migrate the SQLite schema to the latest revision."""
    home_dir().mkdir(parents=True, exist_ok=True)
    engine = make_engine()
    upgrade(engine.url.render_as_string(hide_password=False))
    typer.echo(f"Database at {db_path()} is up to date.")


@data_app.command("instruments")
def data_instruments() -> None:
    """Refresh the NSE equity instrument master from Kite."""
    kite = _kite()
    with _session() as session:
        n = instruments.refresh_instruments(session, kite, utc_now())
    typer.echo(f"{n} NSE equity instruments stored.")


@data_app.command("seed-universe")
def data_seed_universe(
    universe: str = "nifty100",
    replace: bool = typer.Option(False, help="Drop seeded symbols that left the list"),
) -> None:
    """Seed universe membership from config/universe/<name>.csv."""
    with _session() as session:
        n = universe_mod.seed_universe(session, universe, replace=replace)
    typer.echo(f"{n} members added to {universe}. {universe_mod.SURVIVORSHIP_WARNING}")


@data_app.command("actions-import")
def data_actions_import(path: Path) -> None:
    """Import corporate actions from a CSV (idempotent)."""
    with _session() as session:
        n = corporate_actions.import_actions_csv(session, path)
    typer.echo(f"{n} corporate actions imported.")


@data_app.command("backfill")
def data_backfill(
    universe: str = typer.Option("nifty100"),
    years: int = typer.Option(10, min=1, max=20),
    interval: str = typer.Option("day"),
) -> None:
    """Backfill historical candles for a universe (resumable)."""
    kite = _kite()
    cal = TradingCalendar.from_config()
    now = utc_now()
    with _session() as session:
        if not universe_mod.members_as_of(session, universe, now.date()):
            universe_mod.seed_universe(session, universe)
        instruments.refresh_instruments(session, kite, now)
        targets = universe_mod.tradable_members(session, universe, now.date())
        if universe_mod.has_survivorship_bias(session, universe):
            typer.secho(universe_mod.SURVIVORSHIP_WARNING, fg=typer.colors.YELLOW)
    targets.update(kite_client.INDEX_TOKENS)
    result = candles.backfill(
        kite,
        candles.CandleStore(),
        targets,
        years=years,
        now=now,
        calendar=cal,
        limiter=candles.RateLimiter(2.0),  # self-limit below Kite's 3 req/s
        interval=interval,
    )
    typer.echo(
        f"done={len(result.done)} skipped={len(result.skipped)} failed={len(result.failed)} "
        f"rows={result.rows}"
    )
    for sym, err in result.failed.items():
        typer.secho(f"  FAILED {sym}: {err}", fg=typer.colors.RED, err=True)
    if result.failed:
        raise typer.Exit(code=1)


@data_app.command("refetch")
def data_refetch(
    symbol: list[str] = typer.Option([], "--symbol", help="Limit to these symbols (repeatable)"),
    start: str = typer.Option("", "--from", help="YYYY-MM-DD (only with --symbol)"),
    end: str = typer.Option("", "--to", help="YYYY-MM-DD (only with --symbol)"),
) -> None:
    """Re-download missing days INSIDE stored history (read-only market data, no orders)."""
    from datetime import date as _date

    kite = _kite()
    cal = TradingCalendar.from_config()
    store = candles.CandleStore()
    with _session() as session:
        # every stored symbol, not just today's members: past members have history to repair too
        rows = session.execute(
            select(Instrument.tradingsymbol, Instrument.instrument_token).where(
                Instrument.tradingsymbol.in_(store.symbols("day")), Instrument.segment == "NSE"
            )
        )
        targets = {sym: tok for sym, tok in rows}
    targets.update(kite_client.INDEX_TOKENS)
    only: dict[str, tuple[_date, _date]] | None = None
    if symbol:
        unknown = [s for s in symbol if s not in targets]
        if unknown:
            typer.secho(f"Unknown symbols: {', '.join(unknown)}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        lo = _date.fromisoformat(start) if start else _date.min
        hi = _date.fromisoformat(end) if end else _date.max
        only = {s: (lo, hi) for s in symbol}
        targets = {s: targets[s] for s in symbol}
    filled, still, failed = candles.refill_gaps(
        kite, store, targets, calendar=cal, limiter=candles.RateLimiter(2.0), only=only
    )
    for sym, days in filled.items():
        if days:
            typer.echo(f"{sym}: filled {', '.join(map(str, days))}")
    for sym, days in still.items():
        if days:
            typer.secho(
                f"{sym}: Kite has no bar for {', '.join(map(str, days))}", fg=typer.colors.YELLOW
            )
    for sym, err in failed.items():
        typer.secho(f"  FAILED {sym}: {err}", fg=typer.colors.RED, err=True)
    typer.echo(
        f"filled {sum(len(v) for v in filled.values())} days, "
        f"{sum(len(v) for v in still.values())} not available from Kite."
    )
    if failed:
        raise typer.Exit(code=1)


@data_app.command("explain-jumps")
def data_explain_jumps(
    symbol: list[str] = typer.Option([], "--symbol", help="Limit to these symbols"),
    limit: int = typer.Option(0, help="Explain at most this many jumps this run (0 = all)"),
    redo: bool = typer.Option(False, help="Re-explain jumps that already have a reason"),
) -> None:
    """Search the web for the news behind each unexplained jump and let the local LLM name the
    reason from that evidence. Annotation only: changes no data and places no orders."""
    from tradepartner.data import jump_reasons as jr
    from tradepartner.data.corporate_actions import load_actions
    from tradepartner.data.quality import find_jumps

    settings = _startup_config()
    llm = _llm(settings)
    if not llm.health():
        typer.secho("Ollama is not reachable at 127.0.0.1:11434.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    store = candles.CandleStore()
    with _session() as session:
        acts: dict[str, list[Any]] = {}
        for a in load_actions(session):
            acts.setdefault(a.symbol, []).append(a)
        names = {
            sym: (name or sym)
            for sym, name in session.execute(
                select(Instrument.tradingsymbol, Instrument.name).where(Instrument.segment == "NSE")
            )
        }
    universe = [s for s in store.symbols("day") if s not in kite_client.INDEX_TOKENS]
    chosen = symbol or universe
    jumps = {s: j for s in chosen if (j := find_jumps(store.read("day", s), acts.get(s, [])))}
    total = sum(len(v) for v in jumps.values())
    typer.echo(f"{total} unexplained jumps in {len(jumps)} symbols.")
    explained, skipped = jr.explain_all(
        llm,
        jr.WebSearcher(candles.RateLimiter(0.5).acquire),
        store,
        jumps,
        names,
        universe,
        utc_now(),
        limit=limit or None,
        redo=redo,
        progress=typer.echo,
    )
    typer.echo(f"explained {explained}, already done {skipped}. Saved to {jr.reasons_path()}.")


@data_app.command("universe-import")
def data_universe_import(path: Path, universe: str = typer.Option("nifty100")) -> None:
    """Import REAL historical index membership (symbol,valid_from,valid_to), replacing the
    static list that causes survivorship bias."""
    with _session() as session:
        try:
            added, removed = universe_mod.import_membership_csv(session, universe, path)
        except ValueError as exc:
            typer.secho(f"Import failed: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        store = candles.CandleStore()
        have = set(store.symbols("day"))
        missing = [s for s in universe_mod.all_symbols(session, universe) if s not in have]
    typer.echo(f"{added} membership rows added, {removed} static rows removed.")
    if missing:
        typer.secho(
            f"{len(missing)} past members have NO price history (delisted or renamed?): "
            f"{', '.join(missing[:15])}{' ...' if len(missing) > 15 else ''}. Survivorship bias "
            "is only partly removed until their prices are added.",
            fg=typer.colors.YELLOW,
        )


@data_app.command("quality")
def data_quality(universe: str = typer.Option("nifty100")) -> None:
    """Run data quality checks over stored daily candles."""
    store = candles.CandleStore()
    cal = TradingCalendar.from_config()
    with _session() as session:
        acts: dict[str, list[corporate_actions.Action]] = {}
        for a in corporate_actions.load_actions(session):
            acts.setdefault(a.symbol, []).append(a)
    issues = quality.report(store, store.symbols("day"), cal, acts)
    for i in issues:
        typer.echo(f"[{i.severity}] {i.symbol}: {i.kind}: {i.detail}")
    typer.echo(f"{len(issues)} issues across {len(store.symbols('day'))} symbols.")
    if any(i.severity == "error" for i in issues):
        raise typer.Exit(code=1)


@ledger_app.command("rebuild")
def ledger_rebuild() -> None:
    """Rebuild lots and realized P&L from fills (deterministic)."""
    settings = _startup_config()
    with _session() as session:
        res = rebuild.rebuild(session, settings.costs, settings.tax)
    typer.echo(f"Rebuilt: {res.lots} open lots, {res.realized} realized rows.")
    for alert in res.alerts:
        typer.secho(f"ALERT: {alert}", fg=typer.colors.YELLOW)


@ledger_app.command("import-tradebook")
def ledger_import_tradebook(path: Path) -> None:
    """Import a Zerodha Console tradebook CSV (idempotent), then rebuild."""
    settings = _startup_config()
    with _session() as session:
        try:
            n = importers.import_tradebook(session, path)
        except importers.LedgerImportError as exc:
            typer.secho(f"Import failed: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        res = rebuild.rebuild(session, settings.costs, settings.tax)
    typer.echo(f"{n} fills imported; {res.realized} realized rows, {res.lots} open lots.")
    for alert in res.alerts:
        typer.secho(f"ALERT: {alert}", fg=typer.colors.YELLOW)


@ledger_app.command("import-charges")
def ledger_import_charges(path: Path) -> None:
    """Import contract-note charges (they supersede computed ones), then rebuild."""
    settings = _startup_config()
    with _session() as session:
        try:
            n = importers.import_contract_charges(session, path)
        except importers.LedgerImportError as exc:
            typer.secho(f"Import failed: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
        rebuild.rebuild(session, settings.costs, settings.tax)
    typer.echo(f"{n} charge lines imported.")


@ledger_app.command("reconcile")
def ledger_reconcile(
    console_summary: Path = typer.Option(None, help="CSV: fy,category,pnl_inr from Console"),
    acknowledge: str = typer.Option("", help="Acknowledge differences with a reason"),
) -> None:
    """Compare realized P&L with Console Tax P&L (+/-Rs 10 per FY and category)."""
    with _session() as session:
        if acknowledge:
            reconcile.acknowledge(session, acknowledge)
            typer.echo("Acknowledged; AUTO block cleared.")
            return
        if console_summary is None:
            typer.secho("Pass --console-summary or --acknowledge.", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2)
        bad = reconcile.check_realized(session, reconcile.load_console_summary(console_summary))
    for d in bad:
        typer.secho(
            f"{d.fy} {d.category}: ledger {d.ledger_paise / 100:.2f} vs console "
            f"{d.console_paise / 100:.2f} (diff {d.diff_paise / 100:.2f})",
            fg=typer.colors.RED,
        )
    typer.echo("Reconciled." if not bad else f"{len(bad)} differences (AUTO entries blocked).")
    if bad:
        raise typer.Exit(code=1)


@report_app.command("tax")
def report_tax(fy: str = typer.Option(..., help="e.g. 2026-27")) -> None:
    """Tax report for a financial year (CSV + XLSX, for CA review)."""
    settings = _startup_config()
    with _session() as session:
        csv_path, xlsx_path = tax_export.export(session, fy, settings.tax, reports_dir())
    typer.echo(f"Wrote {csv_path} and {xlsx_path}. All suggestions are for CA review.")


@app.command()
def ui() -> None:
    """Launch the Streamlit UI, bound to 127.0.0.1 only."""
    import subprocess
    import sys

    settings = _startup_config()
    target = Path(__file__).parent / "ui" / "app.py"
    raise typer.Exit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(target),
                "--server.address",
                settings.app.ui.host,
                "--server.port",
                str(settings.app.ui.port),
                "--browser.gatherUsageStats",
                "false",
            ],
        )
    )


@report_app.command("daily")
def report_daily() -> None:
    """Daily report."""
    _not_yet("M7")


# ---------------- news (M5) ----------------


def _llm(settings: Settings) -> OllamaClient:
    c = settings.app.llm
    return OllamaClient(c.model, c.num_ctx, c.keep_alive)


def _alias_table(session: Session) -> AliasTable:
    return AliasTable.from_session(session)


@news_app.command("fetch")
def news_fetch(
    sources: str = typer.Option("rss,nse,bse", help="Comma list of rss,nse,bse"),
    hours: int = typer.Option(24, help="Look back this many hours"),
) -> None:
    """Fetch, dedupe and store news; link it to symbols."""
    settings = _startup_config()
    limiter = candles.RateLimiter(0.5)  # polite: one request every 2 s
    chosen = {s.strip() for s in sources.split(",")}
    now = utc_now()
    since = now - timedelta(hours=hours)
    feeds: list[NewsSource] = []
    if "rss" in chosen:
        feeds.append(RssSource(settings.app.rss_feeds))
    if "nse" in chosen:
        feeds.append(NseAnnouncements(acquire=limiter.acquire))
    if "bse" in chosen:
        feeds.append(BseAnnouncements(acquire=limiter.acquire))
    with _session() as session:
        aliases = _alias_table(session)
        total = ingest_mod.IngestResult()
        ambiguous: list[tuple[int, tuple[Ambiguous, ...]]] = []
        for src in feeds:
            res = ingest_mod.ingest(session, src.fetch(since), now, aliases)
            typer.echo(
                f"{src.name}: stored {res.stored}, dup {res.dup_url + res.dup_hash}, "
                f"near-dup {res.near_dup}, links {res.linked}"
            )
            total.stored += res.stored
            ambiguous.extend(res.ambiguous)
        llm = _llm(settings)
        if ambiguous and llm.health():
            analyst = NewsAnalyst(session, llm)
            for news_id, amb in ambiguous:
                analyst.resolve_ambiguous(news_id, amb)
    typer.echo(f"{total.stored} new articles; {len(ambiguous)} needed disambiguation.")


@news_app.command("score")
def news_score(
    limit: int = typer.Option(300, help="Max (article, symbol) pairs this run"),
    max_minutes: float = typer.Option(50.0, help="Stop after this many minutes"),
) -> None:
    """Score pending articles with the local LLM (heavy job: do not run alongside training)."""
    settings = _startup_config()
    llm = _llm(settings)
    if not llm.health():
        typer.secho("Ollama is not reachable at 127.0.0.1:11434.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    with _session() as session:
        names = universe_mod.members_as_of(session, settings.app.universe, utc_now().date())
        analyst = NewsAnalyst(session, llm)
        run = analyst.run(limit, frozenset(names), max_seconds=max_minutes * 60)
        audit.log(
            session,
            "system",
            "news_score_run",
            {
                "scored": run.scored,
                "discarded": run.discarded,
                "seconds": round(run.seconds, 1),
                "per_hour": round(run.per_hour, 1),
                "model": llm.model,
            },
            utc_now(),
        )
        session.commit()
    typer.echo(
        f"scored {run.scored}, discarded {run.discarded}, valid rate {run.valid_rate:.1%}, "
        f"{run.per_hour:.0f}/hour"
    )
    if run.llm_down:
        typer.secho("Stopped early: Ollama became unavailable.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)


@news_app.command("stats")
def news_stats() -> None:
    """Counts, schema-valid rate and the last measured scoring throughput."""
    with _session() as session:
        n_raw = session.scalar(select(func.count()).select_from(NewsRaw)) or 0
        scores = list(session.scalars(select(NewsScore)))
        bad = sum(1 for s in scores if "error" in s.raw_json)
        runs = list(
            session.scalars(
                select(AuditLog)
                .where(AuditLog.action == "news_score_run")
                .order_by(AuditLog.id.desc())
                .limit(1)
            )
        )
    typer.echo(f"articles {n_raw}, scores {len(scores)}, discarded {bad}")
    if scores:
        typer.echo(f"schema-valid rate {(len(scores) - bad) / len(scores):.1%} (target >= 95%)")
    if runs:
        typer.echo(
            f"last run throughput {runs[0].payload_json.get('per_hour')} articles/hour "
            "(target >= 300)"
        )


@news_app.command("results-calendar")
def news_results_calendar(days: int = typer.Option(14)) -> None:
    """Pull upcoming results/board-meeting dates from NSE into ``result_dates``."""
    _startup_config()
    src = NseAnnouncements(acquire=candles.RateLimiter(0.5).acquire)
    today = utc_now().astimezone(IST).date()
    with _session() as session:
        added = 0
        for sym, when, purpose in src.fetch_result_calendar(today, today + timedelta(days=days)):
            if (
                session.scalar(
                    select(ResultDate.id).where(
                        ResultDate.symbol == sym, ResultDate.result_date == when
                    )
                )
                is None
            ):
                session.add(ResultDate(symbol=sym, result_date=when, purpose=purpose, source="nse"))
                added += 1
        session.commit()
    typer.echo(f"{added} result dates added.")


@news_app.command("backfill")
def news_backfill(
    start: str = typer.Option(..., "--from", help="YYYY-MM-DD"),
    end: str = typer.Option(..., "--to", help="YYYY-MM-DD"),
    window_days: int = typer.Option(7),
) -> None:
    """GDELT historical backfill (titles only), one query per company name."""
    from datetime import date as _date

    _startup_config()
    s, e = _date.fromisoformat(start), _date.fromisoformat(end)
    with _session() as session:
        aliases = _alias_table(session)
        queries = {
            f'"{a}"': syms[0]
            for a, syms in aliases.aliases.items()
            if len(syms) == 1 and len(a) > 4
        }
        gd = GdeltSource(
            queries, acquire=candles.RateLimiter(0.2).acquire
        )  # GDELT asks for 1 req / 5 s
        cursor, total = s, 0
        while cursor <= e:
            stop = min(cursor + timedelta(days=window_days), e + timedelta(days=1))
            lo = datetime.combine(cursor, datetime.min.time(), tzinfo=UTC)
            hi = datetime.combine(stop, datetime.min.time(), tzinfo=UTC)
            total += ingest_mod.ingest(session, gd.fetch_range(lo, hi), utc_now(), aliases).stored
            cursor = stop
    typer.echo(f"{total} historical articles stored.")


# ---------------- models (M6) ----------------

models_app = typer.Typer(help="Model registry: list, shadow, approve, rollback (M6).")
app.add_typer(models_app, name="models")


@models_app.command("list")
def models_list() -> None:
    """All models with status and gate result."""
    with _session() as session:
        rows = list(session.scalars(select(Model).order_by(Model.strategy, Model.id)))
    for m in rows:
        gate = m.metrics_json.get("gate", {})
        typer.echo(
            f"#{m.id} {m.strategy} {m.version} {m.status:9s} "
            f"gate_ready={gate.get('ready_for_review')} "
            f"variant={m.metrics_json.get('chosen_variant')} trained_to={m.train_end}"
        )
    if not rows:
        typer.echo("No models yet. Run `tradepartner train`.")


@app.command()
def deploy(model_id: int) -> None:
    """Deploy a model to SHADOW: signals are logged for comparison, never shown or traded.

    It refuses a model that has not passed the promotion gate. It never approves a model, never
    enables AUTO and never places an order; those stay your explicit decisions (`models approve`).
    """
    with _session() as session:
        try:
            model = registry.get(session, model_id)
            gate = model.metrics_json.get("gate", {})
            if not gate.get("ready_for_review"):
                failed = ", ".join(gate.get("failed", ["no gate report"]))
                raise registry.ModelError(f"not deployed: the promotion gate failed ({failed})")
            registry.start_shadow(session, model_id, utc_now())
        except registry.ModelError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
    typer.echo(
        f"Model {model_id} ({model.strategy}) deployed to SHADOW. "
        f"Its signals are logged, not shown "
        f"or traded. After {approval.MIN_SHADOW_DAYS} days and a passed comparison, run "
        f"`tradepartner models approve {model_id}`. Nothing was approved or ordered."
    )


@models_app.command("shadow")
def models_shadow(model_id: int) -> None:
    """Start shadow mode for a candidate (signals logged, never shown)."""
    with _session() as session:
        try:
            registry.start_shadow(session, model_id, utc_now())
        except registry.ModelError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
    typer.echo(f"Model {model_id} is now in shadow mode (min {approval.MIN_SHADOW_DAYS} days).")


@models_app.command("approve")
def models_approve(model_id: int) -> None:
    """Approve a model (only you can): gate, 2 weeks of shadow and a passed comparison."""
    with _session() as session:
        model = registry.get(session, model_id)
        for b in approval.approval_blockers(model, utc_now()):
            typer.secho(f"blocker: {b}", fg=typer.colors.YELLOW)
        phrase = typer.prompt(f"Type APPROVE {model.strategy} to confirm")
        if phrase != f"APPROVE {model.strategy}":
            typer.secho("Not confirmed.", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        try:
            approval.approve(session, model_id, "user", utc_now())
        except registry.ModelError as exc:
            typer.secho(f"Refused: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
    typer.echo(f"Model {model_id} approved. The previous model is kept for `models rollback`.")


@models_app.command("rollback")
def models_rollback(strategy: str) -> None:
    """Reinstate the previously approved model for a strategy."""
    with _session() as session:
        try:
            prev = approval.rollback(session, strategy, "user", utc_now())
        except registry.ModelError as exc:
            typer.secho(f"Refused: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc
    typer.echo(f"Rolled back to model {prev.id} ({prev.version}).")


# ---------------- live acceptance (M8): 1-share orders, by Pranav only ----------------

live_app = typer.Typer(help="M8 live acceptance with 1-share orders. Places REAL orders.")
app.add_typer(live_app, name="live-test")
_CONSENT = "--i-understand-this-places-real-orders"


def _live_runtime() -> tuple[Any, Any, Any]:
    settings = _startup()
    ctx = _job_context(settings)
    kite = kite_or_none()
    if kite is None:
        typer.secho("Not logged in: run `tradepartner login`.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    session = ctx.session_factory()
    return ctx.runtime(session), kite, session


@live_app.command("entry")
def live_entry(
    symbol: str,
    consent: bool = typer.Option(
        False, _CONSENT, help="Required: this places a REAL 1-share order"
    ),
) -> None:
    """Step 1: a 1-share limit entry for a liquid stock through the full risk engine."""
    if not consent:
        typer.secho(f"Refusing: pass {_CONSENT}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    rt, kite, session = _live_runtime()
    now = utc_now()
    try:
        ctl = ModeController(session)
        if ctl.current(GLOBAL).mode == "OFF":
            ctl.set_mode(GLOBAL, "MANUAL", "user", "live test", now)
        if ctl.current(strategy_scope(live_test.STRATEGY)).mode == "OFF":
            ctl.set_mode(strategy_scope(live_test.STRATEGY), "MANUAL", "user", "live test", now)
        res = live_test.prepare_entry(rt, kite, symbol, now)
        typer.echo(
            f"Risk decision: {res.status}; reasons: {res.decision.reasons if res.decision else []}"
        )
        if res.status != "shown" or res.proposal_id is None:
            raise typer.Exit(code=1)
        if typer.prompt("Type PLACE 1 SHARE to send the real order") != "PLACE 1 SHARE":
            typer.echo("Not placed.")
            raise typer.Exit(code=1)
        placed = live_test.place_entry(rt, res.proposal_id, utc_now())
        if placed.order is None:
            reasons = placed.decision.reasons if placed.decision else []
            typer.secho(f"Not placed ({placed.status}): {reasons}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        typer.echo(f"Placed: {placed.order}")
    except live_test.LiveTestError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        session.close()


@live_app.command("protect")
def live_protect(
    symbol: str,
    consent: bool = typer.Option(False, _CONSENT, help="Required: creates a REAL GTT"),
) -> None:
    """Step 2: after the entry fills (sync first), create its GTT OCO and verify it."""
    if not consent:
        typer.secho(f"Refusing: pass {_CONSENT}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    rt, kite, session = _live_runtime()
    try:
        if poll(kite, [symbol], rt.quotes, utc_now()) == 0:
            raise live_test.LiveTestError(f"no quote for {symbol}")
        typer.echo(f"GTT status: {live_test.protect(rt, symbol, utc_now())}")
    except live_test.LiveTestError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    finally:
        session.close()


@live_app.command("cancel")
def live_cancel(
    order_id: str,
    consent: bool = typer.Option(False, _CONSENT, help="Required: cancels a REAL order"),
) -> None:
    """Step 3: cancel an open test order. Then try `tradepartner kill` and `tradepartner sync`."""
    if not consent:
        typer.secho(f"Refusing: pass {_CONSENT}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    rt, _, session = _live_runtime()
    try:
        rt.broker.cancel_order(order_id)
        typer.echo(f"Cancel requested for {order_id}.")
    finally:
        session.close()
