"""M8: protective orders, api_approve, the 1-share size cap and the live-test commands (all faked)."""

import shutil
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from tests.fixtures.kiteapi import FakeKiteApi
from tests.fixtures.market import dt
from tests.fixtures.pipeline import make_env
from tests.fixtures.risk import LTP, cfg, ctx, entry, meta, quote
from tests.fixtures.world import STRATEGY, World
from tradepartner import appctx, cli, jobs, live_test
from tradepartner.db.models import AuditLog, Order, OrderProposal
from tradepartner.execution.kite import KiteBroker
from tradepartner.execution.paper import PaperBroker
from tradepartner.execution.protective import ProtectiveManager, intents
from tradepartner.modes.state import GLOBAL, strategy_scope
from tradepartner.risk import sizing
from tradepartner.settings import load_settings

runner = CliRunner()


@pytest.fixture
def w(db: Session) -> World:
    return World(db)


def bought(w: World, key: str = "k1") -> int:
    w.go_auto()
    r = w.submit_entry(key=key)
    assert r.status == "placed"
    return r.decision.sizing.qty  # type: ignore[union-attr]


def pm(w: World, last: int | None = LTP) -> tuple[ProtectiveManager, list[str]]:
    notes: list[str] = []
    return ProtectiveManager(
        w.db, w.broker, lambda s: last, lambda t, m: notes.append(m), lambda: w.now
    ), notes


# ---------------- protective orders ----------------


def test_intents_come_from_the_proposal_that_opened_the_position(w: World) -> None:
    qty = bought(w)
    w.context_for(STRATEGY)  # rebuild lots
    (it,) = intents(w.db)
    p = w.db.scalars(select(OrderProposal)).one()
    assert (it.symbol, it.qty, it.stop_paise, it.target_paise) == (
        "INFY",
        qty,
        p.stop_loss_paise,
        p.target_paise,
    )
    assert it.strategy == STRATEGY and it.origin == "paper"


def test_missing_gtt_is_created_in_auto_and_only_alerted_otherwise(w: World) -> None:
    qty = bought(w)
    w.context_for(STRATEGY)
    manager, notes = pm(w)
    report = manager.check(allow_place=False)
    assert report == {"INFY": "MISSING (alert only)"} and notes and w.broker.list_gtts() == []
    report = manager.check(allow_place={STRATEGY: True})
    assert report == {"INFY": "CREATED"} and w.broker.list_gtts()[0].qty == qty
    assert manager.check(allow_place=True) == {"INFY": "OK"}  # verified, nothing re-created
    assert len(w.broker.list_gtts()) == 1
    payload = w.db.scalars(
        select(AuditLog).where(AuditLog.action == "protective_check").order_by(AuditLog.id.desc())
    ).first()
    assert payload is not None and payload.payload_json == {"symbols": {"INFY": "OK"}}


def test_gtt_is_resized_to_the_position_and_mismatch_alerts_without_permission(w: World) -> None:
    bought(w)
    w.context_for(STRATEGY)
    manager, notes = pm(w)
    manager.check(allow_place=True)
    gid = w.broker.list_gtts()[0].gtt_id
    w.broker.modify_gtt_oco(
        gid, 1, LTP, w.broker.list_gtts()[0].stop_paise, w.broker.list_gtts()[0].target_paise
    )  # wrong qty
    assert manager.check(allow_place=False)["INFY"] == "MISMATCH (alert only)" and notes
    assert manager.check(allow_place=True)["INFY"] == "RESIZED"
    assert manager.check(allow_place=True)["INFY"] == "OK"


def test_breached_target_reached_and_no_quote_cannot_get_a_gtt(w: World) -> None:
    bought(w)
    w.context_for(STRATEGY)
    p = w.db.scalars(select(OrderProposal)).one()
    for last, expect in (
        (p.stop_loss_paise - 1, "BREACHED"),
        (p.target_paise + 1, "TARGET_REACHED"),
        (None, "NO_QUOTE"),
    ):
        manager, notes = pm(w, last)
        assert manager.check(allow_place=True)["INFY"] == expect and "act now" in notes[-1]
    assert w.broker.list_gtts() == []


def test_orphan_gtt_is_reported_never_deleted(w: World) -> None:
    w.broker.place_gtt_oco("TCS", 5, LTP, LTP - 5000, LTP + 8000)
    manager, notes = pm(w)
    assert (
        manager.check(allow_place=True) == {"TCS": "ORPHAN"} and "no matching position" in notes[0]
    )
    assert w.broker.list_gtts()[0].status == "active"


def test_place_for_is_a_user_action(w: World) -> None:
    bought(w)
    w.context_for(STRATEGY)
    manager, _ = pm(w)
    assert manager.place_for("NOPE") == "NOT_MANAGED"
    assert manager.place_for("INFY") == "CREATED" and manager.place_for("INFY") == "RESIZED"
    assert (
        w.db.scalar(
            select(func.count()).select_from(AuditLog).where(AuditLog.action == "protective_place")
        )
        == 2
    )


def test_manual_positions_without_a_proposal_are_not_managed(db: Session) -> None:
    from tests.unit.test_ledger_db import BOOK, tb, write
    from tradepartner.ledger import importers, rebuild

    importers.import_tradebook(
        db, write(Path(__import__("tempfile").mkdtemp()), "t.csv", tb(BOOK[:1]))
    )
    rebuild.rebuild(db, load_settings().costs, load_settings().tax)
    assert intents(db) == []


# ---------------- api_approve ----------------


def test_approve_and_place_rechecks_and_places_once(w: World) -> None:
    w.go_manual()
    w.router.manual_execution = "api_approve"
    r = w.submit_entry()
    assert r.status == "shown" and r.proposal_id is not None
    row = w.db.get(OrderProposal, r.proposal_id)
    assert row is not None
    row.created_at = w.now  # the real clock and the test clock differ; keep the signal fresh
    w.db.commit()
    w.advance(minutes=2)
    placed = w.router.approve_and_place(r.proposal_id, w.now)
    assert (
        placed.status == "placed"
        and placed.order is not None
        and placed.order.tag == f"tp-{r.proposal_id}"
    )
    assert w.db.scalar(select(func.count()).select_from(Order)) == 1
    with pytest.raises(ValueError, match="already been placed"):
        w.router.approve_and_place(r.proposal_id, w.now)


def test_approve_and_place_refuses_in_the_wrong_mode_or_state(w: World) -> None:
    w.go_manual()
    r = w.submit_entry()
    with pytest.raises(PermissionError, match="kite_app"):
        w.router.approve_and_place(r.proposal_id, w.now)  # type: ignore[arg-type]
    w.router.manual_execution = "api_approve"
    with pytest.raises(ValueError, match="risk-approved"):
        w.router.approve_and_place(99999, w.now)
    rejected = w.submit_entry(key="bad", stop_loss_paise=None)
    with pytest.raises(ValueError, match="risk-approved"):
        w.router.approve_and_place(rejected.proposal_id, w.now)  # type: ignore[arg-type]


def test_approval_is_refused_when_the_signal_has_gone_stale_or_the_world_changed(w: World) -> None:
    w.go_manual()
    w.router.manual_execution = "api_approve"
    r = w.submit_entry()
    row = w.db.get(OrderProposal, r.proposal_id)
    row.created_at = w.now  # type: ignore[union-attr]
    w.db.commit()
    w.advance(hours=3)  # max signal age is 120 minutes
    res = w.router.approve_and_place(r.proposal_id, w.now)  # type: ignore[arg-type]
    assert res.status == "rejected" and any("signal_age" in x for x in res.decision.reasons)  # type: ignore[union-attr]
    assert w.db.get(OrderProposal, r.proposal_id).risk_decision == "rejected"  # type: ignore[union-attr]
    assert w.db.scalar(select(func.count()).select_from(Order)) == 0


# ---------------- the 1-share cap ----------------


def test_max_qty_can_only_lower_the_size() -> None:
    c = cfg()
    base = sizing.compute_sizing(entry(), ctx(), c)
    assert base.qty > 1
    capped = sizing.compute_sizing(replace(entry(), max_qty=1), ctx(), c)
    assert capped.qty == 1 and capped.risk_amount_paise == 6000 and capped.notional_paise == LTP
    assert (
        sizing.compute_sizing(replace(entry(), max_qty=10**6), ctx(), c).qty == base.qty
    )  # never raises
    assert sizing.compute_sizing(replace(entry(), max_qty=0), ctx(), c).qty == 0


# ---------------- appctx ----------------


def test_broker_factory_defaults_to_paper_and_never_silently_falls_back(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradepartner.data import kite_client
    from tradepartner.data.quotes import QuoteBook

    s = load_settings()
    assert isinstance(appctx.broker_factory(s)(db, QuoteBook()), PaperBroker)
    live = s.model_copy(update={"app": s.app.model_copy(update={"broker": "kite"})})
    with pytest.raises(kite_client.NotLoggedInError):
        appctx.broker_factory(live)(
            db, QuoteBook()
        )  # no token: it raises, it does not trade on paper
    monkeypatch.setattr(kite_client, "get_kite", lambda: FakeKiteApi())
    assert isinstance(appctx.broker_factory(live)(db, QuoteBook()), KiteBroker)
    assert appctx.kite_or_none() is not None
    monkeypatch.setattr(
        kite_client, "get_kite", lambda: (_ for _ in ()).throw(kite_client.NotLoggedInError("x"))
    )
    assert appctx.kite_or_none() is None


def test_internet_check(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200))
    assert appctx.internet_ok()

    def boom(*a: object, **k: object) -> None:
        raise httpx.ConnectError("x")

    monkeypatch.setattr(httpx, "get", boom)
    assert not appctx.internet_ok()


# ---------------- scheduled protective jobs ----------------


def test_protective_jobs_skip_on_paper_and_run_on_kite(db: Session, tmp_path: Path) -> None:
    env = make_env(db, tmp_path)
    assert "paper broker" in jobs.job_protective_check(env.ctx)
    assert jobs.job_poll_fills(env.ctx) == "skipped"
    live = env.ctx.settings.model_copy(
        update={"app": env.ctx.settings.app.model_copy(update={"broker": "kite"})}
    )
    env.ctx.settings = live
    api = FakeKiteApi()
    env.ctx.broker_factory = lambda s, q: KiteBroker(api)
    assert jobs.job_protective_check(env.ctx) == "0 positions OK"
    assert jobs.job_poll_fills(env.ctx) == "fills+0"
    env.ctx.kite = lambda: None
    assert jobs.job_poll_fills(env.ctx) == "skipped"
    assert {"poll_fills", "protective_check"} <= set(jobs.BY_NAME)


# ---------------- live-test commands (everything faked) ----------------


@pytest.fixture
def live_world(db: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    from tests.fixtures.pipeline import NOW0

    c = tmp_path / "cfg"
    shutil.copytree(Path(__file__).parents[2] / "config", c)
    risk = yaml.safe_load((c / "risk.yaml").read_text())
    risk["capital_base_inr"] = 1_000_000
    (c / "risk.yaml").write_text(yaml.safe_dump(risk))
    app = yaml.safe_load((c / "app.yaml").read_text())
    app.update(broker="kite", manual_execution="api_approve")
    (c / "app.yaml").write_text(yaml.safe_dump(app))
    monkeypatch.setenv("TRADEPARTNER_CONFIG_DIR", str(c))
    env = make_env(db, tmp_path)  # seeds candles in this test's home
    tz = dt(2026, 1, 1).tzinfo
    now = datetime(NOW0.year, NOW0.month, NOW0.day, 10, 0, tzinfo=tz).astimezone(UTC)

    class Both(FakeKiteApi):
        profile = env.kite.profile
        instruments = env.kite.instruments

        def quote(self, instruments: list[str]) -> dict[str, dict[str, object]]:
            out = env.kite.quote(instruments)
            for v in out.values():
                v["timestamp"] = now.astimezone(tz).replace(
                    tzinfo=None
                )  # fresh, naive IST like Kite
            return out

    both = Both()
    from tradepartner.db.models import UniverseMember

    with cli._session() as home:  # the CLI uses the home database, not the fixture's session
        for sym in env.view.symbols:
            home.add(
                UniverseMember(universe_name="nifty100", symbol=sym, valid_from=date(2000, 1, 1))
            )
        home.commit()
    clock = {"t": now}
    monkeypatch.setattr(cli, "utc_now", lambda: clock["t"])
    monkeypatch.setattr(appctx, "utc_now", lambda: clock["t"])
    from tradepartner.data import kite_client

    monkeypatch.setattr(kite_client, "get_kite", lambda: both)
    monkeypatch.setattr(cli, "kite_or_none", lambda: both)
    monkeypatch.setattr(cli, "_llm", lambda s: env.ollama.client())
    return env, both, clock


def test_live_test_refuses_without_explicit_consent(live_world) -> None:  # type: ignore[no-untyped-def]
    _, both, _ = live_world
    for args in (
        ["live-test", "entry", "SYM00"],
        ["live-test", "protect", "SYM00"],
        ["live-test", "cancel", "1"],
    ):
        r = runner.invoke(cli.app, args)
        assert r.exit_code == 2 and "i-understand-this-places-real-orders" in r.output
    assert both.calls == []  # not a single broker call


def test_live_test_places_exactly_one_share_after_typed_confirmation(live_world) -> None:  # type: ignore[no-untyped-def]
    env, both, clock = live_world
    consent = "--i-understand-this-places-real-orders"
    declined = runner.invoke(cli.app, ["live-test", "entry", "SYM00", consent], input="no\n")
    assert declined.exit_code == 1 and "Not placed" in declined.output
    assert not [c for c in both.calls if c[0] == "place_order"]
    clock["t"] += timedelta(seconds=2)  # a new second means a new idempotency key
    r = runner.invoke(cli.app, ["live-test", "entry", "SYM00", consent], input="PLACE 1 SHARE\n")
    assert r.exit_code == 0, r.output
    (call,) = [c for c in both.calls if c[0] == "place_order"]
    kw = call[1]
    assert (
        kw["quantity"] == 1
        and kw["exchange"] == "NSE"
        and kw["product"] == "CNC"
        and kw["order_type"] == "LIMIT"
    )
    assert kw["tag"].startswith("tp-")
    with cli._session() as home:
        rows = home.scalars(
            select(OrderProposal).where(OrderProposal.strategy == "live_test")
        ).all()
        row = rows[-1]  # the declined run left an unplaced proposal behind
        assert len(rows) == 2 and home.scalar(select(func.count()).select_from(Order)) == 0
    assert (
        env.ctx is not None
        and row.max_qty == 1
        and row.qty == 1
        and row.risk_decision == "approved"
        and row.mode_at_creation == "MANUAL"
    )
    order_id = both.orders()[0]["order_id"]
    c = runner.invoke(cli.app, ["live-test", "cancel", order_id, consent])
    assert (
        c.exit_code == 0
        and ("cancel_order", {"variety": "regular", "order_id": order_id}) in both.calls
    )


def test_live_test_needs_the_kite_broker_api_approve_and_capital(
    live_world, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    env, both, _ = live_world
    from tradepartner.runtime import build_runtime

    s = load_settings()
    rt = build_runtime(env.db, s)
    with pytest.raises(live_test.LiveTestError):
        live_test.require_live(
            build_runtime(
                env.db, s.model_copy(update={"app": s.app.model_copy(update={"broker": "paper"})})
            )
        )
    with pytest.raises(live_test.LiveTestError, match="api_approve"):
        live_test.require_live(
            build_runtime(
                env.db,
                s.model_copy(
                    update={"app": s.app.model_copy(update={"manual_execution": "kite_app"})}
                ),
            )
        )
    with pytest.raises(live_test.LiveTestError, match="capital"):
        live_test.require_live(
            build_runtime(
                env.db,
                s.model_copy(update={"risk": s.risk.model_copy(update={"capital_base_inr": 0})}),
            )
        )
    live_test.require_live(rt)
    assert live_test.MAX_QTY == 1 and not both.calls


def test_live_test_not_logged_in_and_protect_flow(
    live_world, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    _, both, _ = live_world
    consent = "--i-understand-this-places-real-orders"
    monkeypatch.setattr(cli, "kite_or_none", lambda: None)
    assert runner.invoke(cli.app, ["live-test", "entry", "SYM00", consent]).exit_code == 2
    monkeypatch.setattr(cli, "kite_or_none", lambda: both)
    r = runner.invoke(cli.app, ["live-test", "protect", "SYM00", consent])
    assert (
        r.exit_code == 0 and "NOT_MANAGED" in r.output
    )  # no filled bot position yet: nothing to protect
    assert not [c for c in both.calls if c[0] in ("place_gtt", "place_order")]
    assert (
        STRATEGY
        and GLOBAL
        and strategy_scope("x")
        and quote(1)
        and meta("X")
        and timedelta(0) == timedelta()
        and date(2026, 1, 1)
    )


# ---------------- coverage of the skip paths in the protective helpers ----------------


def test_protection_ignores_positions_it_does_not_manage(w: World) -> None:
    from tradepartner.db.models import Fill, Lot
    from tradepartner.execution.mis_protect import mis_intents

    w.go_auto()
    assert w.submit_entry(product="MIS").status == "placed"  # a MIS lot: not a CNC GTT candidate
    w.advance(minutes=1)
    w.context_for(STRATEGY)
    assert intents(w.db) == []
    cnc_fill = Fill(broker_trade_id="m1", origin="manual_external", account="a", symbol="TCS", exchange="NSE", side="BUY", product="CNC", qty=5,
                    price_paise=100000, filled_at=w.now, trade_date=w.now.date(), source="console_csv")  # fmt: skip
    w.db.add(cnc_fill)
    w.db.flush()
    w.db.add(
        Lot(
            symbol="TCS",
            open_fill_id=cnc_fill.id,
            open_date=w.now.date(),
            qty_open=5,
            qty_remaining=5,
            cost_paise_per_share=100000,
        )
    )
    manual_mis = Fill(broker_trade_id="m2", origin="manual_external", account="a", symbol="WIPRO", exchange="NSE", side="BUY", product="MIS", qty=5,
                      price_paise=100000, filled_at=w.now, trade_date=w.now.date(), source="console_csv")  # fmt: skip
    w.db.add(manual_mis)
    w.db.flush()
    w.db.add(
        Lot(
            symbol="WIPRO",
            open_fill_id=manual_mis.id,
            open_date=w.now.date(),
            qty_open=5,
            qty_remaining=5,
            cost_paise_per_share=100000,
        )
    )
    w.db.commit()
    assert [
        i.symbol for i in intents(w.db)
    ] == []  # manual fills with no proposal are not bot-managed
    assert [i.symbol for i in mis_intents(w.db)] == [
        "INFY"
    ]  # the manual MIS lot has no stop to enforce


def test_mis_stop_is_not_reported_placed_when_the_router_refuses(w: World) -> None:
    from types import SimpleNamespace

    from tradepartner.execution.mis_protect import place_mis_stops

    w.go_auto()
    assert w.submit_entry(product="MIS").status == "placed"
    w.advance(minutes=1)
    w.context_for(STRATEGY)
    w.quotes.clear()  # no quote: the protective exit cannot pass the freshness rule
    rt = SimpleNamespace(session=w.db, broker=w.broker, router=w.router)
    assert place_mis_stops(rt, w.now) == []  # type: ignore[arg-type]
    assert w.broker.open_orders() == []


def test_unlock_evidence_picks_up_the_first_bot_fill_before_any_trade_closes(w: World) -> None:
    from tradepartner.modes.evidence import collect_unlock_evidence

    w.go_auto()
    assert w.submit_entry().status == "placed"
    w.context_for(STRATEGY)
    ev = collect_unlock_evidence(w.db, STRATEGY)
    assert ev.closed_trades == 0 and ev.first_trade_date == w.now.date()


def test_a_manual_mis_order_without_a_proposal_has_no_bot_stop(w: World) -> None:
    from tradepartner.db.models import Fill, Lot, Order
    from tradepartner.execution.mis_protect import mis_intents

    o = Order(broker_order_id="X1", origin="manual_external", symbol="WIPRO", side="BUY", product="MIS", order_type="LIMIT", qty=5,
              placed_at=w.now, status="COMPLETE")  # fmt: skip
    w.db.add(o)
    w.db.flush()
    f = Fill(broker_trade_id="x1", broker_order_id="X1", order_id=o.id, origin="manual_external", account="a", symbol="WIPRO", exchange="NSE",
             side="BUY", product="MIS", qty=5, price_paise=100000, filled_at=w.now, trade_date=w.now.date(), source="kite_api")  # fmt: skip
    w.db.add(f)
    w.db.flush()
    w.db.add(
        Lot(
            symbol="WIPRO",
            open_fill_id=f.id,
            open_date=w.now.date(),
            qty_open=5,
            qty_remaining=5,
            cost_paise_per_share=100000,
        )
    )
    w.db.commit()
    assert mis_intents(w.db) == []
