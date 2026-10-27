"""Kill switch (spec 7.1 rule 4), kill file, notifications and limit-state helpers."""

from datetime import timedelta

import httpx
import keyring
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from tests.fixtures.risk import LTP
from tests.fixtures.world import STRATEGY, World
from tradepartner import notify
from tradepartner.db.models import AuditLog, Fill, Order, RiskEvent
from tradepartner.modes.state import GLOBAL
from tradepartner.risk import kill, limits
from tradepartner.risk.types import Context


@pytest.fixture
def w(db: Session) -> World:
    return World(db)


def _setup_book(w: World) -> None:
    """A CNC holding, a MIS position, and a resting (unfilled) entry order."""
    w.go_auto()
    assert w.submit_entry("INFY", "cnc", product="CNC").status == "placed"
    w.advance(seconds=2)
    w.instruments["TCS"] = w.instruments["INFY"].__class__("TCS", "IT", 20_000_000_000)
    w.set_price("TCS", LTP)
    assert w.submit_entry("TCS", "mis", product="MIS").status == "placed"
    w.advance(seconds=2)
    w.instruments["WIPRO"] = w.instruments["INFY"].__class__("WIPRO", "IT", 20_000_000_000)
    w.set_price("WIPRO", LTP)
    resting = w.submit_entry("WIPRO", "rest", limit_price_paise=LTP - 2000)
    assert (
        resting.status == "placed" and resting.order is not None and resting.order.status == "OPEN"
    )


def test_kill_switch_full_procedure(w: World) -> None:
    _setup_book(w)
    rep = w.kill.trigger(w.now, "manual test")
    assert w.controller.current(GLOBAL).mode == "OFF"
    assert len(rep.cancelled) == 1 and rep.squared_off == ["TCS"] and rep.skipped == []
    assert w.broker.open_orders() == []
    sells = list(w.db.scalars(select(Fill).where(Fill.side == "SELL")))
    assert [f.symbol for f in sells] == ["TCS"]  # CNC holding (and its GTT) untouched
    held = {p.symbol for p in w.plain_context().positions}
    assert "INFY" in held and "TCS" not in held
    assert kill.kill_active()
    assert (
        w.db.scalar(
            select(func.count()).select_from(RiskEvent).where(RiskEvent.rule == "kill_switch")
        )
        == 1
    )
    assert (
        w.db.scalar(
            select(func.count()).select_from(AuditLog).where(AuditLog.action == "kill_switch")
        )
        == 1
    )
    assert w.notes and "KILL SWITCH" in w.notes[0][0]
    # nothing new can be placed while the kill file exists, even if someone raises the mode
    w.controller.set_mode(GLOBAL, "MANUAL", "user", "x", w.now)
    w.advance(seconds=5)
    r = w.submit_entry("INFY", "after-kill")
    assert r.status == "rejected" and any("kill_switch" in x for x in r.decision.reasons)  # type: ignore[union-attr]
    kill.clear()
    assert not kill.kill_active()


def test_square_off_limit_is_ltp_minus_market_protection(w: World) -> None:
    _setup_book(w)
    w.kill.trigger(w.now, "x")
    sell = w.db.scalars(select(Order).where(Order.side == "SELL")).one()
    assert sell.order_type == "LIMIT" and sell.price_paise == LTP * 99 // 100  # protection 1%
    assert sell.tag and sell.tag.startswith("tpk-") and len(sell.tag) <= 20


def test_unfilled_square_off_escalates_to_protected_market_after_60s(w: World) -> None:
    _setup_book(w)
    w.broker.quotes = lambda s: None  # the exchange stops filling: orders stay open
    w.kill.trigger(w.now, "x")
    assert w.kill.tick(w.now + timedelta(seconds=59)) == []
    assert w.kill.tick(w.now + timedelta(seconds=61)) == ["TCS"]
    assert w.kill.tick(w.now + timedelta(seconds=120)) == []  # only once
    orders = list(w.db.scalars(select(Order).where(Order.side == "SELL").order_by(Order.id)))
    assert [o.order_type for o in orders] == ["LIMIT", "MARKET"] and orders[0].status == "CANCELLED"


def test_filled_square_off_needs_no_escalation(w: World) -> None:
    _setup_book(w)
    w.kill.trigger(w.now, "x")
    assert w.kill.tick(w.now + timedelta(seconds=90)) == []


def test_position_without_quote_is_reported_not_traded(w: World) -> None:
    _setup_book(w)
    del w.quotes["TCS"]
    rep = w.kill.trigger(w.now, "x")
    assert rep.skipped == ["TCS"] and rep.squared_off == []


def test_emergency_rejection_is_reported(w: World) -> None:
    _setup_book(w)
    calls = {"n": 0}
    real = w.kill.context_factory

    def flaky() -> Context:
        calls["n"] += 1
        ctx = real()
        # first call (positions scan) is real; the order-building call sees no positions
        return ctx if calls["n"] == 1 else Context(**{**ctx.__dict__, "positions": ()})

    w.kill.context_factory = flaky
    rep = w.kill.trigger(w.now, "x")
    assert rep.skipped == ["TCS"] and rep.squared_off == []


def test_non_bot_or_cnc_positions_are_left_alone(w: World) -> None:
    w.go_auto()
    assert w.submit_entry("INFY", "cnc", product="CNC").status == "placed"
    rep = w.kill.trigger(w.now, "x")
    assert (
        rep == type(rep)()
        and w.db.scalar(select(func.count()).select_from(Fill).where(Fill.side == "SELL")) == 0
    )


def test_kill_file_helpers(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "sub" / "KILL"
    assert not kill.kill_active(p)
    assert kill.arm(p) == p and kill.kill_active(p)
    kill.clear(p)
    kill.clear(p)  # clearing twice is fine
    assert not kill.kill_active(p)


# ---------- limits helpers ----------


def test_consecutive_loss_streak_resets_on_a_win() -> None:
    from datetime import date

    def trade(i: int, pnl: int) -> limits._Closed:
        from tests.fixtures.market import dt

        return limits._Closed(i, date(2026, 3, 1), dt(2026, 3, 1, 10, i), pnl, "s", "X", False)

    assert limits.consecutive_losses([trade(1, -5), trade(2, -5), trade(3, -5)]) == {"s": 3}
    assert limits.consecutive_losses([trade(1, -5), trade(2, 9), trade(3, -1)]) == {"s": 1}
    no_strategy = limits._Closed(9, date(2026, 3, 1), trade(1, 1).close_time, -1, None, "X", False)
    assert limits.consecutive_losses([no_strategy]) == {}


# ---------- notifications ----------


def test_macos_notify_escapes_and_survives_failure() -> None:
    seen: list[list[str]] = []
    notify.macos_notify('Ti"tle', 'say "hi"', runner=lambda cmd, **k: seen.append(cmd))
    assert 'say \\"hi\\"' in seen[0][2] and 'Ti\\"tle' in seen[0][2]

    def boom(cmd: list[str], **k: object) -> None:
        raise OSError("no osascript")

    notify.macos_notify("t", "m", runner=boom)  # must not raise


def test_telegram_is_optional_and_uses_keychain() -> None:
    assert notify.telegram_notify("t", "m") is False
    keyring.set_password("tradepartner", "telegram_token", "T0K")
    keyring.set_password("tradepartner", "telegram_chat_id", "42")
    posted: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        posted.append(req)
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert notify.telegram_notify("Title", "Body", client) is True
    assert posted[0].url.path.endswith("/sendMessage") and b"Title" in posted[0].content

    def fail(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    assert (
        notify.telegram_notify("t", "m", httpx.Client(transport=httpx.MockTransport(fail))) is False
    )


def test_notify_fans_out(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(notify, "macos_notify", lambda t, m: calls.append("mac"))
    monkeypatch.setattr(notify, "telegram_notify", lambda t, m: calls.append("tg") or False)
    notify.notify("a", "b")
    assert calls == ["mac", "tg"] and STRATEGY


def test_only_bot_buy_entries_are_cancelled(w: World) -> None:
    from tests.unit.test_paper_router import order
    from tradepartner.risk.approved import ApprovedOrder  # noqa: F401

    w.go_auto()
    foreign = w.broker.place_order(order(limit_price_paise=LTP - 3000, tag=None))  # untagged
    sell = w.broker.place_order(order(side="SELL", limit_price_paise=LTP + 3000, tag="tp-77"))
    assert foreign.status == "OPEN" and sell.status == "OPEN"
    rep = w.kill.trigger(w.now, "x")
    assert rep.cancelled == []  # neither is a bot BUY entry
    assert len(w.broker.open_orders()) == 2


def test_escalation_skipped_when_market_exit_is_rejected(w: World) -> None:
    _setup_book(w)
    w.broker.quotes = lambda s: None
    w.kill.trigger(w.now, "x")
    real = w.kill.context_factory
    w.kill.context_factory = lambda: Context(**{**real().__dict__, "positions": ()})
    assert w.kill.tick(w.now + timedelta(seconds=61)) == []
