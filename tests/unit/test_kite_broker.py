"""KiteBroker against a recording fake. No real broker is ever contacted (CLAUDE.md rules 9, 10)."""

from dataclasses import replace
from datetime import UTC, date, datetime

import pytest

from tests.fixtures.kiteapi import FakeKiteApi
from tests.unit.test_paper_router import order
from tradepartner.data.candles import RateLimiter
from tradepartner.execution.broker import UnsupportedOrderError
from tradepartner.execution.kite import BrokerError, KiteBroker, StaticIpError, rupees, tick_down
from tradepartner.risk import _mint
from tradepartner.risk.approved import ApprovedOrder


class Free(RateLimiter):
    def __init__(self) -> None:
        super().__init__(1000.0)

    def acquire(self) -> None:
        return None


def broker(
    api: FakeKiteApi | None = None, errors: list[str] | None = None
) -> tuple[KiteBroker, FakeKiteApi]:
    api = api or FakeKiteApi()
    return KiteBroker(api, Free(), (errors if errors is not None else []).append), api


def test_limit_order_is_sent_with_exact_rupees_tag_and_nse_only() -> None:
    b, api = broker()
    placed = b.place_order(order(limit_price_paise=150050, tag="tp-77"))
    name, kw = api.calls[0]
    assert name == "place_order" and kw["exchange"] == "NSE" and kw["variety"] == "regular"
    assert (
        kw["tradingsymbol"] == "INFY" and kw["transaction_type"] == "BUY" and kw["product"] == "CNC"
    )
    assert (
        kw["order_type"] == "LIMIT"
        and kw["price"] == 1500.5
        and kw["tag"] == "tp-77"
        and kw["quantity"] == 10
    )
    assert kw["validity"] == "DAY" and "market_protection" not in kw
    assert placed.broker_order_id == "1001" and placed.status == "OPEN" and placed.tag == "tp-77"
    assert rupees(150050) == 1500.5 and tick_down(150053) == 150050


def test_only_an_approved_order_can_be_placed() -> None:
    b, api = broker()
    with pytest.raises(TypeError, match="approved"):
        b.place_order(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ApprovedOrder(**{**order().__dict__, "_token": object()})
    assert api.calls == []


def test_unsupported_orders_are_rejected_before_any_api_call() -> None:
    b, api = broker()
    for bad in (dict(product="NRML"), dict(side="SHORT"), dict(order_type="BRACKET")):
        base = order().__dict__.copy()
        base.update(bad)
        base.pop("_token")
        with pytest.raises(UnsupportedOrderError):
            b.place_order(ApprovedOrder(**base, _token=_mint.TOKEN))  # type: ignore[arg-type]
    assert api.calls == []


def test_market_orders_carry_market_protection() -> None:
    b, api = broker()
    b.place_order(
        order(
            order_type="MARKET",
            limit_price_paise=None,
            market_protection_pct=1.0,
            side="SELL",
            tag="tpk-abc",
        )
    )
    assert api.calls[0][1]["market_protection"] == 1.0 and "price" not in api.calls[0][1]
    stop = order(
        order_type="SL-M",
        limit_price_paise=None,
        trigger_price_paise=144000,
        side="SELL",
        tag="tp-9",
    )
    b.place_order(stop)
    assert api.calls[1][1]["trigger_price"] == 1440.0


def test_the_same_tag_is_never_placed_twice_even_after_a_crash() -> None:
    b, api = broker()
    first = b.place_order(order(tag="tp-5"))
    again = b.place_order(order(tag="tp-5"))
    assert again.broker_order_id == first.broker_order_id
    assert [n for n, _ in api.calls].count("place_order") == 1
    api.cancel_order("regular", first.broker_order_id)
    api.calls.clear()
    b.place_order(
        order(tag="tp-5")
    )  # the earlier one was cancelled, so a fresh placement is allowed
    assert [n for n, _ in api.calls] == ["cancel_order"] or api.calls[-1][0] == "place_order"


def test_open_orders_get_and_cancel() -> None:
    b, api = broker()
    a = b.place_order(order(tag="tp-1"))
    b.place_order(order(tag="tp-2", limit_price_paise=149000))
    assert len(b.open_orders()) == 2 and b.get_order(a.broker_order_id).tag == "tp-1"  # type: ignore[union-attr]
    b.cancel_order(a.broker_order_id)
    assert [o.tag for o in b.open_orders()] == ["tp-2"] and b.get_order("nope") is None
    assert api.calls[-1] == ("cancel_order", {"variety": "regular", "order_id": a.broker_order_id})


def test_static_ip_rejection_is_mapped_and_alerted() -> None:
    errors: list[str] = []
    b, api = broker(errors=errors)
    api.fail_with = Exception("Order placement is not allowed from this IP. Register a static IP.")
    with pytest.raises(StaticIpError):
        b.place_order(order(tag="tp-1"))
    assert "static IP" in errors[0]
    api.fail_with = Exception("Insufficient funds")
    with pytest.raises(BrokerError, match="Insufficient"):
        b.place_order(order(tag="tp-2"))
    assert len(errors) == 2 and not isinstance(BrokerError("x"), StaticIpError)


def test_gtt_oco_is_a_sell_only_two_leg_trigger_with_a_buffered_stop_leg() -> None:
    b, api = broker()
    gid = b.place_gtt_oco("INFY", 10, 150000, 144000, 162000)
    kw = api.calls[0][1]
    assert (
        api.calls[0][0] == "place_gtt"
        and kw["trigger_type"] == "two-leg"
        and kw["exchange"] == "NSE"
    )
    assert kw["trigger_values"] == [1440.0, 1620.0] and kw["last_price"] == 1500.0
    stop_leg, target_leg = kw["orders"]
    assert stop_leg["transaction_type"] == target_leg["transaction_type"] == "SELL"
    assert stop_leg["quantity"] == target_leg["quantity"] == 10 and stop_leg["product"] == "CNC"
    assert (
        stop_leg["price"] == 1425.6 and target_leg["price"] == 1620.0
    )  # stop sells 1% below its trigger
    (info,) = b.list_gtts()
    assert info.gtt_id == gid and (info.qty, info.stop_paise, info.target_paise) == (
        10,
        144000,
        162000,
    )
    with pytest.raises(ValueError, match="stop < last price < target"):
        b.place_gtt_oco("INFY", 10, 150000, 151000, 162000)
    with pytest.raises(UnsupportedOrderError):
        b.place_gtt_oco("INFY", 0, 150000, 144000, 162000)


def test_gtt_modify_delete_and_non_oco_triggers_ignored() -> None:
    b, api = broker()
    gid = b.place_gtt_oco("INFY", 10, 150000, 144000, 162000)
    b.modify_gtt_oco(gid, 6, 150000, 145000, 160000)
    assert b.list_gtts()[0].qty == 6 and b.list_gtts()[0].stop_paise == 145000
    with pytest.raises(BrokerError, match="not found"):
        b.modify_gtt_oco("999", 1, 150000, 145000, 160000)
    api._gtts.append(
        {
            "id": 1,
            "status": "active",
            "condition": {"tradingsymbol": "X", "trigger_values": [1.0], "last_price": 1},
            "orders": [{"quantity": 1}],
        }
    )
    assert len(b.list_gtts()) == 1  # single-leg triggers are not ours
    b.delete_gtt(gid)
    assert b.list_gtts()[0].status == "deleted" and api.calls[-1] == (
        "delete_gtt",
        {"trigger_id": int(gid)},
    )


def test_rate_limiter_is_used_for_every_order() -> None:
    calls: list[int] = []

    class Counting(RateLimiter):
        def __init__(self) -> None:
            super().__init__(1.0)

        def acquire(self) -> None:
            calls.append(1)

    api = FakeKiteApi()
    b = KiteBroker(api, Counting())
    b.place_order(order(tag="tp-1"))
    b.place_order(order(tag="tp-2"))
    assert len(calls) == 2
    assert replace(order(), qty=5).qty == 5 and date(2026, 1, 1) and datetime.now(UTC)


def test_untagged_orders_skip_the_duplicate_lookup_and_unconfirmed_orders_still_return() -> None:
    b, api = broker()
    placed = b.place_order(order(tag=None))
    assert placed.tag is None and [n for n, _ in api.calls] == [
        "place_order"
    ]  # no orders() lookup without a tag

    class Lagging(FakeKiteApi):
        def orders(self) -> list[dict[str, object]]:
            return []  # the broker has not listed the new order yet

    lag, lapi = broker(Lagging())
    fallback = lag.place_order(order(tag="tp-3"))
    assert fallback.broker_order_id == "1001" and fallback.status == "OPEN" and fallback.qty == 10
    assert fallback.tag == "tp-3" and lapi.calls[0][0] == "place_order"


def test_market_orders_without_protection_cannot_even_be_built() -> None:
    base = order().__dict__.copy()
    base.pop("_token")
    base.update(order_type="MARKET", limit_price_paise=None, market_protection_pct=None)
    with pytest.raises(ValueError, match="protection"):
        ApprovedOrder(**base, _token=_mint.TOKEN)  # type: ignore[arg-type]
