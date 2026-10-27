"""Charge calculations vs contract-note fixtures (within +/- Rs 1) and invariants."""

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tradepartner.costs import zerodha
from tradepartner.costs.zerodha import FillIn
from tradepartner.settings import CostsConfig, load_settings

CFG = load_settings().costs
NOTES = sorted((Path(__file__).parents[1] / "fixtures" / "contract_notes").glob("*.json"))


def _fills(raw: list[dict[str, object]]) -> list[FillIn]:
    return [
        FillIn(
            id=int(f["id"]),  # type: ignore[call-overload]
            order_ref=str(f["order_ref"]),
            symbol=str(f["symbol"]),
            side=str(f["side"]),
            product=str(f["product"]),
            qty=int(f["qty"]),  # type: ignore[call-overload]
            price_paise=int(Decimal(str(f["price"])) * 100),
            trade_date=date.fromisoformat(str(f["trade_date"])),
        )
        for f in raw
    ]


def test_fixtures_present() -> None:
    assert len(NOTES) >= 4


@pytest.mark.parametrize("path", NOTES, ids=lambda p: p.stem)
def test_total_charges_match_contract_note_within_one_rupee(path: Path) -> None:
    note = json.loads(path.read_text())
    lines = zerodha.compute(_fills(note["fills"]), CFG)
    assert abs(zerodha.total_paise(lines) / 100 - note["total"]) <= 1.0


def _one(side: str, product: str, qty: int = 100, price: int = 150050) -> FillIn:
    return FillIn(1, "o", "INFY", side, product, qty, price, date(2026, 3, 2))


def test_delivery_buy_components() -> None:
    comps = zerodha.compute_order([_one("BUY", "CNC")], CFG)
    assert comps["brokerage"] == 0
    assert comps["stt"] == 15_000  # 0.1% of 15,005 -> Rs 15.005 -> Rs 15
    assert comps["stamp"] > 0 and comps["exchange_txn"] > 0 and comps["gst"] > 0


def test_delivery_sell_has_dp_once_per_scrip_day_and_no_stamp() -> None:
    a = FillIn(1, "o1", "INFY", "SELL", "CNC", 10, 100000, date(2026, 3, 2))
    b = FillIn(2, "o2", "INFY", "SELL", "CNC", 10, 100000, date(2026, 3, 2))
    c = FillIn(3, "o3", "INFY", "SELL", "CNC", 10, 100000, date(2026, 3, 3))
    lines = zerodha.compute([a, b, c], CFG)
    dp = [x for x in lines if x.component == "dp"]
    assert sorted(x.fill_id for x in dp) == [1, 3]  # first sell order of each scrip-day
    assert all(x.component != "stamp" for x in lines)


def test_intraday_sell_stt_and_brokerage_cap() -> None:
    sell = zerodha.compute_order([_one("SELL", "MIS", qty=100000, price=300000)], CFG)
    assert sell["brokerage"] == 2000 and sell["stamp"] == 0 and sell["stt"] > 0
    buy = zerodha.compute_order([_one("BUY", "MIS")], CFG)
    assert buy["stt"] == 0 and buy["stamp"] > 0


def test_stt_rounding_flag_is_configurable() -> None:
    raw = CFG.model_dump()
    raw["rounding"]["stt_round_to_rupee"] = False
    precise = CostsConfig.model_validate(raw)
    assert zerodha.compute_order([_one("BUY", "CNC")], precise)["stt"] == 15_005
    assert zerodha.compute_order([_one("BUY", "CNC")], CFG)["stt"] == 15_000


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="share side"):
        zerodha.compute_order([_one("BUY", "CNC"), _one("SELL", "CNC")], CFG)
    with pytest.raises(ValueError, match="unsupported"):
        zerodha.compute_order([_one("BUY", "NRML")], CFG)
    with pytest.raises(ValueError, match="positive"):
        zerodha.allocate(10, [0, 0])


def test_no_fills_no_charges() -> None:
    assert zerodha.compute([], CFG) == []
    assert zerodha.total_paise([]) == 0


@given(
    total=st.integers(0, 10**7),
    weights=st.lists(st.integers(1, 10**9), min_size=1, max_size=8),
)
def test_allocation_preserves_total_and_is_nonnegative(total: int, weights: list[int]) -> None:
    parts = zerodha.allocate(total, weights)
    assert sum(parts) == total and all(p >= 0 for p in parts) and len(parts) == len(weights)


@given(
    qty=st.integers(1, 5000),
    price=st.integers(100, 10_000_000),
    side=st.sampled_from(["BUY", "SELL"]),
    product=st.sampled_from(["CNC", "MIS"]),
    split=st.integers(1, 4),
)
def test_charges_nonnegative_integers_and_independent_of_fill_split(
    qty: int, price: int, side: str, product: str, split: int
) -> None:
    whole = [FillIn(1, "o", "X", side, product, qty, price, date(2026, 3, 2))]
    base, rem = divmod(qty, split)
    parts = [base + (1 if i < rem else 0) for i in range(split)]
    pieces = [
        FillIn(i + 1, "o", "X", side, product, q, price, date(2026, 3, 2))
        for i, q in enumerate(parts)
        if q > 0
    ]
    a, b = zerodha.compute(whole, CFG), zerodha.compute(pieces, CFG)
    assert all(x.amount_paise > 0 for x in a + b)
    assert zerodha.total_paise(a) == zerodha.total_paise(
        b
    )  # splitting fills never changes the bill
