"""FIFO matching, intraday netting, corporate actions, tax categories (spec 9.4, 15)."""

from datetime import date, datetime, timedelta
from fractions import Fraction

from hypothesis import given, settings
from hypothesis import strategies as st

from tests.fixtures.market import dt
from tradepartner.data.corporate_actions import Action
from tradepartner.ledger import lots, tax
from tradepartner.ledger.lots import FillRec, RealizedRow
from tradepartner.settings import load_settings

TAX = load_settings().tax


def f(i: int, side: str, qty: int, px: int, day: date, hh: int = 10, sym: str = "INFY") -> FillRec:
    ts = dt(day.year, day.month, day.day, hh, i % 60)
    return FillRec(i, sym, "INE1", side, qty, px, ts, day)


D = date(2026, 3, 2)


def test_simple_fifo_two_lots_partial() -> None:
    fills = [
        f(1, "BUY", 10, 100_00, D),
        f(2, "BUY", 10, 120_00, D + timedelta(days=1)),
        f(3, "SELL", 15, 150_00, D + timedelta(days=5)),
    ]
    res = lots.match(fills)
    assert [(r.qty, r.buy_value_paise, r.holding_days) for r in res.realized] == [
        (10, 10 * 100_00, 5),
        (5, 5 * 120_00, 4),
    ]
    assert res.lots[0].qty_rem == 5 and res.lots[0].cost_per_share_paise == 120_00
    assert all(r.kind == "delivery" for r in res.realized)


def test_intraday_netting_even_when_cnc_and_leftover_buy_opens_lot() -> None:
    fills = [f(1, "BUY", 10, 100_00, D, 9), f(2, "SELL", 4, 110_00, D, 11)]
    res = lots.match(fills)
    assert len(res.realized) == 1 and res.realized[0].kind == "intraday"
    assert res.realized[0].qty == 4 and res.realized[0].holding_days == 0
    assert res.lots[0].qty_rem == 6  # leftover 6 shares carried as a delivery lot


def test_intraday_short_sell_then_buy() -> None:
    res = lots.match([f(1, "SELL", 5, 110_00, D, 9), f(2, "BUY", 5, 100_00, D, 11)])
    r = res.realized[0]
    assert (r.kind, r.open_fill_id, r.close_fill_id) == ("intraday", 1, 2)
    assert r.sell_value_paise - r.buy_value_paise == 5 * 10_00


def test_intraday_pairing_spans_multiple_fills() -> None:
    fills = [
        f(1, "BUY", 3, 100_00, D, 9),
        f(2, "BUY", 3, 101_00, D, 10),
        f(3, "SELL", 4, 102_00, D, 11),
        f(4, "SELL", 2, 103_00, D, 12),
    ]
    res = lots.match(fills)
    assert sorted((r.buy_fill_id, r.sell_fill_id, r.qty) for r in res.realized) == [
        (1, 3, 3), (2, 3, 1), (2, 4, 2),
    ]  # fmt: skip
    assert not res.lots


def test_sell_leftover_consumes_older_lots_after_intraday() -> None:
    fills = [
        f(1, "BUY", 10, 100_00, D),
        f(2, "BUY", 2, 105_00, D + timedelta(days=3), 9),
        f(3, "SELL", 5, 110_00, D + timedelta(days=3), 11),
    ]
    res = lots.match(fills)
    kinds = sorted((r.kind, r.qty) for r in res.realized)
    assert kinds == [("delivery", 3), ("intraday", 2)]
    assert sum(lot.qty_rem for lot in res.lots) == 7


def test_unmatched_sell_never_invents_cost_basis() -> None:
    res = lots.match([f(1, "SELL", 5, 110_00, D)])
    r = res.realized[0]
    assert r.kind == "unmatched" and r.buy_value_paise == 0 and r.open_fill_id is None
    assert res.alerts and "no matching buy" in res.alerts[0]
    partial = lots.match(
        [f(1, "BUY", 2, 100_00, D), f(2, "SELL", 5, 110_00, D + timedelta(days=2))]
    )
    assert [r.kind for r in partial.realized] == ["delivery", "unmatched"]


def test_split_adjusts_lot_qty_and_cost() -> None:
    fills = [f(1, "BUY", 10, 1000_00, D), f(2, "SELL", 50, 220_00, D + timedelta(days=30))]
    split = Action("INFY", "split", D + timedelta(days=10), 5, 1)
    res = lots.match(fills, [split])
    r = res.realized[0]
    assert r.qty == 50 and r.buy_value_paise == 10 * 1000_00  # total cost preserved
    assert r.open_date == D  # holding period unchanged by a split


def test_split_fractional_and_wiped_lots_alert() -> None:
    fills = [f(1, "BUY", 3, 100_00, D), f(2, "BUY", 1, 100_00, D + timedelta(days=1))]
    res = lots.match(fills, [Action("INFY", "split", D + timedelta(days=5), 1, 2)])
    assert [lot.qty_rem for lot in res.lots] == [1]  # 3->1 (fraction dropped), 1->0 removed
    assert any("fractional" in a for a in res.alerts)
    assert res.lots[0].adjusted and res.lots[0].qty_open == 1


def test_bonus_adds_zero_cost_lot_with_new_open_date() -> None:
    ex = D + timedelta(days=10)
    fills = [f(1, "BUY", 10, 100_00, D), f(2, "SELL", 20, 60_00, ex + timedelta(days=400))]
    res = lots.match(fills, [Action("INFY", "bonus", ex, 1, 1)])
    by = {r.open_date: r for r in res.realized}
    assert by[D].buy_value_paise == 10 * 100_00
    assert by[ex].buy_value_paise == 0 and by[ex].holding_days == 400  # clock starts at allotment
    assert tax.classify(by[ex], TAX) == "ltcg" and tax.classify(by[D], TAX) == "ltcg"


def test_bonus_with_no_extra_shares_and_unrelated_actions_ignored() -> None:
    res = lots.match(
        [f(1, "BUY", 1, 100_00, D)],
        [Action("INFY", "bonus", D + timedelta(days=1), 1, 3), Action("INFY", "dividend", D)],
    )
    assert len(res.lots) == 1 and res.lots[0].qty_rem == 1


def test_action_before_same_day_fill_does_not_adjust_new_buy() -> None:
    ex = D + timedelta(days=1)
    res = lots.match(
        [f(1, "BUY", 10, 100_00, D), f(2, "BUY", 10, 20_00, ex)],
        [Action("INFY", "split", ex, 5, 1)],
    )
    assert sorted(lot.qty_rem for lot in res.lots) == [10, 50]


def test_round_half_up() -> None:
    assert lots.round_half_up(Fraction(5, 2)) == 3
    assert lots.round_half_up(Fraction(-5, 2)) == -3
    assert lots.round_half_up(Fraction(7, 3)) == 2
    assert lots.LotState(1, "X", None, D, 0, 0, Fraction(0)).cost_per_share_paise == 0


def _row(open_date: date | None, close: date, kind: str = "delivery") -> RealizedRow:
    return RealizedRow("X", kind, 1, 2, 1, 2, 1, 100, 110, 0, open_date, close)  # type: ignore[arg-type]


def test_tax_category_boundaries() -> None:
    buy = date(2025, 3, 10)
    assert tax.classify(_row(buy, date(2026, 3, 10)), TAX) == "stcg"  # exactly 12 months
    assert tax.classify(_row(buy, date(2026, 3, 11)), TAX) == "ltcg"  # 12 months + 1 day
    assert tax.classify(_row(buy, buy, "intraday"), TAX) == "intraday_spec"
    assert tax.classify(_row(None, buy, "unmatched"), TAX) == "unmatched"


def test_business_income_treatment_only_short_term() -> None:
    cfg = TAX.model_copy(update={"swing_treatment": "business_income"})
    assert tax.classify(_row(date(2026, 1, 1), date(2026, 2, 1)), cfg) == "business"
    assert tax.classify(_row(date(2024, 1, 1), date(2026, 2, 1)), cfg) == "ltcg"


def test_fy_boundary_31_mar_1_apr() -> None:
    assert tax.fy_of(date(2027, 3, 31)) == "2026-27" and tax.fy_of(date(2027, 4, 1)) == "2027-28"


def test_pnl_stt_deductibility() -> None:
    # capital gains: STT is NOT deductible; speculative / business: it is
    assert tax.pnl_paise("stcg", 1000, 600, 50, 30) == 350
    assert tax.pnl_paise("ltcg", 1000, 600, 50, 30) == 350
    assert tax.pnl_paise("intraday_spec", 1000, 600, 50, 30) == 320
    assert tax.pnl_paise("business", 1000, 600, 50, 30) == 320
    assert tax.pnl_paise("unmatched", 1000, 0, 50, 30) == 0


def test_setoff_rules() -> None:
    lakh = 100_000 * 100
    s = tax.suggest_setoff(-50 * lakh // 100, 3 * lakh, 0, 0, TAX)  # STCL 0.5L vs LTCG 3L
    assert s.ltcg_after == 3 * lakh - 50 * lakh // 100 and "Short-term loss" in s.notes[0]
    assert s.ltcg_exemption_used == TAX.ltcg_exemption_inr * 100
    assert s.ltcg_taxable == s.ltcg_after - s.ltcg_exemption_used
    t = tax.suggest_setoff(2 * lakh, -lakh, -lakh, 7, TAX)  # LTCL cannot offset STCG
    assert t.stcg_after == 2 * lakh and t.carry_forward_capital_loss == lakh
    assert t.carry_forward_speculative_loss == lakh and t.speculative_after == 0
    assert any("cannot be set off" in n for n in t.notes) and t.business_after == 7
    assert any("8 years" in n for n in t.notes) and any("4 years" in n for n in t.notes)
    u = tax.suggest_setoff(0, 0, 5, 0, TAX)
    assert u.notes == () and u.speculative_after == 5 and u.ltcg_exemption_used == 0


@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.sampled_from(["BUY", "SELL"]),
            st.integers(1, 50),
            st.integers(1, 500),
            st.integers(0, 40),
            st.integers(0, 6),
        ),
        min_size=1,
        max_size=25,
    )
)
def test_matching_conserves_quantity_and_is_deterministic(
    ops: list[tuple[str, int, int, int, int]],
) -> None:
    base = date(2026, 1, 1)
    fills = [
        FillRec(
            i + 1,
            "X",
            None,
            side,
            qty,
            px * 100,
            datetime(2026, 1, 1, 9 + h % 6, tzinfo=dt(2026, 1, 1).tzinfo),
            base + timedelta(days=d),
        )
        for i, (side, qty, px, d, h) in enumerate(ops)
    ]
    a, b = lots.match(fills), lots.match(list(reversed(fills)))
    assert a.realized == b.realized and a.lots == b.lots  # input order never matters
    bought = sum(x.qty for x in fills if x.side == "BUY")
    sold = sum(x.qty for x in fills if x.side == "SELL")
    open_qty = sum(lot.qty_rem for lot in a.lots)
    matched_buy = sum(r.qty for r in a.realized if r.buy_fill_id is not None)
    unmatched = sum(r.qty for r in a.realized if r.kind == "unmatched")
    assert bought == open_qty + matched_buy
    assert sold == sum(r.qty for r in a.realized)
    assert unmatched >= 0 and all(r.qty > 0 for r in a.realized)


@given(st.lists(st.integers(1, 30), min_size=1, max_size=10))
def test_buy_then_sell_all_leaves_no_lots_and_pnl_sums(qtys: list[int]) -> None:
    fills = [f(i + 1, "BUY", q, 100_00 + i, D + timedelta(days=i)) for i, q in enumerate(qtys)]
    fills.append(f(99, "SELL", sum(qtys), 120_00, D + timedelta(days=60)))
    res = lots.match(fills)
    assert not res.lots and sum(r.qty for r in res.realized) == sum(qtys)
    assert sum(r.buy_value_paise for r in res.realized) == sum(
        q * (100_00 + i) for i, q in enumerate(qtys)
    )
