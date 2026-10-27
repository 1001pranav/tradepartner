from datetime import date

from tests.fixtures.market import dt, make_cal
from tradepartner.calendar import (
    TradingCalendar,
    fmt_ist,
    fy_bounds,
    indian_fy,
    months_between_exceeds,
    session_close_utc,
    session_open_utc,
)


def test_weekends_and_holidays() -> None:
    cal = make_cal()
    assert cal.is_trading_day(date(2026, 1, 23))  # Friday
    assert not cal.is_trading_day(date(2026, 1, 24))  # Saturday
    assert not cal.is_trading_day(date(2026, 1, 26))  # holiday
    assert cal.next_trading_day(date(2026, 1, 23)) == date(2026, 1, 27)
    assert cal.prev_trading_day(date(2026, 1, 27)) == date(2026, 1, 23)


def test_extra_trading_day_and_ranges() -> None:
    cal = TradingCalendar(extra_trading_days=frozenset({date(2026, 11, 8)}))  # Sunday special
    assert cal.is_trading_day(date(2026, 11, 8))
    assert cal.trading_days(date(2026, 11, 6), date(2026, 11, 9)) == [
        date(2026, 11, 6),
        date(2026, 11, 8),
        date(2026, 11, 9),
    ]
    assert cal.add_trading_days(date(2026, 11, 6), 2) == date(2026, 11, 9)
    assert cal.add_trading_days(date(2026, 11, 9), -2) == date(2026, 11, 6)


def test_from_config_repo_file_and_missing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert not TradingCalendar.from_config().is_trading_day(date(2026, 12, 25))
    assert TradingCalendar.from_config(tmp_path / "nope.yaml").is_trading_day(date(2026, 12, 25))


def test_session_times_are_ist() -> None:
    assert session_close_utc(date(2026, 3, 2)).hour == 10  # 15:30 IST = 10:00 UTC
    assert session_open_utc(date(2026, 3, 2)).hour == 3
    assert fmt_ist(session_close_utc(date(2026, 3, 2))) == "2026-03-02 15:30 IST"
    assert dt(2026, 1, 1).utcoffset() is not None


def test_fy_boundaries() -> None:
    assert indian_fy(date(2027, 3, 31)) == "2026-27"
    assert indian_fy(date(2027, 4, 1)) == "2027-28"
    assert fy_bounds("2026-27") == (date(2026, 4, 1), date(2027, 3, 31))


def test_long_term_boundary_is_strictly_more_than_12_months() -> None:
    buy = date(2025, 3, 10)
    assert not months_between_exceeds(buy, date(2026, 3, 10), 12)  # exactly 12 months: short term
    assert months_between_exceeds(buy, date(2026, 3, 11), 12)
    leap = date(2024, 2, 29)
    assert not months_between_exceeds(leap, date(2025, 2, 28), 12)
    assert months_between_exceeds(leap, date(2025, 3, 1), 12)
    assert months_between_exceeds(date(2025, 1, 31), date(2025, 12, 1), 10)  # month-end clamp
    assert not months_between_exceeds(date(2025, 12, 31), date(2026, 12, 31), 12)
