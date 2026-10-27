from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import DatabaseError, IntegrityError
from sqlalchemy.orm import Session

from tradepartner.db import models
from tradepartner.db.migrate import upgrade
from tradepartner.db.models import APPEND_ONLY_TABLES, Base, Fill, RiskEvent
from tradepartner.db.session import make_engine, make_session_factory

NOW = datetime(2026, 10, 26, 4, 0, tzinfo=UTC)


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    url = f"sqlite:///{tmp_path / 'test.sqlite'}"
    upgrade(url)
    return url


def test_migration_creates_every_model_table(db_url: str) -> None:
    tables = set(inspect(make_engine(db_url)).get_table_names())
    assert set(Base.metadata.tables) <= tables
    assert {"fills", "news_raw", "signals", "audit_log", "mode_state"} <= tables


def test_wal_and_foreign_keys(db_url: str) -> None:
    with make_engine(db_url).connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1


def _fill(trade_id: str = "T1") -> Fill:
    return Fill(
        broker_trade_id=trade_id,
        origin="paper",
        account="X",
        symbol="INFY",
        exchange="NSE",
        side="BUY",
        product="CNC",
        qty=1,
        price_paise=150000,
        filled_at=NOW,
        trade_date=date(2026, 10, 26),
        source="paper",
    )


def test_money_round_trips_as_integer_paise_and_utc(db_url: str) -> None:
    factory = make_session_factory(make_engine(db_url))
    with factory() as s:
        s.add(_fill())
        s.commit()
    with factory() as s:
        row = s.query(Fill).one()
        assert row.price_paise == 150000
        assert row.filled_at == NOW
        assert row.created_at.tzinfo is not None


def test_naive_datetime_rejected(db_url: str) -> None:
    factory = make_session_factory(make_engine(db_url))
    bad = _fill()
    bad.filled_at = datetime(2026, 10, 26, 9, 30)
    with factory() as s:
        s.add(bad)
        with pytest.raises(Exception, match="naive"):
            s.commit()


def test_null_datetime_passthrough() -> None:
    t = models.UTCDateTime()
    assert t.process_bind_param(None, None) is None  # type: ignore[arg-type]
    assert t.process_result_value(None, None) is None  # type: ignore[arg-type]


def test_fills_are_append_only(db_url: str) -> None:
    factory = make_session_factory(make_engine(db_url))
    with factory() as s:
        s.add(_fill())
        s.commit()
    with factory() as s, pytest.raises(DatabaseError, match="append-only"):
        s.execute(text("UPDATE fills SET qty = 99"))
    with factory() as s, pytest.raises(DatabaseError, match="append-only"):
        s.execute(text("DELETE FROM fills"))


def test_every_append_only_table_has_both_triggers(db_url: str) -> None:
    with make_engine(db_url).connect() as conn:
        names = {
            r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='trigger'"))
        }
    for table in APPEND_ONLY_TABLES:
        assert f"{table}_no_update" in names
        assert f"{table}_no_delete" in names


def test_duplicate_broker_trade_id_rejected(db_url: str) -> None:
    factory = make_session_factory(make_engine(db_url))
    with factory() as s:
        s.add_all([_fill("T1"), _fill("T1")])
        with pytest.raises(IntegrityError):
            s.commit()


def test_non_nse_fill_rejected(db_url: str) -> None:
    factory = make_session_factory(make_engine(db_url))
    bad = _fill()
    bad.exchange = "NFO"
    with factory() as s:
        s.add(bad)
        with pytest.raises(IntegrityError):
            s.commit()


def test_risk_event_severity_constraint(db_url: str) -> None:
    factory = make_session_factory(make_engine(db_url))
    with factory() as s:
        s.add(RiskEvent(at=NOW, rule="r", severity="panic", scope="global", details_json={}))
        with pytest.raises(IntegrityError):
            s.commit()


def test_downgrade_removes_tables(db_url: str) -> None:
    from alembic import command

    from tradepartner.db.migrate import alembic_config

    command.downgrade(alembic_config(db_url), "base")
    assert set(inspect(make_engine(db_url)).get_table_names()) <= {"alembic_version"}


def test_default_engine_uses_home(isolated_home: Path) -> None:
    engine = make_engine()
    with Session(engine) as s:
        s.execute(text("SELECT 1"))
    assert (isolated_home / "db.sqlite").exists()
