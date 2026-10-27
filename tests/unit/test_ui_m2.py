from pathlib import Path

from streamlit.testing.v1 import AppTest

from tests.unit.test_ledger_db import BOOK, tb, write
from tradepartner.cli import _session
from tradepartner.ledger import importers, rebuild
from tradepartner.settings import load_settings
from tradepartner.ui import data

PAGES = Path(__file__).parents[2] / "src" / "tradepartner" / "ui" / "pages"


def _seed(tmp_path: Path) -> None:
    s = load_settings()
    with _session() as session:
        importers.import_tradebook(session, write(tmp_path, "tb.csv", tb(BOOK)))
        rebuild.rebuild(session, s.costs, s.tax)


def test_rupee_formatting_indian_grouping() -> None:
    assert data.rupees(0) == "₹0.00"
    assert data.rupees(123456789) == "₹12,34,567.89"
    assert data.rupees(-100000) == "-₹1,000.00"
    assert data.rupees(99999900) == "₹9,99,999.00"
    assert data.rupees(1234500) == "₹12,345.00"


def test_ledger_frame_filters(tmp_path: Path) -> None:
    _seed(tmp_path)
    with _session() as s:
        assert len(data.ledger_frame(s)) == 5
        assert len(data.ledger_frame(s, origins=["paper"])) == 0
        assert len(data.ledger_frame(s, products=["CNC"], symbol="tcs")) == 2
        from datetime import date

        assert len(data.ledger_frame(s, start=date(2026, 1, 1))) == 3
        assert len(data.ledger_frame(s, end=date(2025, 12, 31))) == 2


def test_ledger_page_renders_and_corrects(tmp_path: Path) -> None:
    _seed(tmp_path)
    at = AppTest.from_file(str(PAGES / "ledger.py"), default_timeout=30).run()
    assert not at.exception
    assert at.dataframe[0].value.shape[0] == 5
    at.number_input[0].set_value(1)
    at.number_input[1].set_value(12)
    at.number_input[2].set_value(1490.0)
    at.button[0].click().run()
    assert at.error[0].value == "a reason is required"
    at.text_input[1].set_value("typo").run()
    at.button[0].click().run()
    assert not at.exception and at.success


def test_pnl_tax_page_renders(tmp_path: Path) -> None:
    _seed(tmp_path)
    at = AppTest.from_file(str(PAGES / "pnl_tax.py"), default_timeout=30).run()
    assert not at.exception
    assert len(at.metric) == 5
    at.button[0].click().run()
    assert at.success and not at.exception


def test_app_entry_builds_navigation() -> None:
    at = AppTest.from_file(str(PAGES.parent / "app.py"), default_timeout=30).run()
    assert not at.exception
