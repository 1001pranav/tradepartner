"""Structural guarantees (CLAUDE.md hard rules 1, 3, 5, 6): checked from the source, not by trust."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).parents[2] / "src" / "tradepartner"
FORBIDDEN_FOR_AI = ("strategies", "models", "llm", "signals", "features", "news")


def modules() -> list[Path]:
    return [p for p in SRC.rglob("*.py") if "migrations" not in p.parts]


def test_only_the_router_calls_broker_place_order() -> None:
    callers: set[str] = set()
    for path in modules():
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "place_order"
            ):
                callers.add(str(path.relative_to(SRC)))
    assert callers == {"execution/router.py"}


def test_import_linter_contracts_hold() -> None:
    exe = Path(sys.executable).parent / "lint-imports"
    out = subprocess.run(
        [str(exe)], cwd=SRC.parents[1], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "4 kept, 0 broken" in out.stdout


@pytest.mark.parametrize("package", FORBIDDEN_FOR_AI)
def test_ai_and_strategy_code_never_import_execution_or_the_mode_controller(package: str) -> None:
    for path in (SRC / package).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module, *[f"{node.module}.{a.name}" for a in node.names]]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            for n in names:
                assert not n.startswith("tradepartner.execution"), f"{path} imports {n}"
                assert n != "tradepartner.modes.controller", f"{path} imports the mode controller"


def test_the_system_has_no_f_and_o_or_short_delivery_path() -> None:
    from tradepartner.settings import load_settings

    risk = load_settings().risk
    assert risk.allowed_segments == ["NSE_EQ"] and set(risk.allowed_products) == {"CNC", "MIS"}
    assert risk.allow_intraday_short is False and risk.allow_market_orders is False
    assert risk.rate_limits.max_orders_per_second == 1 and risk.rate_limits.max_orders_per_day <= 30


def test_risk_defaults_in_the_repo_are_the_spec_defaults() -> None:
    """Rule 2: nobody loosened the shipped defaults. Change these only with Pranav's approval."""
    from tradepartner.settings import load_settings

    r = load_settings().risk
    assert (
        r.per_trade.max_risk_pct,
        r.per_trade.max_position_pct,
        r.per_trade.max_stop_distance_pct,
    ) == (0.75, 10, 8)
    assert (
        r.loss_limits.daily_loss_pct,
        r.loss_limits.weekly_loss_pct,
        r.loss_limits.max_drawdown_pct,
    ) == (2.0, 4.0, 10.0)
    assert (r.portfolio.max_open_positions, r.portfolio.max_sector_exposure_pct) == (8, 30)
    assert (
        r.auto_only.max_orders_per_day,
        r.auto_only.max_position_pct,
        r.auto_only.daily_loss_pct,
    ) == (10, 5, 1.0)
    assert r.capital_base_inr == 0  # Pranav sets this


def test_the_ui_binds_to_localhost_only() -> None:
    from tradepartner.settings import load_settings

    assert load_settings().app.ui.host == "127.0.0.1"


def test_no_money_amounts_are_floats_in_the_database_schema() -> None:
    from sqlalchemy import Float

    from tradepartner.db.models import Base

    money_words = ("paise",)
    for table in Base.metadata.tables.values():
        for col in table.columns:
            if any(w in col.name for w in money_words):
                assert not isinstance(col.type, Float), (
                    f"{table.name}.{col.name} must be an integer"
                )
