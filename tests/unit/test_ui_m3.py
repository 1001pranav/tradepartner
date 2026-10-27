from pathlib import Path

import yaml
from streamlit.testing.v1 import AppTest

from tradepartner import cli
from tradepartner.modes.controller import ModeController
from tradepartner.modes.state import GLOBAL, strategy_scope
from tradepartner.risk import kill
from tradepartner.settings import load_settings

PAGES = Path(__file__).parents[2] / "src" / "tradepartner" / "ui" / "pages"


def page(name: str) -> AppTest:
    return AppTest.from_file(str(PAGES / name), default_timeout=30).run()


def test_modes_page_sets_global_mode_and_hides_locked_auto() -> None:
    at = page("modes.py")
    assert not at.exception
    assert "AUTO" not in at.selectbox(key="sel-strategy:swing_v1").options  # locked: not offered
    assert "AUTO" in at.selectbox(key="sel-global").options
    at.selectbox(key="sel-global").select("MANUAL")
    at.button(key="apply-global").click().run()
    with cli._session() as s:
        assert ModeController(s).current(GLOBAL).mode == "MANUAL"


def test_modes_page_unlock_refused_without_evidence() -> None:
    at = page("modes.py")
    at.selectbox(key="unlock-target").select("swing_v1")
    at.text_input[0].set_value("swing_v1")
    at.button[-1].click().run()
    assert at.error and "cannot unlock AUTO" in at.error[0].value
    with cli._session() as s:
        assert not ModeController(s).current(strategy_scope("swing_v1")).auto_unlocked


def test_risk_page_with_zero_capital_warns() -> None:
    at = page("risk.py")
    assert not at.exception and at.error and "capital_base_inr is 0" in at.error[0].value


def test_risk_page_with_capital_reset_and_kill(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = tmp_path / "config"
    import shutil

    shutil.copytree(Path(__file__).parents[2] / "config", cfg)
    d = yaml.safe_load((cfg / "risk.yaml").read_text())
    d["capital_base_inr"] = 500_000
    (cfg / "risk.yaml").write_text(yaml.safe_dump(d))
    monkeypatch.setenv("TRADEPARTNER_CONFIG_DIR", str(cfg))
    assert load_settings().risk.capital_base_inr == 500_000
    with cli._session() as s:
        ModeController(s).set_mode(
            GLOBAL,
            "MANUAL",
            "user",
            "x",
            __import__("datetime").datetime.now(__import__("datetime").UTC),
        )
        ModeController(s).halt(
            GLOBAL, "test halt", __import__("datetime").datetime.now(__import__("datetime").UTC)
        )
    at = page("risk.py")
    assert not at.exception and len(at.progress) >= 4
    at.text_input[0].set_value("nope")
    at.button[0].click().run()
    assert at.error  # wrong phrase refused
    at.text_input[0].set_value("RESET")
    at.button[0].click().run()
    assert at.success
    at.text_input[1].set_value("wrong")
    at.button[1].click().run()
    assert not kill.kill_active()
    at.text_input[1].set_value("KILL")
    at.button[1].click().run()
    assert kill.kill_active() and at.warning
    with cli._session() as s:
        assert ModeController(s).current(GLOBAL).mode == "OFF"
