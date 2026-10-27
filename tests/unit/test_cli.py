from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from tradepartner.cli import app

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("login", "run", "kill", "mode", "backtest", "train", "sync", "report", "db"):
        assert cmd in result.output


def test_check_refuses_with_zero_capital() -> None:
    result = runner.invoke(app, ["check"])
    assert result.exit_code == 2
    assert "capital_base_inr" in result.output


def test_run_refuses_with_zero_capital() -> None:
    assert runner.invoke(app, ["run"]).exit_code == 2


def test_check_passes_when_capital_set(config_copy: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = config_copy / "risk.yaml"
    data = yaml.safe_load(path.read_text())
    data["capital_base_inr"] = 100_000
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("TRADEPARTNER_CONFIG_DIR", str(config_copy))
    assert runner.invoke(app, ["check"]).exit_code == 0
    # Config is valid, so `run` gets past the startup gate and starts the scheduler (patched out).
    monkeypatch.setattr("tradepartner.scheduler.start", lambda ctx: None)
    assert runner.invoke(app, ["run"]).exit_code == 0


def test_db_upgrade_creates_schema(isolated_home: Path) -> None:
    result = runner.invoke(app, ["db", "upgrade"])
    assert result.exit_code == 0, result.output
    assert (isolated_home / "db.sqlite").exists()


def test_kill_creates_kill_file(isolated_home: Path) -> None:
    assert runner.invoke(app, ["kill"]).exit_code == 0
    assert (isolated_home / "KILL").exists()


@pytest.mark.parametrize(
    "args",
    [
        ["report", "daily"],
    ],
)
def test_future_milestone_stubs_exit_nonzero(args: list[str]) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "Not implemented" in result.output
