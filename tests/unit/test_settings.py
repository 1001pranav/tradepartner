from pathlib import Path

import pytest
import yaml

from tradepartner.settings import ConfigError, load_settings, require_ready


def _edit(path: Path, fn) -> None:  # type: ignore[no-untyped-def]
    data = yaml.safe_load(path.read_text())
    fn(data)
    path.write_text(yaml.safe_dump(data))


def test_repo_config_loads() -> None:
    s = load_settings()
    assert s.risk.per_trade.max_risk_pct == 0.75
    assert set(s.strategies) == {"swing_v1", "positional_v1", "intraday_v1"}
    assert s.risk.capital_base_paise == 0


def test_refuses_to_start_with_zero_capital() -> None:
    with pytest.raises(ConfigError, match="capital_base_inr"):
        require_ready(load_settings())


def test_starts_with_capital_set(config_copy: Path) -> None:
    _edit(config_copy / "risk.yaml", lambda d: d.update(capital_base_inr=500_000))
    s = load_settings(config_copy)
    require_ready(s)
    assert s.risk.capital_base_paise == 50_000_000


def test_unknown_key_rejected(config_copy: Path) -> None:
    _edit(config_copy / "risk.yaml", lambda d: d.update(surprise=1))
    with pytest.raises(ConfigError):
        load_settings(config_copy)


def test_missing_file(config_copy: Path) -> None:
    (config_copy / "tax.yaml").unlink()
    with pytest.raises(ConfigError, match="missing"):
        load_settings(config_copy)


def test_invalid_yaml_and_non_mapping(config_copy: Path) -> None:
    (config_copy / "tax.yaml").write_text("a: [unclosed")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_settings(config_copy)
    (config_copy / "tax.yaml").write_text("- just\n- a list\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_settings(config_copy)


def test_order_rate_cannot_exceed_hard_cap(config_copy: Path) -> None:
    _edit(config_copy / "risk.yaml", lambda d: d["rate_limits"].update(max_orders_per_second=2))
    with pytest.raises(ConfigError):
        load_settings(config_copy)
    _edit(config_copy / "risk.yaml", lambda d: d["rate_limits"].update(max_orders_per_second=1))
    _edit(config_copy / "risk.yaml", lambda d: d["rate_limits"].update(max_orders_per_day=31))
    with pytest.raises(ConfigError):
        load_settings(config_copy)


def test_auto_limits_cannot_loosen(config_copy: Path) -> None:
    _edit(config_copy / "risk.yaml", lambda d: d["auto_only"].update(max_position_pct=20))
    with pytest.raises(ConfigError, match="max_position_pct"):
        load_settings(config_copy)
    _edit(config_copy / "risk.yaml", lambda d: d["auto_only"].update(max_position_pct=5))
    _edit(config_copy / "risk.yaml", lambda d: d["rate_limits"].update(max_orders_per_day=5))
    with pytest.raises(ConfigError, match="max_orders_per_day"):
        load_settings(config_copy)
    _edit(config_copy / "risk.yaml", lambda d: d["rate_limits"].update(max_orders_per_day=30))
    _edit(config_copy / "risk.yaml", lambda d: d["auto_only"].update(daily_loss_pct=3.0))
    with pytest.raises(ConfigError, match="daily_loss_pct"):
        load_settings(config_copy)


def test_non_equity_segment_rejected(config_copy: Path) -> None:
    _edit(config_copy / "risk.yaml", lambda d: d.update(allowed_segments=["NFO"]))
    with pytest.raises(ConfigError):
        load_settings(config_copy)


def test_ui_must_bind_localhost(config_copy: Path) -> None:
    _edit(config_copy / "app.yaml", lambda d: d["ui"].update(host="0.0.0.0"))
    with pytest.raises(ConfigError):
        load_settings(config_copy)


def test_no_strategies_dir(config_copy: Path) -> None:
    import shutil

    shutil.rmtree(config_copy / "strategies")
    assert load_settings(config_copy).strategies == {}
