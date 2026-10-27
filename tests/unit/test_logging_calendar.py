import json
from datetime import UTC, datetime

import pytest
import structlog

from tradepartner.calendar import to_ist, utc_now
from tradepartner.logging import REDACTED, configure_logging, redact_secrets


def test_redacts_secret_keys() -> None:
    out = redact_secrets(
        None, "info", {"event": "x", "access_token": "abc", "api_key": "k", "n": 1}
    )
    assert out["access_token"] == REDACTED
    assert out["api_key"] == REDACTED
    assert out["n"] == 1


def test_logging_end_to_end_to_file(capsys: pytest.CaptureFixture[str]) -> None:
    from tradepartner.paths import logs_dir

    configure_logging(to_file=True)
    structlog.get_logger().info("hello", password="hunter2", symbol="INFY")
    err = capsys.readouterr().err
    record = json.loads(err.strip().splitlines()[-1])
    assert record["password"] == REDACTED
    assert record["symbol"] == "INFY"
    assert (logs_dir() / "tradepartner.log").exists()


def test_to_ist() -> None:
    assert to_ist(datetime(2026, 1, 1, 0, 0, tzinfo=UTC)).hour == 5
    with pytest.raises(ValueError):
        to_ist(datetime(2026, 1, 1))
    assert utc_now().tzinfo is not None
