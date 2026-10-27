"""Filesystem locations. Runtime data lives outside the repo, in ~/.tradepartner."""

import os
from pathlib import Path


def home_dir() -> Path:
    return Path(os.environ.get("TRADEPARTNER_HOME", "~/.tradepartner")).expanduser()


def config_dir() -> Path:
    env = os.environ.get("TRADEPARTNER_CONFIG_DIR")
    if env:
        return Path(env).expanduser()
    # src/tradepartner/paths.py -> repo root / config
    return Path(__file__).resolve().parents[2] / "config"


def db_path() -> Path:
    return home_dir() / "db.sqlite"


def logs_dir() -> Path:
    return home_dir() / "logs"


def kill_file() -> Path:
    return home_dir() / "KILL"


def reports_dir() -> Path:
    return home_dir() / "reports"


def backups_dir() -> Path:
    return home_dir() / "backups"


def models_dir() -> Path:
    return home_dir() / "models"
