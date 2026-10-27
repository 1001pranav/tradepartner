"""Kill-file helpers (spec 7.1 rule 4). The kill *procedure* lives in ``execution.killswitch``."""

from pathlib import Path

from tradepartner.paths import kill_file


def kill_active(path: Path | None = None) -> bool:
    return (path or kill_file()).exists()


def arm(path: Path | None = None) -> Path:
    p = path or kill_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    return p


def clear(path: Path | None = None) -> None:
    (path or kill_file()).unlink(missing_ok=True)
