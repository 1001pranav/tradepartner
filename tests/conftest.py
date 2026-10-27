import shutil
from pathlib import Path

import keyring
import keyring.backend
import keyring.errors
import pytest

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config"


@pytest.fixture
def config_copy(tmp_path: Path) -> Path:
    """A writable copy of config/ so tests can alter values."""
    dest = tmp_path / "config"
    shutil.copytree(REPO_CONFIG, dest)
    return dest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Tests never touch the real ~/.tradepartner."""
    home = tmp_path / "home"
    monkeypatch.setenv("TRADEPARTNER_HOME", str(home))
    return home


class MemoryKeyring(keyring.backend.KeyringBackend):
    """Tests must never touch the real macOS Keychain."""

    priority = 1  # type: ignore[assignment]

    def __init__(self) -> None:
        super().__init__()
        self.store: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, username: str, password: str) -> None:
        self.store[(service, username)] = password

    def get_password(self, service: str, username: str) -> str | None:
        return self.store.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        if (service, username) not in self.store:
            raise keyring.errors.PasswordDeleteError("missing")
        del self.store[(service, username)]


@pytest.fixture(autouse=True)
def memory_keyring() -> MemoryKeyring:
    backend = MemoryKeyring()
    keyring.set_keyring(backend)
    return backend


@pytest.fixture
def db(tmp_path: Path):  # type: ignore[no-untyped-def]
    """A migrated, empty SQLite session."""
    from tradepartner.db.migrate import upgrade
    from tradepartner.db.session import make_engine, make_session_factory

    url = f"sqlite:///{tmp_path / 'db.sqlite'}"
    upgrade(url)
    with make_session_factory(make_engine(url))() as session:
        yield session


@pytest.fixture(autouse=True)
def pristine_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Tests run against a copy of config/ with capital 0 (the shipped default), so editing your
    real config/risk.yaml never changes test outcomes. Tests that need another config override
    TRADEPARTNER_CONFIG_DIR themselves."""
    import yaml

    dest = tmp_path / "pristine_config"
    shutil.copytree(REPO_CONFIG, dest)
    risk = yaml.safe_load((dest / "risk.yaml").read_text())
    risk["capital_base_inr"] = 0
    (dest / "risk.yaml").write_text(yaml.safe_dump(risk, sort_keys=False))
    monkeypatch.setenv("TRADEPARTNER_CONFIG_DIR", str(dest))
    return dest
