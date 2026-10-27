"""Shared Streamlit helpers. The UI binds to 127.0.0.1 only (see ``ops`` and ``cli ui``)."""

import streamlit as st
from sqlalchemy.orm import Session

from tradepartner.db.migrate import upgrade
from tradepartner.db.session import make_engine, make_session_factory
from tradepartner.paths import home_dir
from tradepartner.settings import Settings, load_settings


@st.cache_resource
def _factory(home: str):  # type: ignore[no-untyped-def]
    # keyed by the data directory so a changed TRADEPARTNER_HOME never reuses a stale engine
    home_dir().mkdir(parents=True, exist_ok=True)
    engine = make_engine()
    upgrade(engine.url.render_as_string(hide_password=False))
    return make_session_factory(engine)


def get_session() -> Session:
    return _factory(str(home_dir()))()  # type: ignore[no-any-return]


def get_settings() -> Settings:
    return load_settings()
