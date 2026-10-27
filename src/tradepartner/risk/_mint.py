"""The private token that authorises construction of an ``ApprovedOrder``.

Only ``risk/engine.py`` and ``risk/approved.py`` may import this module. This is enforced by an
import-linter contract (see pyproject.toml), so strategy, model, LLM and signal code can never
manufacture an order the broker would accept (spec 3, CLAUDE.md rule 1).
"""


class _MintToken:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<mint-token>"


TOKEN = _MintToken()
