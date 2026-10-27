"""Map articles to symbols (spec 10.1): alias table from instrument names, ISINs and common short
names. Title is matched before body. Ambiguous aliases are resolved by the LLM (it must pick one
of the candidates, or none)."""

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.db.models import Instrument
from tradepartner.paths import config_dir

ISIN_RE = re.compile(r"\bINE[0-9A-Z]{8}\d\b")
SUFFIXES = re.compile(
    r"\b(limited|ltd\.?|corporation|corp\.?|company|co\.?|\(india\)|of india|pvt\.?)\b", re.I
)
TICKER_STOPLIST = {"IT", "ALL", "ONE", "NEW", "CAN", "FOR", "ARE", "NOT", "BIG", "TOP"}
TITLE_CONFIDENCE, BODY_CONFIDENCE, AMBIGUOUS_CONFIDENCE = 0.9, 0.6, 0.4


@dataclass(frozen=True)
class Match:
    symbol: str
    method: str  # alias | isin
    confidence: float


@dataclass(frozen=True)
class Ambiguous:
    alias: str
    candidates: tuple[str, ...]


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", SUFFIXES.sub(" ", name.lower())).strip(" .,-")


class AliasTable:
    def __init__(
        self,
        names: dict[str, str] | None = None,
        isins: dict[str, str] | None = None,
        extra: dict[str, list[str]] | None = None,
    ) -> None:
        """``names``: symbol -> company name; ``isins``: isin -> symbol;
        ``extra``: alias -> symbols (several symbols = ambiguous)."""
        self.isins = isins or {}
        alias_map: dict[str, set[str]] = defaultdict(set)
        for sym, name in (names or {}).items():
            norm = normalize_name(name)
            if len(norm) >= 4:
                alias_map[norm].add(sym)
        for alias, syms in (extra or {}).items():
            alias_map[alias.lower().strip()].update(syms)
        self.aliases = {a: tuple(sorted(s)) for a, s in alias_map.items() if a}
        self.tickers = {s for s in (names or {}) if len(s) >= 3 and s not in TICKER_STOPLIST}
        ordered = sorted(self.aliases, key=len, reverse=True)
        self._alias_re = (
            re.compile(
                r"(?<![\w&])(" + "|".join(re.escape(a) for a in ordered) + r")(?![\w&])", re.I
            )
            if ordered
            else None
        )
        self._ticker_re = (
            re.compile(
                r"(?<![\w&])("
                + "|".join(re.escape(t) for t in sorted(self.tickers, key=len, reverse=True))
                + r")(?![\w&])"
            )
            if self.tickers
            else None
        )

    @classmethod
    def from_session(cls, session: Session, aliases_csv: Path | None = None) -> "AliasTable":
        rows = session.execute(
            select(Instrument.tradingsymbol, Instrument.name, Instrument.isin).where(
                Instrument.segment == "NSE"
            )
        ).all()
        names = {sym: name for sym, name, _ in rows if name}
        names.update({sym: sym for sym, name, _ in rows if not name})
        isins = {isin: sym for sym, _, isin in rows if isin}
        extra: dict[str, list[str]] = defaultdict(list)
        path = aliases_csv or config_dir() / "universe" / "aliases.csv"
        if path.exists():
            lines = [
                ln for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")
            ]
            for r in csv.DictReader(lines):
                extra[r["alias"]].append(r["symbol"].strip())
        return cls(names, isins, dict(extra))

    def _scan(self, text: str, confidence: float) -> tuple[list[Match], list[Ambiguous]]:
        matches: list[Match] = []
        amb: list[Ambiguous] = []
        for isin in ISIN_RE.findall(text):
            if isin in self.isins:
                matches.append(Match(self.isins[isin], "isin", 0.99))
        if self._alias_re:
            for m in self._alias_re.finditer(text):
                syms = self.aliases[m.group(1).lower()]
                if len(syms) == 1:
                    matches.append(Match(syms[0], "alias", confidence))
                else:
                    amb.append(Ambiguous(m.group(1).lower(), syms))
        if self._ticker_re:
            for m in self._ticker_re.finditer(text):
                matches.append(Match(m.group(1), "alias", confidence))
        return matches, amb

    def match(self, title: str, body: str = "") -> tuple[list[Match], list[Ambiguous]]:
        """Title first; the body is only consulted if the title names nothing."""
        found, amb = self._scan(title, TITLE_CONFIDENCE)
        if not found and not amb:
            found, amb = self._scan(body, BODY_CONFIDENCE)
        best: dict[str, Match] = {}
        for m in found:
            if m.symbol not in best or m.confidence > best[m.symbol].confidence:
                best[m.symbol] = m
        # an ambiguous alias is dropped when an unambiguous match already covers its candidates
        amb = [a for a in amb if not set(a.candidates) & set(best)]
        return sorted(best.values(), key=lambda m: m.symbol), amb
