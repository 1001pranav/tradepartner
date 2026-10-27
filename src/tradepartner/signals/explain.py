"""LLM explanations with a number check (spec 10.2).

The LLM sees the top feature contributions, recent scored news and price context, and writes 3 to
5 sentences: why the model likes it and what would make it wrong. Every number in the output must
appear in the facts it was given; otherwise it is retried once and finally replaced by a
deterministic template that is built only from those same facts.
"""

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, ConfigDict, field_validator

from tradepartner.llm import prompts
from tradepartner.llm.client import LLMUnavailableError, OllamaClient

NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
MAX_CHARS = 900


class ExplanationOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    explanation: str

    @field_validator("explanation")
    @classmethod
    def _bounded(cls, v: str) -> str:
        if not v.strip() or len(v) > MAX_CHARS:
            raise ValueError("explanation empty or too long")
        return v.strip()


def canonical(number: str) -> str:
    """Comparable form: '1,500.00' -> '1500', '62.50' -> '62.5'."""
    try:
        d = Decimal(number.replace(",", "").rstrip("."))
    except InvalidOperation:
        return number
    return format(d.normalize(), "f")


def numbers_in(text: str) -> set[str]:
    return {canonical(m.group(0)) for m in NUMBER.finditer(text)}


def unsupported_numbers(explanation: str, facts: str) -> list[str]:
    """Numbers in the explanation that are not in the facts (must be empty to pass)."""
    allowed = numbers_in(facts)
    return sorted(n for n in numbers_in(explanation) if n not in allowed)


def f2(x: float) -> str:
    return f"{x:.2f}"


@dataclass(frozen=True)
class Facts:
    symbol: str
    prob_pct: float | None
    contributions: tuple[tuple[str, float], ...]
    news: tuple[str, ...]
    close_rupees: float
    ret_20_pct: float | None
    entry_rupees: float
    stop_rupees: float
    target_rupees: float
    horizon_days: int

    def text(self) -> str:
        lines = [f"Symbol: {self.symbol}"]
        if self.prob_pct is not None:
            lines.append(f"Model score (percent): {f2(self.prob_pct)}")
        lines.append(f"Last close (rupees): {f2(self.close_rupees)}")
        if self.ret_20_pct is not None:
            lines.append(f"20-day return (percent): {f2(self.ret_20_pct)}")
        lines.append(f"Planned entry limit (rupees): {f2(self.entry_rupees)}")
        lines.append(f"Protective stop (rupees): {f2(self.stop_rupees)}")
        lines.append(f"Target (rupees): {f2(self.target_rupees)}")
        lines.append(f"Time stop (trading days): {self.horizon_days}")
        for name, value in self.contributions:
            lines.append(f"Driver: {name} (contribution {f2(value)})")
        for item in self.news:
            lines.append(f"News: {item}")
        return "\n".join(lines)


def template_explanation(f: Facts) -> str:
    """Deterministic fallback. Built from the facts only, so it passes the number check."""
    drivers = (
        ", ".join(f"{n} ({f2(v)})" for n, v in f.contributions[:3]) or "no single dominant driver"
    )
    parts = [
        f"{f.symbol} ranks highly in the model"
        + (f" with a score of {f2(f.prob_pct)} percent" if f.prob_pct is not None else "")
        + f". The main drivers are {drivers}.",
        f"The last close was {f2(f.close_rupees)} rupees"
        + (
            f" after a 20-day move of {f2(f.ret_20_pct)} percent"
            if f.ret_20_pct is not None
            else ""
        )
        + ".",
    ]
    if f.news:
        parts.append(f"Recent news: {f.news[0]}.")
    parts.append(
        f"The idea is wrong if price falls to the stop at {f2(f.stop_rupees)} rupees, or if it has "
        f"not reached {f2(f.target_rupees)} rupees within {f.horizon_days} trading days."
    )
    return " ".join(parts)


@dataclass(frozen=True)
class Explanation:
    text: str
    used_llm: bool
    rejected_numbers: tuple[str, ...] = ()


class Explainer:
    def __init__(self, llm: OllamaClient | None) -> None:
        self.llm = llm
        self.prompt = prompts.load("explain", "v1")

    def explain(self, facts: Facts) -> Explanation:
        text = facts.text()
        if self.llm is None:
            return Explanation(template_explanation(facts), False)
        bad: list[str] = []
        for _ in range(2):  # one retry, then the template
            try:
                res = self.llm.generate(self.prompt.render(facts=text), ExplanationOut)
            except LLMUnavailableError:
                break
            if res.valid and isinstance(res.parsed, ExplanationOut):
                bad = unsupported_numbers(res.parsed.explanation, text)
                if not bad:
                    return Explanation(res.parsed.explanation, True)
        return Explanation(template_explanation(facts), False, tuple(bad))
