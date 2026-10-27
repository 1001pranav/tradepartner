"""Pydantic output schemas for the LLM analyst (spec 10.2). Validated on every response."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

EventType = Literal[
    "earnings", "guidance", "order_win", "management_change", "regulatory", "litigation",
    "m&a", "capital_raise", "rating_change", "macro", "other",
]  # fmt: skip
Horizon = Literal["intraday", "days", "weeks", "months"]
MAX_SUMMARY_WORDS = 40


class NewsScoreOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    relevant: bool
    relevance: float = Field(ge=0, le=1)
    event_type: EventType
    sentiment: float = Field(ge=-1, le=1)
    materiality: float = Field(ge=0, le=1)
    expected_horizon: Horizon
    confidence: float = Field(ge=0, le=1)
    summary: str

    @field_validator("summary")
    @classmethod
    def _short_summary(cls, v: str) -> str:
        if len(v.split()) > MAX_SUMMARY_WORDS:
            raise ValueError(f"summary longer than {MAX_SUMMARY_WORDS} words")
        return v


class SymbolChoice(BaseModel):
    """Entity-link disambiguation: the model must pick from the candidate list, or none."""

    model_config = ConfigDict(extra="forbid")

    symbol: str | None


def json_schema(model: type[BaseModel]) -> dict[str, object]:
    return model.model_json_schema()
