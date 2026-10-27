"""Versioned prompt templates (spec 10.2). Changing a prompt means a new file and a new version."""

from dataclasses import dataclass
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"


@dataclass(frozen=True)
class Prompt:
    name: str  # e.g. news_score
    version: str  # e.g. v1
    template: str

    @property
    def id(self) -> str:
        return f"{self.name}_{self.version}"

    def render(self, **values: str) -> str:
        return self.template.format(**values)


def load(name: str, version: str = "v1", directory: Path | None = None) -> Prompt:
    path = (directory or PROMPT_DIR) / f"{name}_{version}.md"
    return Prompt(name, version, path.read_text())


NEWS_SCORE_VERSION = "v1"
