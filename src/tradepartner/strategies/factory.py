"""Build a runnable strategy from a registered model."""

from typing import Any

from tradepartner.db.models import Model
from tradepartner.models import registry
from tradepartner.settings import Settings
from tradepartner.strategies import positional, swing
from tradepartner.strategies.model_strategy import ModelStrategy, NewsProvider

CLASSES: dict[str, type[ModelStrategy]] = {
    swing.NAME: swing.SwingStrategy,
    positional.NAME: positional.PositionalStrategy,
}


def build_model_strategy(
    model: Model, settings: Settings, news_provider: NewsProvider | None = None
) -> ModelStrategy:
    cfg = settings.strategies.get(model.strategy)
    params: dict[str, Any] = {
        **(cfg.model_dump() if cfg else {}),
        "horizon_days": model.metrics_json.get("horizon_days", 5),
        "top_k": model.metrics_json.get("top_k", 5),
    }
    return CLASSES[model.strategy](
        model.strategy, registry.load_booster(model), list(model.metrics_json["feature_columns"]),
        params, str(model.metrics_json["objective"]), news_provider,
    )  # fmt: skip
