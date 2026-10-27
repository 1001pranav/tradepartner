"""Drift monitors (spec 10.6): population-stability index on the top features and the Brier
score of predictions over a rolling window of realised outcomes. Exceeding a threshold raises an
alert and downgrades the strategy to MANUAL (the downgrade itself is applied by the runtime)."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PSI_THRESHOLD = 0.25  # > 0.25 is a major shift by the usual convention
BRIER_MARGIN = 0.05  # allowed worsening over the training-time Brier score
ROLLING_TRADES = 60
BINS = 10


def feature_bins(values: pd.Series, bins: int = BINS) -> dict[str, list[float]]:
    """Quantile edges and expected proportions of a training-time feature distribution."""
    v = values.dropna().to_numpy(dtype="float64")
    edges = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.array([v.min() - 1, v.mean(), v.max() + 1])
    edges[0], edges[-1] = -np.inf, np.inf
    counts, _ = np.histogram(v, bins=edges)
    return {"edges": [float(e) for e in edges], "expected": [float(c / len(v)) for c in counts]}


def psi(edges: list[float], expected: list[float], actual: pd.Series) -> float:
    a = actual.dropna().to_numpy(dtype="float64")
    if len(a) == 0:
        return float("nan")
    counts, _ = np.histogram(a, bins=np.array(edges))
    act = np.clip(counts / len(a), 1e-4, None)
    exp = np.clip(np.array(expected), 1e-4, None)
    return float(np.sum((act - exp) * np.log(act / exp)))


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


@dataclass
class DriftReport:
    psi: dict[str, float] = field(default_factory=dict)
    brier: float | None = None
    brier_reference: float | None = None
    n_outcomes: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return bool(self.reasons)


def check_drift(
    stored_bins: dict[str, dict[str, list[float]]],
    recent_features: pd.DataFrame,
    train_brier: float | None,
    recent_probs: np.ndarray,
    recent_outcomes: np.ndarray,
    psi_threshold: float = PSI_THRESHOLD,
    brier_margin: float = BRIER_MARGIN,
) -> DriftReport:
    rep = DriftReport()
    for feat, b in stored_bins.items():
        if feat in recent_features:
            value = psi(b["edges"], b["expected"], recent_features[feat])
            rep.psi[feat] = value
            if value == value and value > psi_threshold:
                rep.reasons.append(f"PSI {value:.2f} on {feat} exceeds {psi_threshold}")
    probs, outs = recent_probs[-ROLLING_TRADES:], recent_outcomes[-ROLLING_TRADES:]
    rep.n_outcomes = len(outs)
    if len(outs) >= 20 and train_brier is not None:
        rep.brier = brier(probs, outs)
        rep.brier_reference = train_brier
        if rep.brier > train_brier + brier_margin:
            rep.reasons.append(
                f"Brier {rep.brier:.3f} over the last {len(outs)} outcomes "
                f"vs {train_brier:.3f} in training"
            )
    return rep
