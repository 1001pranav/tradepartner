"""Walk-forward LightGBM training (spec 10.5). Hyperparameter search (Optuna, at most 50 trials)
runs INSIDE each training fold only, so the test window never influences the choice."""

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from tradepartner.models.cv import Split, inner_split, walk_forward_splits

MAX_TRIALS = 50
Objective = Literal["binary", "regression"]
optuna.logging.set_verbosity(optuna.logging.WARNING)


def base_params(objective: Objective, seed: int, threads: int = 4) -> dict[str, Any]:
    return {
        "objective": objective,
        "metric": "binary_logloss" if objective == "binary" else "l2",
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": threads,
        "verbosity": -1,
        "feature_pre_filter": False,
    }


def fit(
    x: pd.DataFrame,
    y: np.ndarray,
    params: dict[str, Any],
    rounds: int,
    valid: tuple[pd.DataFrame, np.ndarray] | None = None,
) -> lgb.Booster:
    train = lgb.Dataset(x, y, free_raw_data=False)
    if valid is None:
        return lgb.train(params, train, num_boost_round=rounds)
    vset = lgb.Dataset(valid[0], valid[1], reference=train)
    return lgb.train(
        params,
        train,
        num_boost_round=rounds,
        valid_sets=[vset],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )


def tune(
    x: pd.DataFrame,
    y: np.ndarray,
    horizon: int,
    objective: Objective,
    seed: int,
    n_trials: int,
    threads: int = 4,
) -> tuple[dict[str, Any], int]:
    """Best params and boosting rounds from a time-ordered inner split of the training fold."""
    n_trials = min(n_trials, MAX_TRIALS)
    tr, va = inner_split(len(x), horizon)
    if len(tr) < 50 or len(va) < 20 or n_trials <= 0:
        return {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 50}, 100
    xt, yt, xv, yv = x.iloc[tr], y[tr], x.iloc[va], y[va]

    def objective_fn(trial: optuna.Trial) -> float:
        p = {
            **base_params(objective, seed, threads),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
        }
        booster = fit(xt, yt, p, 300, (xv, yv))
        trial.set_user_attr("rounds", max(1, booster.best_iteration))
        return float(booster.best_score["valid_0"][p["metric"]])

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial
    return dict(best.params), int(best.user_attrs["rounds"])


@dataclass
class WalkForward:
    oos: pd.DataFrame  # index (date, symbol): pred, y, fwd_ret, excess, window
    splits: list[Split]
    params: dict[str, Any]
    rounds: int
    feature_names: list[str]
    importance: dict[str, float] = field(default_factory=dict)
    window_params: list[dict[str, Any]] = field(default_factory=list)


def _prep(matrix: pd.DataFrame, labels: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = matrix[columns].join(labels[["y", "fwd_ret", "excess"]], how="inner")
    return df.sort_index()


def walk_forward(
    matrix: pd.DataFrame,
    labels: pd.DataFrame,
    columns: list[str],
    *,
    objective: Objective,
    horizon: int,
    n_trials: int = 25,
    seed: int = 42,
    test_months: int = 3,
    min_train_days: int = 504,
    threads: int = 4,
) -> WalkForward:
    df = _prep(matrix, labels, columns)
    dates = sorted(df.index.get_level_values("date").unique())
    splits = walk_forward_splits(dates, horizon, test_months, min_train_days)
    date_pos = {d: i for i, d in enumerate(dates)}
    pos = np.array([date_pos[d] for d in df.index.get_level_values("date")])
    labelled = df["y"].notna().to_numpy()
    preds: list[pd.DataFrame] = []
    window_params: list[dict[str, Any]] = []
    last_params: dict[str, Any] = {}
    last_rounds = 100
    for sp in splits:
        tr_mask = np.isin(pos, sp.train_idx) & labelled
        te_mask = np.isin(pos, sp.test_idx) & labelled
        if tr_mask.sum() < 100 or te_mask.sum() == 0:
            continue
        x_tr, y_tr = df.loc[tr_mask, columns], df.loc[tr_mask, "y"].to_numpy()
        params, rounds = tune(x_tr, y_tr, horizon, objective, seed, n_trials, threads)
        booster = fit(
            x_tr,
            y_tr,
            {**base_params(objective, seed, threads), **params, "bagging_freq": 1},
            rounds,
        )
        out = df.loc[te_mask, ["y", "fwd_ret", "excess"]].copy()
        out["pred"] = booster.predict(df.loc[te_mask, columns])
        out["window"] = sp.window
        preds.append(out)
        window_params.append({"window": sp.window, "params": params, "rounds": rounds})
        last_params, last_rounds = params, rounds
    oos = (
        pd.concat(preds)
        if preds
        else pd.DataFrame(columns=["y", "fwd_ret", "excess", "pred", "window"])
    )
    return WalkForward(oos, splits, last_params, last_rounds, columns, {}, window_params)


def fit_final(
    matrix: pd.DataFrame,
    labels: pd.DataFrame,
    columns: list[str],
    wf: WalkForward,
    objective: Objective,
    seed: int = 42,
    threads: int = 4,
) -> tuple[lgb.Booster, date, date]:
    """Train on everything labelled up to the latest date (the weekly retrain, spec 10.6)."""
    df = _prep(matrix, labels, columns)
    df = df[df["y"].notna()]
    params = {**base_params(objective, seed, threads), **wf.params, "bagging_freq": 1}
    booster = fit(df[columns], df["y"].to_numpy(), params, wf.rounds)
    dts = df.index.get_level_values("date")
    return booster, min(dts), max(dts)


def gain_share(booster: lgb.Booster) -> dict[str, float]:
    gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype="float64")
    total = gain.sum()
    names = booster.feature_name()
    return {n: float(g / total) if total > 0 else 0.0 for n, g in zip(names, gain, strict=True)}
