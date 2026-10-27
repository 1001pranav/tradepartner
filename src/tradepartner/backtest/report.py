"""HTML backtest report saved under ~/.tradepartner/reports/ (spec 13)."""

from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd

from tradepartner.backtest import metrics
from tradepartner.backtest.engine import BacktestResult


def _fmt(x: object) -> str:
    if isinstance(x, float):
        return "n/a" if x != x else f"{x:,.4f}"
    return escape(str(x))


def _table(df: pd.DataFrame) -> str:
    head = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{_fmt(v)}</td>" for v in r) + "</tr>"
        for r in df.itertuples(index=False)
    )
    return f"<table><tr>{head}</tr>{rows}</table>"


def metrics_frame(results: dict[str, metrics.Metrics]) -> pd.DataFrame:
    rows = []
    for name, m in results.items():
        d = m.as_dict()
        rows.append({"name": name, **{k: v for k, v in d.items() if k not in ("start", "end")}})
    return pd.DataFrame(rows)


def render(
    result: BacktestResult,
    baselines: dict[str, metrics.Metrics],
    index_close: pd.Series | None = None,
    title: str = "Backtest report",
) -> str:
    parts = [f"<h1>{escape(title)}: {escape(result.strategy)}</h1>"]
    for w in result.warnings:
        parts.append(f'<p class="warn"><b>{escape(w)}</b></p>')
    p = result.params
    parts.append(
        f"<p>Period {p['start']} to {p['end']}. Mode {p['mode']}. Slippage {p['slippage_bps']} bps "
        f"per fill. Costs: full Zerodha schedule from <code>config/costs.yaml</code> "
        f"(brokerage, STT, exchange, SEBI, stamp, GST, DP). Same RiskEngine as live trading.</p>"
    )
    parts.append("<h2>Performance (after costs and slippage)</h2>")
    parts.append(_table(metrics_frame({result.strategy: result.metrics, **baselines})))
    m = result.metrics
    parts.append(
        f"<p>Cost drag: {_fmt(m.cost_drag)} of gross profit. Trades: {m.n_trades}. "
        f"Unfilled limit orders: {result.unfilled}. Skipped for cash: {result.skipped_cash}.</p>"
    )
    if len(result.equity) > 1:
        parts.append("<h2>By year</h2>" + _table(metrics.by_year(result.equity)))
        if index_close is not None and len(index_close) > 200:
            parts.append(
                "<h2>By regime (Nifty vs 200-day average, previous day)</h2>"
                + _table(metrics.by_regime(result.equity, index_close))
            )
    if result.rejections:
        top = pd.DataFrame(result.rejections.most_common(15), columns=["risk rule", "blocks"])
        parts.append("<h2>Why orders were rejected by the risk engine</h2>" + _table(top))
    if result.trades:
        tf = result.trade_frame()
        for c in ("entry_price_paise", "exit_price_paise", "gross_pnl_paise", "charges_paise",
                  "slippage_paise", "pnl_paise"):  # fmt: skip
            tf[c] = tf[c] / 100
        parts.append("<h2>Last 50 trades (INR)</h2>" + _table(tf.tail(50)))
    style = (
        "body{font-family:system-ui,sans-serif;margin:2rem;max-width:1100px}"
        "table{border-collapse:collapse;margin:1rem 0}td,th{border:1px solid #ccc;padding:4px 8px;"
        "font-size:13px}th{background:#f2f2f2}.warn{background:#fff3cd;padding:8px}"
    )
    return (
        f"<!doctype html><meta charset=utf-8><title>{escape(title)}</title><style>{style}</style>"
        + "".join(parts)
    )


def save(html: str, directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = directory / f"{name}-{stamp}.html"
    path.write_text(html)
    return path
