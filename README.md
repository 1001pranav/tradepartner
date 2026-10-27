# TradePartner

A local, AI-assisted trading partner for Indian cash equities (NSE) via Zerodha Kite Connect, built to run on a MacBook.

- A local LLM (Ollama) acts as a **news analyst** and explains signals. It never places orders.
- LightGBM models make predictions, with walk-forward validation and a promotion gate that Pranav approves.
- A plain-code **risk engine** gates every order. It has **Off / Manual / Auto** modes per strategy, never starts in Auto, and has a kill switch.
- A full **trade ledger** (bot and manual trades) with Zerodha charges and Indian tax categories, reconciled to Console.

No F&O. Swing and positional strategies first; intraday last.

## Status

All milestones M0 to M9 are built, each with its note in [`docs/milestones/`](docs/milestones/). The specification is [`docs/BUILD_SPEC.md`](docs/BUILD_SPEC.md); rules for coding agents are in [`CLAUDE.md`](CLAUDE.md).

| # | Milestone | Status |
|---|---|---|
| M0 | Scaffold (uv, config, DB schema, CLI) | done |
| M1 | Market data (Kite login, instruments, candles) | done (run the backfill yourself) |
| M2 | Ledger, costs and tax | done (add your real contract notes) |
| M3 | Risk engine, modes, paper broker | done, 100% branch coverage |
| M4 | Backtester and baselines | done |
| M5 | News ingestion and LLM analyst | done (7-day collection + throughput check are yours) |
| M6 | ML pipeline and model registry | done (nothing is ever auto-approved) |
| M7 | Signals, explanations, UI, Manual flow | done (10 unattended days are yours) |
| M8 | Auto execution via Kite | code done; **live 1-share acceptance is yours** (`docs/milestones/M8.md`) |
| M9 | Intraday | code done; **paper-only until 3 months of evidence** |

Nothing in this repository has traded real money. Live-order paths were tested only against fakes.

## Quick start

```bash
uv sync
uv run tradepartner --help
# 1. Set your capital in config/risk.yaml (capital_base_inr). The system refuses to run at 0.
uv run tradepartner check
uv run tradepartner login                       # manual Kite login, token goes to the macOS Keychain
uv run tradepartner data backfill --universe nifty100 --years 10
uv run tradepartner ledger import-tradebook ~/Downloads/tradebook.csv   # your history
uv run tradepartner report tax --fy 2026-27
uv run tradepartner backtest --strategy baseline_momentum20
uv run tradepartner train --strategy swing_v1   # registers a CANDIDATE only
uv run tradepartner ui                          # http://127.0.0.1:8501
uv run tradepartner run                         # the scheduler (see ops/ for launchd)
```

Modes: `tradepartner mode` shows them; the system never starts in AUTO; `tradepartner kill` is the kill switch.

## Before going live

Facts marked **[VERIFY]** in the spec (SEBI rules, Zerodha charges, tax) must be re-checked before real money is involved. Tax treatment should be confirmed with a CA. This project is for personal use only; signals must not be shared with others.
