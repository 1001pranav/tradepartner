# TradePartner

A local, AI-assisted trading partner for Indian cash equities (NSE) via Zerodha Kite Connect, built to run on a MacBook.

- A local LLM (Ollama) acts as a **news analyst** and explains signals. It never places orders.
- LightGBM models make predictions, with walk-forward validation and a promotion gate that Pranav approves.
- A plain-code **risk engine** gates every order. It has **Off / Manual / Auto** modes per strategy, never starts in Auto, and has a kill switch.
- A full **trade ledger** (bot and manual trades) with Zerodha charges and Indian tax categories, reconciled to Console.

No F&O. Swing and positional strategies first; intraday last.

## Status

Planning done; nothing built yet. The full specification is in [`docs/BUILD_SPEC.md`](docs/BUILD_SPEC.md).

## Build order

Build **one milestone per session**, and only move on when its acceptance tests pass (spec section 16):

| # | Milestone |
|---|---|
| M0 | Scaffold (uv, config, DB schema, CLI) |
| M1 | Market data (Kite login, instruments, candles) |
| M2 | Ledger, costs and tax (usable on its own) |
| M3 | Risk engine, modes, paper broker |
| M4 | Backtester and baselines |
| M5 | News ingestion and LLM analyst (start collecting news early) |
| M6 | ML pipeline and model registry |
| M7 | Signals, explanations, UI, Manual flow |
| M8 | Auto execution via Kite (needs a static IP; 1-share tests first) |
| M9 | Intraday |

## Before going live

Facts marked **[VERIFY]** in the spec (SEBI rules, Zerodha charges, tax) must be re-checked before real money is involved. Tax treatment should be confirmed with a CA. This project is for personal use only; signals must not be shared with others.
