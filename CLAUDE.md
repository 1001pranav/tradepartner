# CLAUDE.md

Guidance for Claude (and other coding agents) working in this repo.

## What this is
TradePartner is a local, AI-assisted trading partner for **NSE cash equities** via **Zerodha Kite Connect**, running on a 16 GB MacBook. The full spec is in `docs/BUILD_SPEC.md`, and it is the source of truth. Read the sections relevant to your task before writing code.

## How to work
- **Build one milestone per session**, in the order given in spec section 16. Do not start the next milestone until the current one's acceptance criteria pass. Write a short note in `docs/milestones/Mx.md` when one is done.
- Stay inside the milestone you were asked for. Do not scaffold future milestones "while you're there".
- Where the spec is ambiguous, choose the **safer** option (fewer orders, smaller size, OFF over AUTO), leave a `# SPEC-QUESTION:` comment, and mention it in your summary.
- Values marked **[VERIFY]** in the spec (fees, tax, SEBI rules) live in `config/*.yaml`. Never hard-code them.

## Commands
```bash
uv sync                          # install
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest --cov=tradepartner
uv run tradepartner --help       # CLI
```
Run all four checks before saying a task is done, and report any failures honestly.

## Hard rules (do not break these)
1. **Every order goes through `RiskEngine.check()` and `ModeController`.** Only `execution/router.py` may call `Broker.place_order`, and only with an `ApprovedOrder` created by the risk engine. Strategy, model, LLM and signal code must never import `execution.router` or `modes.controller.set_mode`. This is enforced by import-linter.
2. **Never change risk defaults** in `config/risk.yaml`, mode transition rules, or the promotion gate without Pranav explicitly asking. Call out any such change in the PR description.
3. **The system never starts in AUTO, never self-approves a model, and never automates the Kite login** (no stored passwords or TOTP seeds).
4. **The LLM is an analyst only.** It scores news and writes explanations. It never decides quantity, prices or whether to trade.
5. **No F&O, no other segments, no delivery short selling.** The broker adapter rejects anything except NSE cash equity.
6. **Money is integer paise; timestamps are UTC in storage and IST in display.** Never use float for money.
7. **`fills`, `news_raw`, `signals`, `order_events`, `risk_events` and `audit_log` are append-only.** Corrections are new rows.
8. **No lookahead.** Features use only data timestamped at or before `as_of`; labels start at the next bar. Respect the LLM knowledge-cutoff rule (spec section 10.3). Any new feature needs a leakage test.
9. **Tests never hit the network or a real broker.** Mock the Kite client and Ollama, and use fixtures in `tests/fixtures/`.
10. **Never place real orders while developing.** Live-order code paths (M8) are exercised only by Pranav, using 1-share orders.

## Testing expectations
- Write tests first for `risk/`, `modes/`, `costs/`, `ledger/` and `execution/`, with 100% branch coverage. Use hypothesis property tests for risk limits and ledger rebuilds.
- 80% coverage overall.
- Charge calculations must match the contract-note fixtures within ±₹1.

## Secrets and data
- Kite keys and tokens go in the macOS Keychain via `keyring`. Never put them in code, `.env`, logs or tests.
- Runtime data lives in `~/.tradepartner/`, never in the repo. Do not commit databases, Parquet files, models or real trade data. Fixtures must be synthetic or anonymized.
- The Streamlit UI binds to `127.0.0.1` only.

## Style
- Python 3.12, typed (mypy strict for `risk`, `modes`, `costs`, `ledger`, `execution`), with ruff for lint and format.
- Small pure functions, especially for risk rules: one function per rule, returning a `RuleResult`.
- Use structlog for logs. Never log secrets.
- Keep commits focused, and use one PR per milestone.
