# TradePartner: Build Specification

A local, AI-assisted trading partner for Indian cash equities (NSE), built to run on a 16 GB MacBook Pro.

- **Owner:** Pranav
- **Spec version:** 1.0 (2026-10-15)
- **Intended reader:** a coding agent (for example Claude Sonnet) implementing this one milestone at a time, and Pranav reviewing it.

> **How to use this spec (for the coding agent)**
> 1. Build one milestone per session, in order (section 16). Do not start a milestone until the previous one's acceptance tests pass.
> 2. Write tests first for anything in `risk/`, `ledger/`, `costs/` and `modes/`. These modules protect money and must have 100% branch coverage.
> 3. Tests never touch the network or a real broker. Mock the Kite client.
> 4. Never change a default in `config/risk.yaml` or the mode state machine without Pranav's explicit approval in the PR description.
> 5. When this spec is ambiguous, pick the safer option (fewer orders, smaller size, OFF over AUTO) and leave a `# SPEC-QUESTION:` comment.
> 6. Items marked **[VERIFY]** are facts (fees, tax, regulation) to check against the official source before relying on them in live trading. Keep them configurable, never hard-coded.

---

## 1. Goals and non-goals

### Goals
1. Generate trade ideas for **swing** (2 to 15 trading days), **positional/delivery** (1 to 12+ months) and, later, **intraday** strategies in NSE cash equities.
2. Use a **local LLM** as an analyst. It reads news, filings and announcements, scores sentiment and events, and writes a plain-English explanation for every signal.
3. Use **classical ML** (LightGBM) to predict well-defined, tractable targets from price, volume and LLM-derived features.
4. Enforce **hard risk limits in plain code** that no model can bypass.
5. Provide an **Off / Manual / Auto mode switch** per strategy, plus a global switch and kill switch.
6. Keep a **complete transaction ledger** of every trade, both bot trades and Pranav's own manual trades in the same account, with charges, P&L and tax classification ready for a CA.
7. **Learn over time** through scheduled, gated retraining, never through live self-modification.

### Non-goals (do not build)
- F&O, commodities, currency, crypto. The broker adapter must reject any non-equity-cash segment.
- Short selling of delivery positions. Intraday shorts are disabled by default (section 7).
- High-frequency trading. The design ceiling is far below SEBI's 10 orders-per-second threshold (section 2.1).
- Sharing signals with anyone else. That would need SEBI RA/IA registration (section 2.3).
- An LLM that predicts prices or places orders directly.
- Automated broker login with stored passwords or TOTP secrets (section 2.1).

### Honest expectations (keep these in mind)
- Limits cap losses; they do not prevent them. A worst-case daily and total loss is defined in config and must be acceptable to Pranav before AUTO is enabled.
- SEBI's study of FY23 found that around 7 in 10 individual intraday equity traders lost money. Intraday is therefore built **last** and gated hardest.
- A strategy that cannot beat a simple baseline after costs in walk-forward testing is not deployed, however good the model "looks".

---

## 2. Regulatory, broker and tax constraints

These were checked on 2026-10-15. Everything here is **[VERIFY]** before live trading, because the rules have been changing.

### 2.1 SEBI retail algo framework (fully in force from 1 April 2026)
- **What counts as API trading:** orders placed through a broker API (Kite Connect) are a separate, tagged category. Orders Pranav places by hand in the Kite app or web are ordinary manual orders, even when the idea came from this system.
- **Static IP:** API orders must come from a static IP registered with the broker. Kite allows one primary and one secondary IP. A home ISP static IP or a small VPS (roughly ₹1,500/year upward) works. *This is only needed for API order placement (AUTO mode and the optional "approve via API" path). Reading data and placing orders manually in Kite do not need it.*
- **10 orders per second (OPS):** strategies above 10 OPS must be registered with the exchange through the broker and audited. This system is hard-capped at **1 order per second and 30 orders per day** (configurable lower, never higher without a code change).
- **Market orders through the API** must carry a non-zero **market protection** percentage. This system uses **limit orders by default**. Market orders are allowed only for emergency exits, and then always with market protection set.
- **Daily login:** Kite access tokens expire daily. Pranav logs in manually each morning through the Kite login redirect. Do **not** automate the login with stored passwords or TOTP seeds.
- Running a strategy and trading manually in the same account can interfere. The ledger and position sync must handle positions the bot did not create (section 9.4).

### 2.2 Broker: Zerodha Kite Connect
| Plan | Price | Orders | Live WebSocket | Historical candles |
|---|---|---|---|---|
| Personal | Free | Yes | No | No |
| Connect | ₹500/app/month | Yes | Yes | Yes (incl. intraday history) |

- **Decision:** use **Kite Connect (₹500/month)**, because the system needs historical and live data. From October 2026 Zerodha credits the fee back in months with at least ₹2,000 of brokerage. Do not plan around that, since delivery brokerage is zero.
- Python client: `kiteconnect` (official).
- Respect Kite rate limits. Defaults: quotes 1 req/s, historical 3 req/s, orders 10 req/s. This system self-limits to well below these, using a token-bucket limiter per endpoint. **[VERIFY]** the current limits in Kite docs.
- Groww is out of scope for v1. Keep the `Broker` interface clean so a Groww adapter could be added later.

### 2.3 Investment advice
Signals are for Pranav's own account only. Publishing or sharing them, paid or free, risks being treated as unregistered investment advice or research. The UI must not have any "share" or "publish" feature.

### 2.4 Tax (India, FY 2026-27 onward) **[VERIFY with a CA]**
The Income-tax Act, 2025 applies from 1 April 2026, so section numbers differ from the 1961 Act. The tax engine stores **categories**, not section numbers.

| Trade type | How the system identifies it | Tax treatment | Notes |
|---|---|---|---|
| Intraday equity | Bought and sold the same scrip on the same day (MIS, or CNC squared off the same day) | **Speculative business income**, at slab rate | Losses can only be set off against speculative income. Carry forward 4 years. Needs ITR-3. |
| Delivery, held ≤ 12 months | CNC buy, sold on a later day within 12 months | **STCG**: 20% | STT must have been paid. |
| Delivery, held > 12 months | CNC, sold after 12 months | **LTCG**: 12.5% on gains above ₹1.25 lakh per FY | Exemption is per FY across all LTCG. |
| Capital losses | | STCL can be set off against STCG and LTCG; LTCL only against LTCG | Carry forward 8 years, and only if the return is filed on time. |

- **Frequent swing trading:** a high-frequency swing book may be argued to be business income rather than capital gains. CBDT guidance (2016) lets taxpayers choose for listed shares, but the choice must be applied consistently. The engine must support both treatments via `tax.swing_treatment: capital_gains | business_income` (default `capital_gains`). This is Pranav's call, with their CA.
- **Expenses:** brokerage and charges reduce capital gains, but STT does **not** (STT is not deductible for capital gains). For business income, STT and charges are business expenses. The engine records each charge component separately so either treatment can be computed.
- **Advance tax:** the daily report shows realized gains per quarter so advance tax can be estimated.
- **Corporate actions** (splits, bonuses) adjust cost basis and holding period per tax rules. Use the ex-date. The holding period of bonus shares starts at allotment.
- The source of truth for filing is **Zerodha Console's Tax P&L and contract notes**. The internal ledger must reconcile to them (section 9.5).

---

## 3. Architecture overview

```
                   ┌──────────────────────────────────────────────┐
                   │                  Scheduler                   │
                   │       (APScheduler, IST, trading calendar)   │
                   └───┬──────────┬───────────┬───────────┬───────┘
                       │          │           │           │
              ┌────────▼──┐ ┌─────▼─────┐ ┌───▼──────┐ ┌──▼────────────┐
              │ Market    │ │ News      │ │ Feature  │ │ Model training│
              │ data      │ │ ingest +  │ │ builder  │ │ (weekly,      │
              │ (Kite)    │ │ LLM       │ │          │ │  gated)       │
              └────┬──────┘ │ analyst   │ └────┬─────┘ └──────┬────────┘
                   │        └─────┬─────┘      │              │
                   │              │            │       ┌──────▼────────┐
                   │              │            │       │ Model registry│
                   │              │            │       └──────┬────────┘
                   │              │      ┌─────▼──────────────▼──┐
                   │              │      │ Strategy engines      │
                   │              │      │ (swing/positional/    │
                   │              │      │  intraday)            │
                   │              │      └─────┬─────────────────┘
                   │              │            │ Signal + explanation
                   │              │      ┌─────▼─────────────────┐
                   │              └─────►│ Order proposal builder│
                   │                     └─────┬─────────────────┘
                   │                     ┌─────▼─────────────────┐
                   │                     │ RISK ENGINE + MODES   │◄── config/risk.yaml
                   │                     │ (plain code, no AI)   │◄── kill switch
                   │                     └─────┬─────────────────┘
                   │           OFF: log only   │  MANUAL: show in UI / AUTO: execute
                   │                     ┌─────▼─────────────────┐
                   └────────────────────►│ Broker adapter        │
                                         │ Paper | Kite          │
                                         └─────┬─────────────────┘
                                         ┌─────▼─────────────────┐
                                         │ Ledger + tax engine   │◄── Kite trade sync,
                                         │ (SQLite, append-only) │    Console CSV import
                                         └─────┬─────────────────┘
                                         ┌─────▼─────────────────┐
                                         │ Streamlit UI (local)  │
                                         └───────────────────────┘
```

**Key invariant:** the *only* path to `Broker.place_order()` goes through `RiskEngine.check()` and `ModeController.allows()`. Enforce this structurally. `KiteBroker.place_order` is private to the `execution` package and takes an `ApprovedOrder` type that only `RiskEngine` can construct. Use a module-private factory, not a public constructor.

---

## 4. Tech stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Ecosystem |
| Env/packaging | `uv`, `pyproject.toml` | Fast and reproducible |
| Transactional DB | SQLite (WAL mode) via SQLAlchemy 2.0 + Alembic | Local, single file, simple to back up |
| Market data store | Parquet files, partitioned by `interval/symbol`, queried with DuckDB | Fast analytics on 16 GB |
| DataFrames | pandas 2.x (polars allowed in feature builder) | |
| ML | LightGBM, scikit-learn, optuna (bounded trials) | Strong on tabular data, cheap on CPU |
| Local LLM | Ollama with `qwen2.5:7b-instruct-q4_K_M` (default) or `llama3.1:8b-instruct-q4_K_M` | Fits 16 GB alongside other work |
| LLM structured output | Ollama `format` = JSON schema, validated with pydantic | |
| Broker | `kiteconnect` | Official |
| Scheduling | APScheduler (in-process), launched by `launchd` | Native macOS |
| UI | Streamlit, bound to `127.0.0.1` | Fast to build |
| Config | YAML + pydantic-settings | Typed and validated |
| Secrets | macOS Keychain via `keyring` | Never in files or the repo |
| Logging | `structlog` JSON to `~/.tradepartner/logs/`, rotated daily | Audit |
| Notifications | macOS notifications (`pync` or `osascript`) + optional Telegram bot | Alerts |
| Tests | pytest, hypothesis, freezegun, pytest-cov | |
| Lint/type | ruff, mypy (strict for `risk`, `ledger`, `costs`, `modes`, `execution`) | |

### Memory budget on 16 GB
- Ollama 7B q4 needs about 5 GB while loaded. Set `OLLAMA_KEEP_ALIVE=5m` so it unloads when idle.
- Never run LLM batch scoring and LightGBM training at the same time. The scheduler enforces a `heavy_job` mutex.
- Intraday tick handling must not depend on the LLM (no LLM in the hot path).

---

## 5. Repository layout

```
tradepartner/
├── pyproject.toml
├── README.md
├── .env.example                 # non-secret settings only
├── config/
│   ├── app.yaml                 # paths, schedule, universe, LLM model
│   ├── risk.yaml                # ALL risk limits (section 7)
│   ├── costs.yaml               # Zerodha charge schedule (section 8)
│   ├── tax.yaml                 # tax rates/treatment (section 2.4)
│   └── strategies/
│       ├── swing_v1.yaml
│       ├── positional_v1.yaml
│       └── intraday_v1.yaml
├── src/tradepartner/
│   ├── __init__.py
│   ├── cli.py                   # Typer CLI: login, run, kill, mode, backtest, train, sync, report
│   ├── settings.py              # pydantic models for all config files
│   ├── calendar.py              # NSE trading days, holidays, session times (IST)
│   ├── db/
│   │   ├── models.py            # SQLAlchemy tables (section 6)
│   │   ├── session.py
│   │   └── migrations/          # Alembic
│   ├── data/
│   │   ├── instruments.py       # Kite instrument master, symbol↔token mapping
│   │   ├── candles.py           # historical download, Parquet store, gap detection
│   │   ├── corporate_actions.py # splits/bonus/dividends, price adjustment
│   │   ├── universe.py          # universe build + filters
│   │   ├── live.py              # KiteTicker websocket (intraday, M9)
│   │   └── quality.py           # data validation checks
│   ├── news/
│   │   ├── sources/             # one module per source (rss.py, nse.py, bse.py, gdelt.py)
│   │   ├── ingest.py            # fetch, dedupe, store raw
│   │   ├── entity_link.py       # map article → symbols
│   │   └── analyst.py           # LLM scoring + explanations
│   ├── llm/
│   │   ├── client.py            # Ollama wrapper, retries, timeouts, JSON-schema output
│   │   ├── prompts/             # versioned prompt templates (.md)
│   │   └── schemas.py           # pydantic output schemas
│   ├── features/
│   │   ├── price.py             # technical features
│   │   ├── news.py              # aggregated LLM features
│   │   ├── market.py            # index/breadth/regime features
│   │   ├── labels.py            # targets
│   │   └── build.py             # point-in-time feature matrix
│   ├── models/
│   │   ├── train.py             # walk-forward training
│   │   ├── cv.py                # purged/embargoed splits
│   │   ├── evaluate.py          # metrics + promotion gate
│   │   └── registry.py          # model versions, status, artifacts
│   ├── strategies/
│   │   ├── base.py              # Strategy protocol
│   │   ├── baseline.py          # simple rule baselines
│   │   ├── swing.py
│   │   ├── positional.py
│   │   └── intraday.py
│   ├── signals/
│   │   ├── generate.py          # run strategies → Signal rows
│   │   ├── explain.py           # LLM explanation per signal
│   │   └── proposals.py         # Signal → OrderProposal (sizing, prices, stops)
│   ├── risk/
│   │   ├── rules.py             # one function per rule (section 7.2)
│   │   ├── engine.py            # RiskEngine.check() → Approved/Rejected
│   │   ├── limits.py            # daily/weekly/drawdown trackers
│   │   └── kill.py              # kill switch
│   ├── modes/
│   │   └── controller.py        # OFF/MANUAL/AUTO state machine (section 7.1)
│   ├── execution/
│   │   ├── broker.py            # Broker protocol
│   │   ├── paper.py             # PaperBroker
│   │   ├── kite.py              # KiteBroker
│   │   ├── router.py            # the ONLY caller of place_order
│   │   └── protective.py        # GTT stop-loss/targets management
│   ├── costs/
│   │   └── zerodha.py           # charge computation (section 8)
│   ├── ledger/
│   │   ├── sync.py              # pull orders/trades/positions/holdings from Kite
│   │   ├── importers.py         # Console tradebook / Tax P&L CSV import
│   │   ├── lots.py              # FIFO lot matching
│   │   ├── tax.py               # classification + FY reports
│   │   └── reconcile.py         # compare ledger vs broker
│   ├── backtest/
│   │   ├── engine.py            # event-driven daily & intraday backtester
│   │   ├── fills.py             # fill/slippage models
│   │   └── metrics.py
│   ├── reports/
│   │   ├── daily.py
│   │   └── tax_export.py        # CSV/XLSX for CA
│   ├── notify.py
│   ├── scheduler.py             # job definitions (section 12)
│   └── ui/
│       ├── app.py               # Streamlit entry
│       └── pages/               # section 11
├── tests/
│   ├── unit/
│   ├── integration/             # with mocked Kite, temp SQLite
│   └── fixtures/                # sample candles, contract notes, news
└── ops/
    ├── com.tradepartner.plist   # launchd
    └── backup.sh                # nightly SQLite + config backup
```

Runtime data lives outside the repo: `~/.tradepartner/{db.sqlite, parquet/, models/, logs/, KILL}`.

---

## 6. Data model (SQLite)

All timestamps are stored as **UTC ISO-8601** and displayed in **IST**. Money is stored as **integer paise** (`INTEGER`), never float. Quantities are integers. Every table has `created_at`. Tables marked *append-only* must never get `UPDATE`/`DELETE` statements in application code. Corrections are new rows that reference the original.

### 6.1 Reference
- **`instruments`**: `instrument_token PK, exchange, tradingsymbol, isin, name, segment, lot_size, tick_size, last_updated`. Refreshed daily. Only `segment = 'NSE'` equity is allowed into the universe.
- **`universe_members`**: `id, universe_name, symbol, valid_from, valid_to`. Historical membership (for example Nifty 100), to limit survivorship bias.
- **`corporate_actions`**: `id, symbol, isin, action_type (split|bonus|dividend|rights|merger), ex_date, ratio_num, ratio_den, amount_paise, source`.
- **`trading_calendar`**: `date PK, is_trading_day, session_open, session_close, note`.

### 6.2 News and LLM
- **`news_raw`** (append-only): `id, source, url UNIQUE, title, body, published_at, fetched_at, content_hash UNIQUE, lang`.
- **`news_entities`**: `news_id, symbol, match_method (alias|isin|llm), confidence`.
- **`news_scores`** (append-only): `id, news_id, symbol, llm_model, llm_model_digest, prompt_version, relevance (0..1), sentiment (-1..1), event_type, materiality (0..1), horizon (intraday|days|weeks|months), confidence (0..1), summary, raw_json, scored_at`.

### 6.3 Features, models and signals
- **`feature_snapshots`**: `id, as_of (timestamp of info cutoff), symbol, feature_set_version, parquet_path`. Pointer only; the matrices themselves go to Parquet.
- **`models`**: `id, strategy, version, algo, feature_set_version, train_start, train_end, params_json, metrics_json, artifact_path, status (candidate|shadow|approved|retired), approved_by, approved_at, notes`.
- **`signals`** (append-only): `id, strategy, model_id, symbol, side (BUY|SELL), as_of, score, prob, expected_return, horizon_days, explanation, features_ref, status (new|proposed|rejected_by_risk|shown|approved|executed|expired|ignored)`.
- **`order_proposals`**: `id, signal_id, symbol, side, product (CNC|MIS), order_type (LIMIT|SL|SL-M|MARKET), qty, limit_price_paise, trigger_price_paise, stop_loss_paise, target_paise, time_stop_date, risk_amount_paise, mode_at_creation, risk_decision (approved|rejected), risk_reasons_json, idempotency_key UNIQUE`.

### 6.4 Execution and ledger (source of truth for money)
- **`orders`** (append-only status history in `order_events`): `id, proposal_id NULL, broker_order_id, origin (bot_auto|bot_manual_api|manual_external|paper), symbol, side, product, order_type, qty, price_paise, trigger_paise, tag, placed_at, status`.
- **`order_events`** (append-only): `id, order_id, status, filled_qty, avg_price_paise, message, at`.
- **`fills`** (append-only, **the core ledger**): `id, broker_trade_id UNIQUE, broker_order_id, order_id NULL, origin, account, symbol, isin, exchange, side, product, qty, price_paise, filled_at, trade_date, source (kite_api|console_csv|paper)`.
- **`charges`** (append-only): `id, fill_id NULL, trade_date, symbol, component (brokerage|stt|exchange_txn|sebi|stamp|gst|dp|other), amount_paise, computed (bool), source (computed|contract_note)`.
- **`lots`** (derived, rebuildable): `id, symbol, isin, open_fill_id, open_date, qty_open, qty_remaining, cost_paise_per_share, adjusted_for_ca (bool)`.
- **`realized`** (derived, rebuildable): `id, symbol, open_fill_id, close_fill_id, qty, buy_value_paise, sell_value_paise, charges_paise (excl STT), stt_paise, holding_days, category (intraday_spec|stcg|ltcg|business), fy, pnl_paise`.
- **`positions_snapshot`**: daily copy of broker positions and holdings for reconciliation.
- **`cash_ledger`** (append-only): `id, date, type (funds_in|funds_out|charge|dividend|pnl_settlement|adjustment), amount_paise, ref`.

### 6.5 Control and audit
- **`mode_state`**: `scope (global|strategy:<name>), mode (OFF|MANUAL|AUTO), auto_unlocked (bool), changed_by (user|system), reason, changed_at`. The latest row per scope is current.
- **`risk_events`** (append-only): `id, at, rule, severity (info|block|halt), scope, details_json`.
- **`audit_log`** (append-only): `id, at, actor (user|system|scheduler), action, payload_json, payload_hash`.
- **`job_runs`**: `id, job, started_at, ended_at, status, error`.

---

## 7. Modes and risk engine (most important section)

### 7.1 Mode controller

**States:** `OFF`, `MANUAL`, `AUTO`. A global state exists plus a state per strategy. **Effective mode = the more restrictive of global and strategy** (OFF < MANUAL < AUTO).

| Mode | Behaviour |
|---|---|
| OFF | Pipelines run, signals are generated and logged, nothing is shown as actionable, no orders go out. Protective exits for existing positions **still run** (see the halt rule below). |
| MANUAL | Proposals pass through the risk engine and the approved ones are shown in the UI with an explanation. Execution depends on `manual_execution`: `kite_app` (default: Pranav places the order himself in Kite, and the system later matches the resulting fill to the proposal) or `api_approve` (a button sends it via the API; needs a static IP). |
| AUTO | Risk-approved proposals are sent via the API automatically, within the AUTO-only limits. |

**Transitions and rules (all must be unit-tested):**
1. On process start, set every scope's mode to `min(persisted_mode, MANUAL)`. **The system never starts in AUTO.** Pranav re-enables AUTO each session (one click), and this is logged.
2. AUTO can be set for a strategy only when `auto_unlocked = true` for it. Unlocking requires all of:
   - an approved model (`status = approved`)
   - at least **40 closed paper or manual trades** and **at least 2 calendar months** of tracking for that strategy
   - the realized paper/manual metrics passing the gate in section 10.5
   - Pranav clicking "Unlock AUTO" and typing the strategy name to confirm.
3. Any **halt** risk event (section 7.2) forces global mode to OFF immediately. Only Pranav can raise it again; the system never does this automatically.
4. The kill switch (CLI `tradepartner kill`, a UI button, or creating the file `~/.tradepartner/KILL`):
   - global mode → OFF
   - cancel all open **entry** orders placed by the bot
   - square off bot-created **MIS** positions with limit orders at LTP ± market protection (falling back to market orders with protection after 60 s)
   - leave CNC holdings and their GTT stop-losses in place
   - notify Pranav.
5. The AI and strategy code have **no import path** to `modes.controller.set_mode`. Only `cli.py`, `ui/` and `risk/engine.py` (for halts) may call it. Enforce this with an import-linter rule in CI.
6. Every transition is written to `mode_state` and `audit_log`.

### 7.2 Risk rules (`config/risk.yaml`)

`capital_base_paise` is set by Pranav. All percentages apply to it unless stated otherwise. These are the defaults. **Tightening is always allowed; loosening requires editing the YAML by hand, and the UI is read-only for these values.**

```yaml
capital_base_inr: 0          # REQUIRED: Pranav sets. System refuses to start if 0.
allowed_segments: [NSE_EQ]   # anything else rejected
allowed_products: [CNC, MIS]
allow_intraday_short: false
allow_market_orders: false   # except kill-switch exits
market_protection_pct: 1.0   # used whenever a market order is unavoidable

per_trade:
  max_risk_pct: 0.75         # (entry - stop) * qty <= 0.75% of capital
  max_position_pct: 10       # notional per symbol
  require_stop_loss: true
  max_stop_distance_pct: 8   # stops wider than this are rejected
  min_reward_risk: 1.5
  limit_price_band_pct: 1.5  # limit price must be within ±1.5% of LTP at submit
  min_price_inr: 50          # avoid penny stocks
  min_avg_daily_value_inr: 50_000_000   # 20-day avg traded value ≥ ₹5 cr
  max_pct_of_adv: 0.5        # order notional ≤ 0.5% of avg daily value

portfolio:
  max_open_positions: 8
  max_sector_exposure_pct: 30
  max_gross_exposure_pct: 100   # no leverage on CNC
  max_mis_exposure_pct: 30
  no_averaging_down: true
  reentry_cooldown_days_after_stop: 3

loss_limits:
  daily_loss_pct: 2.0          # realized + unrealized → HALT (global OFF for the day)
  weekly_loss_pct: 4.0         # → HALT until Pranav resets
  max_drawdown_pct: 10.0       # from equity peak → HALT until Pranav resets
  consecutive_losses_halt: 6   # per strategy → strategy OFF

rate_limits:
  max_orders_per_second: 1
  max_orders_per_day: 30
  max_order_modifications_per_order: 5

time_rules:
  no_new_entries_before: "09:20"
  no_new_entries_after: "15:00"
  mis_square_off_at: "15:10"   # before broker auto square-off
  skip_days: [result_day_of_symbol, ex_date_of_symbol]  # no entries on those symbols

data_rules:
  max_quote_staleness_sec: 10
  max_signal_age_minutes: 120
  block_if_circuit_hit: true
  block_asm_gsm: true          # exchange surveillance lists

auto_only:                     # extra limits applied only in AUTO
  max_orders_per_day: 10
  max_position_pct: 5
  daily_loss_pct: 1.0
```

**Implementation requirements**
- `RiskEngine.check(proposal, context) -> RiskDecision` is a **pure function** of `(proposal, context)`. `context` holds positions, today's P&L, quotes, limits state, mode, calendar and instrument metadata. This makes it fully testable.
- Each rule lives in `risk/rules.py` as `def rule_x(proposal, ctx, cfg) -> RuleResult`. The engine runs **all** rules and collects every failure; it does not stop at the first one. The decision is approved only if every rule passes.
- Rules return `severity`: `block` (reject this order) or `halt` (reject and trigger global OFF).
- Quantity is **computed by the risk engine**, not the strategy: `qty = floor(min(risk_budget / (entry - stop), max_position_notional / entry, adv_cap / entry))`. If `qty < 1`, reject.
- There is an idempotency key per proposal. Duplicate submissions are rejected.
- Loss-limit trackers are recomputed from the `fills` table and live quotes every 30 s while the market is open. They **include manual trades** in the same account, because losses are losses regardless of who placed the trade.
- Every decision is written to `order_proposals.risk_decision` and `risk_reasons_json`, and every halt to `risk_events`.

### 7.3 Protective orders (laptop-off safety)
A MacBook can sleep, lose Wi-Fi or crash, so **protection must live at the broker, not on the laptop**:
- For every CNC entry fill created by the bot, immediately place a **GTT OCO** (stop-loss + target) at the broker. On partial fills, resize the GTT. Verify that the GTT exists every 5 minutes, recreate it if it is missing, and alert.
- For MIS entries, place an exchange **SL** order right after the fill.
- On startup, reconcile: every open bot position must have a live protective order, or the system alerts and places one (in MANUAL mode, it alerts only).
- In MANUAL `kite_app` mode, the UI shows the exact GTT to set alongside the entry, and the daily check alerts if a matched position has no GTT.

---

## 8. Cost model (`config/costs.yaml`) **[VERIFY against Zerodha's charges page and your contract notes]**

```yaml
equity_delivery:
  brokerage: 0
  stt_buy_pct: 0.1
  stt_sell_pct: 0.1
  exchange_txn_pct_nse: 0.00297
  sebi_per_crore_inr: 10
  stamp_buy_pct: 0.015
  gst_pct: 18                  # on brokerage + exchange txn + SEBI fees
  dp_charge_per_scrip_sell_day_inr: 15.34   # incl GST; [VERIFY]
equity_intraday:
  brokerage_pct: 0.03
  brokerage_cap_inr: 20        # per executed order
  stt_sell_pct: 0.025
  exchange_txn_pct_nse: 0.00297
  sebi_per_crore_inr: 10
  stamp_buy_pct: 0.003
  gst_pct: 18
slippage:
  default_bps: 10              # backtests; per-liquidity bucket overrides allowed
  intraday_bps: 5
```

- `costs.zerodha.compute(fills_for_order) -> list[ChargeLine]`. Round per Zerodha's method (STT rounded per contract note, others to paise). **[VERIFY]** the rounding against 3+ real contract notes.
- **Acceptance test:** for every fixture contract note in `tests/fixtures/contract_notes/`, computed total charges must be within ±₹1 of the note's total.
- When a contract note is imported, its charges supersede the computed ones (`source = contract_note`).

---

## 9. Ledger, sync and tax engine

The ledger is useful even before any AI exists, so it is built early (M2).

### 9.1 Principles
- `fills` is append-only and the single source of truth. `lots` and `realized` are **derived** and can be rebuilt at any time with `tradepartner ledger rebuild`. The rebuild must be deterministic.
- All trades in the account are recorded, including manual trades Pranav placed for reasons unrelated to this system (`origin = manual_external`).

### 9.2 Sync from Kite (daily at 15:45 IST, and on demand)
1. `kite.orders()` and `kite.trades()` return today only. Upsert into `orders`/`order_events` and insert new `fills`, deduplicated on `broker_trade_id`.
2. Assign `origin`: match `tag` (bot orders carry `tag = "tp-<proposal_id>"`, max 20 chars **[VERIFY]**). Untagged orders are `manual_external`. In MANUAL `kite_app` mode, try to match untagged fills to today's shown proposals (same symbol, side and product, same qty, price within the band) and set `proposal_id`, keeping `origin = manual_external` with a link.
3. Snapshot `kite.positions()` and `kite.holdings()` into `positions_snapshot`.
4. Run reconciliation (section 9.5).

### 9.3 Importers (history and authoritative data)
- **Console tradebook CSV** (Reports → Tradebook): backfill all historical fills. Map the columns explicitly and fail loudly on unknown columns.
- **Console Tax P&L XLSX**: used only for reconciliation, never as an input to lots.
- **Contract note PDFs/CSVs** (optional, M2.5): extract charges.
- Importers are idempotent: running them twice adds no rows.

### 9.4 Lot matching
- FIFO per `(account, isin)` for delivery. Intraday is determined per trade date: Zerodha nets same-day buys and sells in the same scrip as intraday, even under CNC. Implement the same rule. **[VERIFY]** Zerodha's exact intraday netting rule.
- Corporate actions: on the ex-date of a split or bonus, adjust `qty_remaining` and `cost_paise_per_share` of open lots (bonus lots get cost 0 and a new open date at allotment).
- If the ledger finds a sell with no matching buy (missing history), create an `unmatched` realized row and raise an alert. Never invent a cost basis.

### 9.5 Reconciliation
- Daily: ledger-derived positions vs `kite.positions()` / `holdings()` per symbol (qty and average price within ±0.5%).
- Monthly or on demand: ledger realized P&L per category vs the Console Tax P&L. Show any differences in the UI.
- Any mismatch → `risk_events` with severity `block`, which **blocks AUTO entries** until resolved or acknowledged.

### 9.6 Tax reports
- `tradepartner report tax --fy 2026-27` produces a CSV and XLSX with sheets for intraday speculative, STCG, LTCG and charges by component, plus a quarterly realized summary for advance tax, an LTCG exemption utilization line, and loss set-off/carry-forward suggestions. Each suggestion is labelled "for CA review".

---

## 10. AI layer

### 10.1 News ingestion
**Sources** (each is a module implementing `fetch(since) -> list[RawArticle]`):
- NSE corporate announcements and board-meeting/results calendar (official exchange feeds; respect the terms of use and polite rate limits)
- BSE corporate announcements
- RSS feeds from major Indian financial news outlets (configurable list in `app.yaml`)
- **GDELT** for historical backfill (needed because RSS only gives recent items).

**Rules:**
- Store `published_at` from the source. If it is missing, use `fetched_at` and set `published_at_estimated = true`. **Features may only use an article once `max(published_at, fetched_at) <= as_of`.** In backtests, use `published_at` plus a configurable delay (default 15 min).
- Deduplicate on URL and on a hash of the normalized title and body. Cluster near-duplicates (same story from several outlets) by title similarity of at least 0.85 within 24 h, and keep the earliest.
- **Entity linking:** build an alias table from instrument names, common short names and ISINs. Match against the title first, then the body. If the match is ambiguous, ask the LLM to pick from candidate symbols (it must output a symbol from the candidate list, or none).
- **Start collecting news on day one (M5 can be pulled forward).** Every day not collected is history you cannot get back cheaply.

### 10.2 LLM analyst
- **Default model:** `qwen2.5:7b-instruct-q4_K_M` via Ollama. The model name and digest are stored with every score.
- **Prompt file:** `llm/prompts/news_score_v1.md`. Prompts are versioned, and changing one bumps `prompt_version`.
- **Output schema** (validated with pydantic; one retry on invalid output, then discard and log):

```json
{
  "symbol": "INFY",
  "relevant": true,
  "relevance": 0.0,
  "event_type": "earnings|guidance|order_win|management_change|regulatory|litigation|m&a|capital_raise|rating_change|macro|other",
  "sentiment": 0.0,
  "materiality": 0.0,
  "expected_horizon": "intraday|days|weeks|months",
  "confidence": 0.0,
  "summary": "≤ 40 words, factual, no advice"
}
```

- **Temperature 0**, fixed seed, `num_ctx` 4096. Truncate articles to about 2,500 tokens (title + lede + first paragraphs).
- **Throughput budget:** about 300 to 600 articles per day. Batch-score them in the 08:00 to 09:00 window and after close. If the backlog grows, prioritize universe symbols and exchange announcements.
- **Explanations:** for each shown proposal, the LLM gets the top feature contributions (SHAP values from LightGBM), recent scored news and price context, and writes 3 to 5 sentences explaining *why the model likes it and what would make it wrong*. It must not invent numbers. Every number in the explanation must come from the provided context; a post-check rejects explanations containing numbers not present in the input.

### 10.3 Lookahead and hindsight leakage (strict rules)
1. **Point-in-time features:** the feature row for `(symbol, as_of)` uses only data with timestamp ≤ `as_of`. Daily features computed at the close are usable from the **next session's open**.
2. **Labels** start strictly after `as_of`: the entry is the next open, not the same close.
3. **LLM knowledge cutoff leakage:** the LLM was trained on internet data up to its cutoff, so it may "know" how older news played out. Config value `llm.knowledge_cutoff` (for example `2024-12-31` for the default model; **[VERIFY]**) plus `llm.cutoff_buffer_days: 90`. **News features dated before `cutoff + buffer` must not be used for training evaluation or backtest metrics.** For those dates, either set them to NaN, or train a price-only model for that period. The evaluation report must state the period in which LLM features were actually tested.
4. **Survivorship:** build universes from `universe_members` as of each date. If historical membership isn't available yet, use the current list but print a warning banner in every backtest report.
5. **Corporate actions:** use adjusted prices for features and labels, and raw prices for order prices.
6. A unit test suite (`tests/unit/test_leakage.py`) feeds synthetic data containing future-dated items and asserts they never appear in features.

### 10.4 Features (v1)
For each symbol and trading day:
- **Price/volume:** returns over 1, 5, 10, 20 and 60 days; volatility (20d std, ATR14 / close); distance from 20/50/200-day SMAs; RSI14; 52-week high/low distance; volume z-score (20d); gap %; close location in the day's range; turnover rank.
- **Relative:** returns relative to Nifty 50 and to the sector index; sector-relative momentum rank.
- **Market regime:** Nifty 50 distance from its 200-day SMA, India VIX level and 5-day change, market breadth (% of universe above the 50-day SMA).
- **News (from `news_scores`, point-in-time):** count of relevant articles in 1/3/7 days; mean and max-abs sentiment weighted by relevance × materiality; days since the last material event; event-type one-hot for the last 7 days; "results in next N days" flag from the calendar.
- **Calendar:** day of week, days to month end, days to the next results date.
- All features are versioned as `feature_set_version`. Adding a feature creates a new version and does not mutate the old one.

### 10.5 Targets, models, validation and the promotion gate
| Strategy | Target | Model |
|---|---|---|
| swing_v1 | `y = 1` if the 5-day forward return from the next open beats Nifty 50 by more than (round-trip cost + 0.5%), else 0 | LightGBM binary |
| positional_v1 | rank of 60-day forward excess return (cross-sectional) | LightGBM LambdaRank, or regression on the rank |
| intraday_v1 (M9) | sign of the next-30-minute return after costs, 5-minute bars | LightGBM binary, rule-filtered |

**Validation:** walk-forward with expanding windows. The train window is everything up to T. Then comes an **embargo equal to the label horizon**, then a 3-month test window, rolling forward. Purge training rows whose label windows overlap the test period. Hyperparameter search (optuna, at most 50 trials) runs **inside** the training folds only.

**Baselines** (must be implemented, M4):
- Swing: buy-and-hold Nifty 50, plus a 20-day momentum top-N rule with the same risk rules.
- Positional: equal-weight universe, plus a 12-1 month momentum top-N rule.

**Backtest metrics:** CAGR, annualized volatility, Sharpe, Sortino, max drawdown, Calmar, hit rate, average win/loss, turnover, cost drag (as % of gross P&L), exposure, number of trades. Report them per year, per regime (Nifty above or below its 200-day SMA), and with and without news features.

**Promotion gate** (`models/evaluate.py`, all must pass for `candidate → approved`):
1. Out-of-sample Sharpe after costs ≥ baseline Sharpe + 0.3.
2. Max drawdown ≤ 1.2 × the baseline's max drawdown.
3. At least 100 out-of-sample trades.
4. Positive after-cost return in at least 60% of test windows.
5. Performance with slippage doubled is still positive.
6. Feature importance shows no single feature with more than 40% gain share (a common symptom of leakage).
7. Pranav reviews the report and clicks Approve. **The system never self-approves.**

### 10.6 Learning loop ("keeps on learning", safely)
- **Weekly (Saturday):** retrain each strategy's model on all data up to Friday and run the full walk-forward evaluation. The result is a new `candidate`.
- The candidate runs in **shadow mode** for at least 2 weeks: it generates signals that are logged but not shown. Its live shadow performance is compared to the current approved model.
- If both the gate and the shadow comparison pass, the UI asks Pranav to promote it. The old model is kept for instant rollback.
- **Drift monitors:** daily population-stability index on the top 10 features, and calibration (Brier score) of predictions vs realized outcomes over a rolling 60 trades. If drift exceeds the threshold, raise an alert and downgrade the strategy to MANUAL.
- Revealed outcomes ("what actually happened") are written to `signals` once the horizon passes, giving a growing labeled history for the next retrain. This is the "hide, then reveal" loop Pranav described, implemented as batch retraining.

---

## 11. User interface (Streamlit, `127.0.0.1:8501`)

**Pages:**
1. **Today:** session status (logged in? data fresh? news scored?), global mode badge, and a kill switch button (red, with confirmation). It shows a table of proposals with symbol, side, entry/stop/target, qty, risk ₹, model probability, LLM explanation (expandable), news headlines and risk decision. In MANUAL `kite_app` mode each row has a **"Copy order for Kite"** button that shows exact order and GTT values, plus "Ignore". In `api_approve` mode it has "Approve" and "Reject".
2. **Modes:** global and per-strategy mode selectors (AUTO greyed out unless unlocked), unlock flow, and mode change history.
3. **Positions:** live positions and holdings with ledger lots, origin, protective-order status (green tick or red alert), and unrealized P&L.
4. **Ledger:** searchable fills with filters by origin, product and date. There is no edit, only an "add correction" flow that creates a reversing and a new row.
5. **P&L and Tax:** FY selector, realized P&L by category, charges by component, quarterly advance-tax helper, reconciliation status vs Console, and export buttons.
6. **Strategies and Models:** each model's status, walk-forward report, gate checklist, shadow comparison, drift charts, Approve/Rollback buttons.
7. **Risk:** current limit usage (gauges for daily loss, drawdown, orders today), risk events log, and the halt reset button (requires typing "RESET").
8. **System:** job runs, errors, Ollama status, disk usage, last backup.

Values in `risk.yaml` are **read-only** in the UI.

---

## 12. Daily schedule (IST, trading days only)

| Time | Job |
|---|---|
| 07:45 | Health check (disk, DB, Ollama, internet). Prompt Pranav to log in to Kite (notification with login URL). |
| On login | Store the access token in Keychain (expires the next morning). |
| 08:00 | Refresh instruments, corporate actions and the trading calendar. Rebuild today's universe. |
| 08:00–08:50 | Fetch overnight news, then LLM scoring (heavy_job lock). |
| 08:50 | Build features as of the previous close plus news up to now. Run the swing and positional strategies to get signals, then proposals and risk checks. Generate LLM explanations. |
| 09:00 | Notify: "N proposals ready". |
| 09:15 | Market open. Start the live quote subscription for proposal and position symbols. |
| 09:20 | AUTO: submit approved entry limit orders. MANUAL: proposals become actionable. |
| Every 30 s | Loss-limit tracker, kill-file check, protective-order check (every 5 min). |
| Every 15 min | Incremental news fetch; scoring of exchange announcements for held and proposal symbols (light). |
| 15:00 | Stop new entries. Expire unfilled entry orders. |
| 15:10 | Square off MIS positions. |
| 15:45 | Sync orders, trades, positions and holdings. Reconcile. Write `signals` outcomes for matured horizons. |
| 16:30 | Download EOD candles and update Parquet. Data quality checks. |
| 17:00 | Daily report (notification + UI). Nightly backup. |
| Sat 10:00 | Weekly retrain to get candidates; evaluation report; shadow start. |

Run the scheduler under `launchd` with `caffeinate -i` during market hours. If the laptop was asleep and a job was missed, run idempotent catch-up on wake, but **never** submit stale entry orders. The `max_signal_age_minutes` rule enforces this.

---

## 13. Backtester requirements
- Event-driven, bar by bar. Daily bars for swing and positional; 5-minute bars for intraday.
- Signals at bar T fill at the **next bar's open** ± slippage. Limit orders fill only if the next bar's range touches the limit.
- Stop and target within the same bar: assume the **stop is hit first** (conservative).
- Apply the full cost model (section 8) and **the same RiskEngine code** as live trading (no reimplementation). The backtester builds `context` from simulated state.
- Respect circuit limits: no fill when the bar's open equals its high (upper circuit) for a buy, or its low for a sell.
- Output: trade list, equity curve, metrics (section 10.5), and an HTML report saved under `~/.tradepartner/reports/`.
- Deterministic given a seed. Regression test: a fixed fixture dataset produces an identical trade list across runs.

---

## 14. Security and operations
- Kite API key, secret and daily token are kept in macOS Keychain. Telegram token is optional, also in Keychain. Never log secrets: add a structlog processor that redacts known keys.
- The UI binds to localhost only. No remote access in v1.
- Nightly backup: copy the SQLite file via the `sqlite3 .backup` command, plus `config/`, to `~/.tradepartner/backups/` and optionally an encrypted external drive. Keep 30 days.
- The `ops/` folder documents the static IP setup for when AUTO is first enabled (home ISP static IP, or a VPS relay). **[VERIFY]** Zerodha's current IP registration process.
- If the token is missing or expired at 09:10, notify and keep global mode in OFF or MANUAL. Never try headless login.

---

## 15. Testing strategy
- **Unit tests:** every risk rule (pass, fail and boundary cases), mode transitions (including "never starts in AUTO" and "halt forces OFF"), cost computation vs contract-note fixtures, lot matching (FIFO, intraday netting, split, bonus, missing history), tax categorization (day 365 vs 366 boundaries, FY boundary 31 Mar / 1 Apr), feature leakage, and the LLM schema validator.
- **Property tests** (hypothesis): the risk engine never approves qty < 1, notional above the cap, or a missing stop; ledger rebuild is idempotent.
- **Integration tests:** a full simulated trading day with `PaperBroker` and a mocked Kite client (data and order responses replayed from fixtures), from proposals through fills, ledger, report and reconciliation.
- **Structural test:** import-linter rules (no strategy, model or LLM code imports `execution.router` or `modes.controller.set_mode`).
- CI command: `uv run ruff check && uv run mypy && uv run pytest --cov`. Coverage targets: 100% branch for `risk`, `modes`, `costs`, `ledger`; 80% overall.

---

## 16. Milestones (build in order)

Each milestone ends with its acceptance criteria passing and a short demo note in `docs/milestones/Mx.md`.

### M0: Scaffold
- `uv` project, layout from section 5, config loading with pydantic validation, structlog, Typer CLI skeleton, SQLite + Alembic with the empty schema from section 6, pre-commit (ruff, mypy), and a pytest skeleton.
- **Accept:** `tradepartner --help` works; `uv run pytest` passes; migrations create all tables; the app refuses to start with `capital_base_inr: 0`.

### M1: Market data
- Kite login flow (local redirect server on `127.0.0.1`, token stored in Keychain), instrument master, NSE trading calendar, historical daily candles for the universe (start with Nifty 100) going back 10 years, Parquet storage, gap detection, corporate-action adjustment, data quality report.
- **Accept:** `tradepartner data backfill --universe nifty100 --years 10` completes and can resume; quality checks pass on fixtures; adjusted prices are continuous across a known split (fixture).

### M2: Ledger, costs and tax (usable on its own)
- Kite sync, Console tradebook CSV importer, cost model, FIFO lots, intraday netting, corporate-action adjustment of lots, realized P&L classification, reconciliation, tax report export, UI pages Ledger and P&L/Tax.
- **Accept:** importing Pranav's own historical tradebook gives realized P&L per category matching Console Tax P&L within ±₹10 per FY (or the differences are listed and explained); contract-note fixture tests pass within ±₹1.

### M3: Risk engine, modes, paper broker
- Everything in section 7 except the Kite live execution path. Includes `PaperBroker` (fills against live or replayed quotes), the kill switch, loss trackers, risk events and the Risk and Modes UI pages.
- **Accept:** all risk and mode unit and property tests pass; a simulated day that breaches the daily loss limit halts and forces OFF; restart never resumes AUTO.

### M4: Backtester and baselines
- Section 13 plus the baselines from section 10.5.
- **Accept:** a deterministic regression test passes; the baseline report renders; costs and slippage appear in the report.

### M5: News and LLM analyst (start news collection as early as possible)
- Sources, ingestion, deduplication, entity linking, Ollama client, scoring prompt v1, schema validation, GDELT backfill job, news features.
- **Accept:** a 7-day live collection runs unattended; at least 95% of scored outputs are schema-valid; a leakage test passes; scoring throughput is at least 300 articles per hour on the MacBook (measure and record it).

### M6: ML pipeline and model registry
- Features v1, labels, purged walk-forward CV, LightGBM training, evaluation report, promotion gate, registry statuses, shadow mode, drift monitors.
- **Accept:** a swing_v1 walk-forward report is produced with and without news features; the gate is evaluated automatically; nothing is auto-approved.

### M7: Signals, explanations, UI and MANUAL flow
- Daily pipeline from section 12 (through proposals), LLM explanations with a number check, the Today page, the "Copy order for Kite" flow, matching manual fills to proposals, notifications, launchd setup.
- **Accept:** 10 consecutive trading days running unattended in MANUAL mode with no missed jobs; every shown proposal has an explanation that passes the number check.

### M8: AUTO execution via Kite (only after Pranav decides to)
- `KiteBroker` order placement with tags, limit orders with price bands, GTT OCO protection and verification loop, AUTO-only limits, unlock flow, static IP documentation, `api_approve` mode.
- **Accept:** live (Kite has no sandbox) with **1-share orders** on a liquid stock: an entry, its GTT, a cancel, the kill switch and reconciliation all behave as specified. Pranav signs off in `docs/milestones/M8.md` before any larger size.

### M9: Intraday (last, hardest gate)
- Kite WebSocket live bars, 5-minute feature pipeline, intraday_v1 (rule-based opening-range or VWAP setup filtered by model and news), MIS SL orders, 15:10 square-off.
- **Accept:** at least 3 months of paper trading passes the gate before MANUAL live; AUTO intraday requires a separate unlock.

---

## 17. Go-live checklist (Pranav)
- [ ] `capital_base_inr` set; worst-case daily loss (2%) and max drawdown (10%) amounts written down and accepted
- [ ] Kite Connect subscription active; daily login habit in place
- [ ] Historical tradebook imported and reconciled with Console
- [ ] At least 2 months of MANUAL / paper tracking per strategy
- [ ] Model passed the gate and was approved by me
- [ ] For AUTO: static IP registered with Zerodha; M8 1-share tests done
- [ ] CA consulted on swing treatment (capital gains vs business) and ITR form
- [ ] Backups verified by actually restoring once

## 18. Open decisions
1. **Capital base:** needed for M0.
2. **Static IP route** (home ISP vs VPS) when AUTO is first enabled. Not needed before M8.
3. **Swing tax treatment:** with a CA, before the first FY close.
4. **Universe:** Nifty 100 by default; widening to Nifty 200 or 500 increases data and news load.
5. **Telegram alerts:** optional.

---

*Sources checked 2026-10-15: Zerodha support "What are the charges for Kite APIs"; Zerodha "In the Money", "SEBI Algo Trading Changes, April 2026"; post-Budget-2026 equity tax summaries (STCG 20%, LTCG 12.5% above ₹1.25 lakh, no change to equity delivery/intraday STT). Re-verify before going live.*
