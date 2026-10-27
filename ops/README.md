# Operations

- `com.tradepartner.plist`: launchd job that runs the scheduler under `caffeinate -i`.
- `backup.sh`: nightly SQLite backup plus `config/` (the scheduler runs it at 17:00).

## Static IP (needed only for API order placement: AUTO and `api_approve`)
SEBI's retail algo framework requires API orders to come from a static IP registered with the
broker (one primary, one secondary). Reading data and placing orders by hand in the Kite app do
not need it. Options: a home ISP static IP, or a small VPS relay (roughly Rs 1,500/year upward).
**[VERIFY]** Zerodha's current IP registration process in the Kite developer console before the
first AUTO session. Until then keep `manual_execution: kite_app` (the default).

## Daily routine
1. Around 07:45 a notification asks you to log in: run `tradepartner login` (always manual).
2. `tradepartner ui` opens the dashboard on 127.0.0.1:8501.
3. Restore test: `sqlite3 backup/db.sqlite ".tables"` on a copy, at least once before go-live.
