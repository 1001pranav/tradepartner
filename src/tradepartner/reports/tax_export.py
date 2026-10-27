"""Tax report (spec 9.6): CSV + XLSX for the CA. Every suggestion is labelled "for CA review"."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from tradepartner.calendar import fy_bounds
from tradepartner.db.models import Fill, Realized
from tradepartner.ledger import tax
from tradepartner.ledger.rebuild import effective_charges
from tradepartner.settings import TaxConfig

CA_LABEL = "For CA review"
QUARTERS = {"Q1 (Apr-Jun)": (4, 6), "Q2 (Jul-Sep)": (7, 9), "Q3 (Oct-Dec)": (10, 12),
            "Q4 (Jan-Mar)": (1, 3)}  # fmt: skip


def inr(paise: int) -> Decimal:
    return Decimal(paise) / 100


def realized_frame(session: Session, fy: str) -> pd.DataFrame:
    rows = []
    for r in session.scalars(
        select(Realized).where(Realized.fy == fy).order_by(Realized.close_fill_id, Realized.id)
    ):
        close = session.get(Fill, r.close_fill_id)
        assert close is not None
        rows.append(
            {
                "symbol": r.symbol,
                "category": r.category,
                "close_date": close.trade_date,
                "holding_days": r.holding_days,
                "qty": r.qty,
                "buy_value": inr(r.buy_value_paise),
                "sell_value": inr(r.sell_value_paise),
                "charges_ex_stt": inr(r.charges_paise),
                "stt": inr(r.stt_paise),
                "pnl": inr(r.pnl_paise),
                "pnl_paise": r.pnl_paise,
            }
        )
    cols = ["symbol", "category", "close_date", "holding_days", "qty", "buy_value", "sell_value",
            "charges_ex_stt", "stt", "pnl", "pnl_paise"]  # fmt: skip
    return pd.DataFrame(rows, columns=cols)


def category_totals(df: pd.DataFrame) -> dict[str, int]:
    out = {c: 0 for c in ("intraday_spec", "stcg", "ltcg", "business", "unmatched")}
    if not df.empty:
        for cat, total in df.groupby("category")["pnl_paise"].sum().items():
            out[str(cat)] = int(total)
    return out


def quarterly(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for label, (m1, m2) in QUARTERS.items():
        part = df[df["close_date"].map(lambda d, a=m1, b=m2: a <= d.month <= b)] if len(df) else df
        totals = category_totals(part)
        recs.append({"quarter": label, **{k: inr(v) for k, v in totals.items()}})
    return pd.DataFrame(recs)


def charges_by_component(session: Session, fy: str) -> pd.DataFrame:
    start, end = fy_bounds(fy)
    eff = effective_charges(session)
    totals: dict[str, int] = {}
    for f in session.scalars(select(Fill).where(Fill.trade_date >= start, Fill.trade_date <= end)):
        for comp, amount in eff.get(f.id, {}).items():
            totals[comp] = totals.get(comp, 0) + amount
    return pd.DataFrame(
        [{"component": c, "amount": inr(a)} for c, a in sorted(totals.items())],
        columns=["component", "amount"],
    )


@dataclass(frozen=True)
class TaxReport:
    fy: str
    realized: pd.DataFrame
    totals: dict[str, int]
    quarterly: pd.DataFrame
    charges: pd.DataFrame
    setoff: tax.SetOff


def build_report(session: Session, fy: str, cfg: TaxConfig) -> TaxReport:
    df = realized_frame(session, fy)
    totals = category_totals(df)
    setoff = tax.suggest_setoff(
        totals["stcg"], totals["ltcg"], totals["intraday_spec"], totals["business"], cfg
    )
    return TaxReport(fy, df, totals, quarterly(df), charges_by_component(session, fy), setoff)


def _summary_frame(rep: TaxReport, cfg: TaxConfig) -> pd.DataFrame:
    s = rep.setoff
    lines = [
        ("Intraday speculative P&L", inr(rep.totals["intraday_spec"]), ""),
        ("STCG", inr(rep.totals["stcg"]), f"{cfg.stcg_pct}%"),
        ("LTCG", inr(rep.totals["ltcg"]), f"{cfg.ltcg_pct}% above exemption"),
        ("Business income (swing)", inr(rep.totals["business"]), cfg.swing_treatment),
        ("Unmatched sells (no cost basis, EXCLUDED)", inr(rep.totals["unmatched"]), "fix history"),
        ("LTCG exemption used", inr(s.ltcg_exemption_used), f"limit INR {cfg.ltcg_exemption_inr}"),
        ("LTCG taxable after exemption", inr(s.ltcg_taxable), CA_LABEL),
        ("Capital loss carry-forward", inr(s.carry_forward_capital_loss), CA_LABEL),
        ("Speculative loss carry-forward", inr(s.carry_forward_speculative_loss), CA_LABEL),
    ] + [("Note", "", f"{n} ({CA_LABEL})") for n in s.notes]
    return pd.DataFrame(lines, columns=["item", "amount_inr", "remark"])


def export(session: Session, fy: str, cfg: TaxConfig, out_dir: Path) -> tuple[Path, Path]:
    rep = build_report(session, fy, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"tax_{fy}.csv"
    xlsx_path = out_dir / f"tax_{fy}.xlsx"
    export_cols = [c for c in rep.realized.columns if c != "pnl_paise"]
    rep.realized[export_cols].to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        _summary_frame(rep, cfg).to_excel(xw, sheet_name="Summary", index=False)
        for sheet, cat in (("Intraday speculative", "intraday_spec"), ("STCG", "stcg"),
                           ("LTCG", "ltcg"), ("Business", "business"),
                           ("Unmatched", "unmatched")):  # fmt: skip
            part = rep.realized[rep.realized["category"] == cat][export_cols]
            part.to_excel(xw, sheet_name=sheet, index=False)
        rep.charges.to_excel(xw, sheet_name="Charges", index=False)
        rep.quarterly.to_excel(xw, sheet_name="Quarterly (advance tax)", index=False)
    return csv_path, xlsx_path


def today_fy(today: date) -> str:
    return tax.fy_of(today)
