"""Tax classification and FY helpers (spec 2.4). Rates and treatment come from ``config/tax.yaml``.

The engine stores *categories*, not statute section numbers.
"""

from dataclasses import dataclass
from datetime import date

from tradepartner.calendar import indian_fy, months_between_exceeds
from tradepartner.ledger.lots import RealizedRow
from tradepartner.settings import TaxConfig


def classify(row: RealizedRow, cfg: TaxConfig) -> str:
    """Category: intraday_spec | stcg | ltcg | business | unmatched."""
    if row.kind == "unmatched":
        return "unmatched"
    if row.kind == "intraday":
        return "intraday_spec"
    assert row.open_date is not None
    if months_between_exceeds(row.open_date, row.close_date, cfg.long_term_months):
        return "ltcg"
    # SPEC-QUESTION: under business_income treatment only short-term delivery trades are treated
    # as business income; long-term holdings stay LTCG. Confirm with the CA.
    return "business" if cfg.swing_treatment == "business_income" else "stcg"


def pnl_paise(category: str, sell: int, buy: int, charges_ex_stt: int, stt: int) -> int:
    """Taxable P&L. STT is deductible only for business-type income, never for capital gains."""
    if category == "unmatched":
        return 0  # cost basis unknown: never invent one
    base = sell - buy - charges_ex_stt
    if category in ("intraday_spec", "business"):
        return base - stt
    return base


def fy_of(day: date) -> str:
    return indian_fy(day)


@dataclass(frozen=True)
class SetOff:
    """Loss set-off outcome, as a suggestion 'for CA review' (spec 9.6)."""

    stcg_after: int
    ltcg_after: int
    speculative_after: int
    business_after: int
    carry_forward_capital_loss: int
    carry_forward_speculative_loss: int
    ltcg_exemption_used: int
    ltcg_taxable: int
    notes: tuple[str, ...]


def suggest_setoff(stcg: int, ltcg: int, speculative: int, business: int, cfg: TaxConfig) -> SetOff:
    """Net category totals (paise) under the loss set-off rules.

    * STCL sets off against STCG, then LTCG. LTCL sets off only against LTCG.
    * Speculative loss sets off only against speculative income.
    * Remaining capital loss carries forward 8 years, speculative loss 4 (if the return is on time).
    """
    notes: list[str] = []
    st, lt = stcg, ltcg
    if st < 0 and lt > 0:
        used = min(-st, lt)
        st += used
        lt -= used
        notes.append("Short-term loss set off against long-term gains.")
    if lt < 0 and st > 0:
        notes.append("Long-term loss cannot be set off against short-term gains.")
    capital_cf = max(0, -st) + max(0, -lt)
    spec_cf = max(0, -speculative)
    exemption = min(max(lt, 0), cfg.ltcg_exemption_inr * 100)
    taxable_lt = max(lt, 0) - exemption
    if capital_cf:
        notes.append(
            f"Capital loss carry-forward up to {cfg.capital_loss_carry_forward_years} years "
            "if the return is filed on time."
        )
    if spec_cf:
        notes.append(
            f"Speculative loss carry-forward up to {cfg.speculative_loss_carry_forward_years} "
            "years, against speculative income only."
        )
    return SetOff(
        st if st > 0 else 0,
        lt if lt > 0 else 0,
        max(speculative, 0),
        business,
        capital_cf,
        spec_cf,
        exemption,
        taxable_lt,
        tuple(notes),
    )
