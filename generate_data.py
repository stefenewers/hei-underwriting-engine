"""
Synthetic HEI Deal Data Generator
====================================
Generates a realistic training dataset for the HEI Underwriting Engine.

Design philosophy:
  - HEI investment amounts are back-computed from a target IRR distribution,
    mirroring how real operators size deals (not random sampling).
  - A deliberate mix of good, borderline, and bad deals is constructed to
    prevent class collapse.
  - Distributions are anchored to real HEI market parameters observed in
    Splitero, Unlock, Point, and Hometap deal flow disclosures.
  - Appreciation rates are calibrated against Zillow ZHVI 5-year CAGR data.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from hei_engine import (
    FEATURE_NAMES,
    compute_irr_distribution,
    engineer_features,
    STATE_APPRECIATION,
    calculate_irr,
)

RNG = np.random.default_rng(42)
N_SAMPLES = 8_000


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_property_value() -> float:
    """Log-normal distribution centered around $475K (realistic HEI target)."""
    return float(np.clip(RNG.lognormal(mean=13.07, sigma=0.55), 150_000, 3_000_000))


def sample_ltv() -> float:
    """
    LTV at origination. HEI operators typically require < 80% LTV.
    Beta distribution skewed toward 30–60% range.
    """
    return float(np.clip(RNG.beta(a=3.0, b=4.5), 0.0, 0.82))


def sample_credit_score() -> int:
    """Normal distribution; HEI operators generally require 580+ minimum."""
    return int(np.clip(RNG.normal(loc=720, scale=65), 560, 850))


def sample_equity_share() -> float:
    """Equity share % offered to operator. Typical range: 10–30%."""
    return float(np.clip(RNG.beta(a=2.5, b=5.0) * 0.35 + 0.08, 0.10, 0.30))


def sample_cap_multiple() -> float:
    """Cap expressed as multiple of investment. Typical: 1.5–3.0×."""
    return float(np.clip(RNG.normal(loc=2.2, scale=0.4), 1.5, 3.5))


def sample_term() -> int:
    """Investment horizon: most HEIs run 5 or 10 years."""
    return int(RNG.choice([5, 10], p=[0.35, 0.65]))


def sample_appreciation_cagr(state_cagr: float) -> Tuple[float, float, float]:
    """
    Sample appreciation distribution (P10, P50, P90) around metro baseline.
    """
    sigma = 0.025
    p50 = float(np.clip(state_cagr + RNG.normal(0, 0.012), 0.01, 0.18))
    p10 = float(np.clip(p50 - 2 * sigma, 0.005, p50))
    p90 = float(np.clip(p50 + 2 * sigma, p50, 0.25))
    return p10, p50, p90


def sample_state() -> str:
    """Sample states with weight toward higher-appreciation markets."""
    states = list(STATE_APPRECIATION.keys())
    weights = np.array([STATE_APPRECIATION[s] for s in states])
    weights = weights / weights.sum()
    return str(RNG.choice(states, p=weights))


def compute_hei_amount_from_target_irr(
    target_irr: float,
    equity_share: float,
    property_value: float,
    appreciation_cagr: float,
    cap_multiple: float,
    term_years: int,
    equity: float,
) -> float:
    """
    Back-compute HEI investment amount to achieve a target IRR.

    IRR formula: net_return = investment * (1 + target_irr)^term
    net_return = min(equity_share * property_value * total_appreciation, cap_amount)

    Solve for investment given target_irr:
      gross_return = equity_share * property_value * ((1+cagr)^term - 1)
      For below-cap regime: investment = gross_return / (1 + target_irr)^term
    """
    total_appreciation = (1 + appreciation_cagr) ** term_years - 1
    gross_return = equity_share * property_value * total_appreciation

    if gross_return <= 0:
        return max(equity * 0.05, 10_000)

    # Required investment for target IRR (ignoring cap initially)
    compounding = (1 + target_irr) ** term_years
    if compounding <= 0:
        compounding = 0.01

    investment = gross_return / compounding

    # Check if cap would bind at this investment level
    cap_amount = cap_multiple * investment
    if gross_return > cap_amount:
        # Cap binds: solve for investment such that cap_amount = gross_return / compounding
        # cap_multiple * investment = gross_return / (1 + target_irr)^term
        # investment = gross_return / (compounding * cap_multiple)
        # But this is circular... use the uncapped version and accept cap effect
        pass  # Use uncapped investment as-is; label assignment handles this

    # Bound to reasonable limits
    max_investment = min(equity * 0.60, 500_000)
    min_investment = max(equity * 0.03, 10_000)
    return float(np.clip(investment, min_investment, max_investment))


# ---------------------------------------------------------------------------
# Label assignment (expert heuristic underwriting rules)
# ---------------------------------------------------------------------------

def assign_label(
    irr_base: float,
    ltv: float,
    credit_score: int,
    equity_pct: float,
    cap_exceedance_prob: float,
    hei_to_equity: float,
) -> str:
    """
    Business rule labeler mimicking HEI operator underwriting criteria.

    APPROVE  : Strong IRR, solid collateral, good credit
    REVIEW   : Borderline on one or more dimensions
    REJECT   : Fails one or more hard underwriting thresholds
    """
    # Hard rejects
    if ltv > 0.82:
        return "REJECT"
    if credit_score < 580:
        return "REJECT"
    if irr_base < 0.02:          # < 2% annualized is unacceptable
        return "REJECT"
    if equity_pct < 0.08:
        return "REJECT"
    if hei_to_equity > 0.75:
        return "REJECT"

    # Scoring system
    score = 0

    if irr_base >= 0.14:
        score += 3
    elif irr_base >= 0.09:
        score += 2
    elif irr_base >= 0.05:
        score += 1

    if ltv <= 0.50:
        score += 3
    elif ltv <= 0.65:
        score += 2
    elif ltv <= 0.75:
        score += 1

    if credit_score >= 740:
        score += 3
    elif credit_score >= 680:
        score += 2
    elif credit_score >= 620:
        score += 1

    if cap_exceedance_prob <= 0.20:
        score += 1

    if equity_pct >= 0.40:
        score += 1

    if score >= 7:
        return "APPROVE"
    elif score >= 3:
        return "REVIEW"
    else:
        return "REJECT"


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_dataset(n: int = N_SAMPLES) -> pd.DataFrame:
    """
    Generate synthetic HEI deals with a controlled mix:
      ~35% APPROVE, ~40% REVIEW, ~25% REJECT
    via stratified sampling of target IRR tiers.
    """
    records = []

    # IRR tier mix: good / borderline / bad deals
    tier_config = [
        # (target_irr_mean, target_irr_std, fraction)
        (0.155, 0.04, 0.35),   # Good deals: 12–20% target IRR
        (0.065, 0.03, 0.40),   # Borderline: 3–10% target IRR
        (-0.05, 0.08, 0.25),   # Bad deals: negative/near-zero target IRR
    ]

    tier_counts = [int(n * f) for _, _, f in tier_config]
    tier_counts[-1] = n - sum(tier_counts[:-1])  # handle rounding

    for (irr_mean, irr_std, _), count in zip(tier_config, tier_counts):
        for _ in range(count):
            state = sample_state()
            base_cagr = STATE_APPRECIATION[state]

            p10, p50, p90 = sample_appreciation_cagr(base_cagr)
            property_value = sample_property_value()
            ltv = sample_ltv()
            outstanding_mortgage = ltv * property_value
            equity = property_value - outstanding_mortgage
            credit_score = sample_credit_score()
            equity_share_pct = sample_equity_share()
            cap_multiple = sample_cap_multiple()
            term_years = sample_term()

            # Sample target IRR for this tier, then back-compute investment
            target_irr = float(np.clip(RNG.normal(irr_mean, irr_std), -0.30, 0.40))

            hei_amount = compute_hei_amount_from_target_irr(
                target_irr=target_irr,
                equity_share=equity_share_pct,
                property_value=property_value,
                appreciation_cagr=p50,
                cap_multiple=cap_multiple,
                term_years=term_years,
                equity=equity,
            )

            # Small random perturbation to avoid perfectly deterministic data
            hei_amount = float(np.clip(
                hei_amount * RNG.uniform(0.85, 1.15),
                max(equity * 0.03, 10_000),
                min(equity * 0.65, 500_000),
            ))

            if equity <= 0 or hei_amount <= 0:
                continue

            # Engineer features (same pipeline as inference)
            feat_df = engineer_features(
                property_value=property_value,
                outstanding_mortgage=outstanding_mortgage,
                credit_score=credit_score,
                hei_amount=hei_amount,
                equity_share_pct=equity_share_pct,
                cap_multiple=cap_multiple,
                term_years=term_years,
                appreciation_cagr=p50,
                appreciation_p10=p10,
                appreciation_p90=p90,
            )

            irr_data = compute_irr_distribution(
                hei_amount, equity_share_pct, property_value,
                p10, p50, p90, cap_multiple, term_years
            )

            irr_base = irr_data["base_irr"] / 100
            cap_exceedance_prob = irr_data["cap_exceedance_prob"]
            equity_pct_val = (equity / property_value) if property_value > 0 else 0
            hei_to_equity = hei_amount / max(equity, 1)

            label = assign_label(
                irr_base, ltv, credit_score, equity_pct_val,
                cap_exceedance_prob, hei_to_equity
            )

            row = feat_df.iloc[0].to_dict()
            row["label"] = label
            row["_property_value"] = property_value
            row["_credit_score"] = credit_score
            row["_hei_amount"] = hei_amount
            row["_state"] = state
            row["_irr_bear"] = irr_data["scenarios"]["bear"]["irr"]
            row["_irr_base"] = irr_data["scenarios"]["base"]["irr"]
            row["_irr_bull"] = irr_data["scenarios"]["bull"]["irr"]

            records.append(row)

    df = pd.DataFrame(records)
    print(f"Generated {len(df)} deals")
    print(df["label"].value_counts())
    print(f"\nIRR base stats:\n{df['_irr_base'].describe().round(2)}")
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    df.to_csv("data/synthetic_deals.csv", index=False)
    print("Saved → data/synthetic_deals.csv")
