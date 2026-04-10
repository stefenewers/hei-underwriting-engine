"""
Synthetic HEI Deal Data Generator — v2
=======================================
Generates a realistic 28-feature training dataset for the HEI Underwriting Engine.

New in v2:
  - CLTV with subordinate liens (HELOC, 2nd mortgage, tax/HOA liens)
  - Foreclosure, bankruptcy, and delinquency history flags
  - Property type risk, property age, owner-occupied, ARM, flood zone
  - DTI ratio and employment stability tier
  - IRR-anchored investment sizing (not random amounts)
"""

from typing import Tuple
import numpy as np
import pandas as pd
from hei_engine import (
    FEATURE_NAMES,
    STATE_APPRECIATION,
    compute_irr_distribution,
    engineer_features,
    get_flood_risk,
    get_market_liquidity,
)

RNG = np.random.default_rng(42)
N_SAMPLES = 10_000


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_property_value() -> float:
    return float(np.clip(RNG.lognormal(mean=13.07, sigma=0.55), 150_000, 3_000_000))


def sample_ltv() -> float:
    return float(np.clip(RNG.beta(a=3.0, b=4.5), 0.0, 0.82))


def sample_credit_score(tier: str) -> int:
    """Sample credit score from a tier-aware distribution."""
    if tier == "good":
        return int(np.clip(RNG.normal(735, 40), 680, 850))
    elif tier == "fair":
        return int(np.clip(RNG.normal(660, 35), 600, 720))
    else:
        return int(np.clip(RNG.normal(590, 30), 540, 650))


def sample_equity_share() -> float:
    return float(np.clip(RNG.beta(a=2.5, b=5.0) * 0.35 + 0.08, 0.10, 0.30))


def sample_cap_multiple() -> float:
    return float(np.clip(RNG.normal(loc=2.2, scale=0.4), 1.5, 3.5))


def sample_term() -> int:
    return int(RNG.choice([5, 10], p=[0.35, 0.65]))


def sample_appreciation_cagr(state_cagr: float) -> Tuple[float, float, float]:
    sigma = 0.025
    p50 = float(np.clip(state_cagr + RNG.normal(0, 0.012), 0.01, 0.18))
    p10 = float(np.clip(p50 - 2 * sigma, 0.005, p50))
    p90 = float(np.clip(p50 + 2 * sigma, p50, 0.25))
    return p10, p50, p90


def sample_state() -> str:
    states = list(STATE_APPRECIATION.keys())
    weights = np.array([STATE_APPRECIATION[s] for s in states])
    weights = weights / weights.sum()
    return str(RNG.choice(states, p=weights))


def sample_liens(equity: float, credit_tier_label: str) -> Tuple[float, float, float, float]:
    """
    Sample subordinate liens: HELOC, 2nd mortgage, tax lien, HOA lien.
    Good-credit borrowers have cleaner lien profiles.
    """
    # HELOC: ~40% of borrowers have one
    heloc_prob = 0.20 if credit_tier_label == "good" else 0.35 if credit_tier_label == "fair" else 0.15
    if RNG.random() < heloc_prob:
        heloc_balance = float(RNG.uniform(5_000, min(equity * 0.25, 100_000)))
    else:
        heloc_balance = 0.0

    # Second mortgage: ~15% of borrowers
    second_prob = 0.05 if credit_tier_label == "good" else 0.15 if credit_tier_label == "fair" else 0.25
    second_max = min(equity * 0.20, 80_000)
    if RNG.random() < second_prob and second_max > 10_000:
        second_mortgage = float(RNG.uniform(10_000, second_max))
    else:
        second_mortgage = 0.0

    # Tax lien: rare, more common in distressed profiles
    tax_prob = 0.01 if credit_tier_label == "good" else 0.04 if credit_tier_label == "fair" else 0.12
    tax_lien = float(RNG.uniform(1_000, 25_000)) if RNG.random() < tax_prob else 0.0

    # HOA lien: present when HOA exists and homeowner has arrears
    hoa_prob = 0.01 if credit_tier_label == "good" else 0.03 if credit_tier_label == "fair" else 0.08
    hoa_lien = float(RNG.uniform(500, 8_000)) if RNG.random() < hoa_prob else 0.0

    return heloc_balance, second_mortgage, tax_lien, hoa_lien


def sample_credit_history(credit_tier_label: str) -> Tuple[int, int, int]:
    """
    Sample foreclosure, bankruptcy, and delinquency flags.
    Correlated with credit tier.
    """
    if credit_tier_label == "good":
        foreclosure = int(RNG.random() < 0.005)
        bankruptcy = int(RNG.random() < 0.005)
        delinquency = int(RNG.random() < 0.03)
    elif credit_tier_label == "fair":
        foreclosure = int(RNG.random() < 0.04)
        bankruptcy = int(RNG.random() < 0.03)
        delinquency = int(RNG.random() < 0.15)
    else:
        foreclosure = int(RNG.random() < 0.20)
        bankruptcy = int(RNG.random() < 0.15)
        delinquency = int(RNG.random() < 0.40)
    return foreclosure, bankruptcy, delinquency


def sample_property_attributes() -> Tuple[int, int, int, int]:
    """
    Returns: (property_type_risk, property_age, owner_occupied, arm_flag)
    """
    # Property type: mostly SFR in HEI programs
    prop_type = int(RNG.choice([0, 1, 2, 3], p=[0.68, 0.15, 0.15, 0.02]))
    # Property age: 1-80 years (log-normal skewed toward newer homes)
    prop_age = int(np.clip(RNG.lognormal(mean=3.2, sigma=0.7), 1, 80))
    # Owner-occupied: required by most programs, ~5% slip through as investment
    owner_occ = int(RNG.random() > 0.05)
    # ARM flag: ~18% of mortgages are adjustable
    arm = int(RNG.random() < 0.18)
    return prop_type, prop_age, owner_occ, arm


def sample_homeowner_financials(credit_tier_label: str) -> Tuple[float, int]:
    """
    Returns: (dti_ratio, employment_stability_tier)
    """
    if credit_tier_label == "good":
        dti = float(np.clip(RNG.beta(a=3.0, b=5.0) * 0.55 + 0.10, 0.10, 0.55))
        emp_tier = int(RNG.choice([2, 1, 0], p=[0.78, 0.18, 0.04]))
    elif credit_tier_label == "fair":
        dti = float(np.clip(RNG.beta(a=2.5, b=3.5) * 0.60 + 0.15, 0.15, 0.65))
        emp_tier = int(RNG.choice([2, 1, 0], p=[0.60, 0.28, 0.12]))
    else:
        dti = float(np.clip(RNG.beta(a=2.0, b=2.5) * 0.65 + 0.20, 0.20, 0.75))
        emp_tier = int(RNG.choice([2, 1, 0], p=[0.40, 0.35, 0.25]))
    return dti, emp_tier


def compute_hei_amount(
    target_irr: float,
    equity_share: float,
    property_value: float,
    appreciation_cagr: float,
    cap_multiple: float,
    term_years: int,
    equity: float,
) -> float:
    total_appreciation = (1 + appreciation_cagr) ** term_years - 1
    gross_return = equity_share * property_value * total_appreciation
    if gross_return <= 0:
        return max(equity * 0.05, 10_000)
    compounding = max((1 + target_irr) ** term_years, 0.01)
    investment = gross_return / compounding
    return float(np.clip(investment, max(equity * 0.03, 10_000), min(equity * 0.65, 500_000)))


# ---------------------------------------------------------------------------
# Label assignment — v2 (expanded business rules)
# ---------------------------------------------------------------------------

def assign_label(
    irr_base: float,
    ltv: float,
    cltv: float,
    credit_score: int,
    equity_pct: float,
    cap_exceedance_prob: float,
    hei_to_equity: float,
    foreclosure_flag: int,
    bankruptcy_flag: int,
    mortgage_delinquency_flag: int,
    owner_occupied: int,
    property_type_risk: int,
    dti_ratio: float,
    subordinate_lien_count: int,
) -> str:
    # ---- Hard rejects ----
    if foreclosure_flag:
        return "REJECT"
    if bankruptcy_flag:
        return "REJECT"
    if not owner_occupied:
        return "REJECT"
    if property_type_risk >= 3:          # manufactured home
        return "REJECT"
    if cltv > 0.90:                       # combined position too risky
        return "REJECT"
    if ltv > 0.82:
        return "REJECT"
    if credit_score < 580:
        return "REJECT"
    if irr_base < 0.02:
        return "REJECT"
    if equity_pct < 0.08:
        return "REJECT"
    if hei_to_equity > 0.75:
        return "REJECT"
    if dti_ratio > 0.65:
        return "REJECT"

    # ---- Soft flags that push toward REVIEW ----
    review_flags = 0
    if mortgage_delinquency_flag:
        review_flags += 2
    if cltv > 0.80:
        review_flags += 2
    if ltv > 0.70:
        review_flags += 1
    if credit_score < 640:
        review_flags += 2
    if irr_base < 0.05:
        review_flags += 2
    if dti_ratio > 0.50:
        review_flags += 1
    if subordinate_lien_count >= 2:
        review_flags += 1
    if property_type_risk == 2:          # condo
        review_flags += 1

    # ---- Positive signals ----
    approve_score = 0
    if irr_base >= 0.14:
        approve_score += 3
    elif irr_base >= 0.09:
        approve_score += 2
    elif irr_base >= 0.05:
        approve_score += 1

    if ltv <= 0.50:
        approve_score += 3
    elif ltv <= 0.65:
        approve_score += 2
    elif ltv <= 0.75:
        approve_score += 1

    if cltv <= 0.65:
        approve_score += 2
    elif cltv <= 0.75:
        approve_score += 1

    if credit_score >= 740:
        approve_score += 3
    elif credit_score >= 680:
        approve_score += 2
    elif credit_score >= 620:
        approve_score += 1

    if cap_exceedance_prob <= 0.20:
        approve_score += 1
    if equity_pct >= 0.40:
        approve_score += 1
    if dti_ratio <= 0.36:
        approve_score += 1
    if subordinate_lien_count == 0:
        approve_score += 1

    # Decision
    if review_flags >= 3:
        return "REVIEW" if approve_score >= 3 else "REJECT"
    if approve_score >= 8:
        return "APPROVE"
    elif approve_score >= 3:
        return "REVIEW"
    else:
        return "REJECT"


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_dataset(n: int = N_SAMPLES) -> pd.DataFrame:
    records = []

    # Target IRR tiers to get a balanced class distribution
    tier_config = [
        ("good",  0.155, 0.04, 0.38),
        ("fair",  0.065, 0.03, 0.40),
        ("poor",  -0.04, 0.08, 0.22),
    ]
    tier_counts = [int(n * f) for _, _, _, f in tier_config]
    tier_counts[-1] = n - sum(tier_counts[:-1])

    for (credit_tier_label, irr_mean, irr_std, _), count in zip(tier_config, tier_counts):
        for _ in range(count):
            state = sample_state()
            base_cagr = STATE_APPRECIATION[state]
            p10, p50, p90 = sample_appreciation_cagr(base_cagr)

            property_value = sample_property_value()
            ltv = sample_ltv()
            outstanding_mortgage = ltv * property_value
            equity = max(property_value - outstanding_mortgage, 1.0)
            credit_score = sample_credit_score(credit_tier_label)
            equity_share_pct = sample_equity_share()
            cap_multiple = sample_cap_multiple()
            term_years = sample_term()

            # Liens
            heloc_balance, second_mortgage, tax_lien, hoa_lien = sample_liens(equity, credit_tier_label)
            total_debt = outstanding_mortgage + heloc_balance + second_mortgage + tax_lien + hoa_lien
            cltv = total_debt / max(property_value, 1.0)

            # Credit history
            foreclosure_flag, bankruptcy_flag, delinquency_flag = sample_credit_history(credit_tier_label)

            # Property attributes
            prop_type, prop_age, owner_occ, arm_flag = sample_property_attributes()

            # Homeowner financials
            dti_ratio, emp_tier = sample_homeowner_financials(credit_tier_label)

            # HEI amount (back-computed from target IRR)
            target_irr = float(np.clip(RNG.normal(irr_mean, irr_std), -0.30, 0.40))
            hei_amount = compute_hei_amount(target_irr, equity_share_pct, property_value, p50, cap_multiple, term_years, equity)
            hei_amount = float(np.clip(hei_amount * RNG.uniform(0.85, 1.15), max(equity * 0.03, 10_000), min(equity * 0.65, 500_000)))

            if equity <= 0 or hei_amount <= 0:
                continue

            # Engineer features
            feat_df = engineer_features(
                property_value=property_value,
                outstanding_mortgage=outstanding_mortgage,
                heloc_balance=heloc_balance,
                second_mortgage_balance=second_mortgage,
                tax_lien_amount=tax_lien,
                hoa_lien_amount=hoa_lien,
                credit_score=credit_score,
                foreclosure_flag=foreclosure_flag,
                bankruptcy_flag=bankruptcy_flag,
                mortgage_delinquency_flag=delinquency_flag,
                dti_ratio=dti_ratio,
                employment_stability_tier=emp_tier,
                property_type_risk=prop_type,
                property_age=prop_age,
                owner_occupied=owner_occ,
                arm_flag=arm_flag,
                hei_amount=hei_amount,
                equity_share_pct=equity_share_pct,
                cap_multiple=cap_multiple,
                term_years=term_years,
                appreciation_cagr=p50,
                appreciation_p10=p10,
                appreciation_p90=p90,
                state=state,
            )

            irr_data = compute_irr_distribution(hei_amount, equity_share_pct, property_value, p10, p50, p90, cap_multiple, term_years)
            irr_base = irr_data["base_irr"] / 100
            hei_to_equity = hei_amount / max(equity, 1)
            equity_pct_val = equity / max(property_value, 1)
            subordinate_count = sum([heloc_balance > 0, second_mortgage > 0, tax_lien > 0, hoa_lien > 0])

            label = assign_label(
                irr_base, ltv, cltv, credit_score, equity_pct_val,
                irr_data["cap_exceedance_prob"], hei_to_equity,
                foreclosure_flag, bankruptcy_flag, delinquency_flag,
                owner_occ, prop_type, dti_ratio, subordinate_count,
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
            row["_cltv"] = cltv

            records.append(row)

    df = pd.DataFrame(records)
    print(f"Generated {len(df)} deals")
    print(df["label"].value_counts())
    print(f"\nIRR base: mean={df['_irr_base'].mean():.2f}% | median={df['_irr_base'].median():.2f}%")
    print(f"CLTV: mean={df['_cltv'].mean():.2%} | max={df['_cltv'].max():.2%}")
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    df.to_csv("data/synthetic_deals.csv", index=False)
    print("Saved → data/synthetic_deals.csv")
