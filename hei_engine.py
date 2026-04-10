"""
HEI Underwriting Engine — Core Logic
=====================================
Feature engineering, IRR calculation, deal scoring, and SHAP explainability
for Home Equity Investment (HEI) deal analysis.

v2: Expanded from 14 → 28 features, adding:
  - Combined LTV (CLTV) across all lien positions
  - Subordinate liens (HELOC, 2nd mortgage, tax/HOA liens)
  - Foreclosure, bankruptcy, and delinquency history flags
  - Property type risk tier (SFR / townhome / condo / manufactured)
  - Flood zone risk, ARM flag, property age
  - DTI ratio and employment stability tier
  - Market liquidity score
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Metro / Regional Appreciation Data
# Compiled from Zillow ZHVI public research data (5-year CAGR approximations)
# ---------------------------------------------------------------------------

METRO_APPRECIATION = {
    "austin": 0.092,       "dallas": 0.088,       "phoenix": 0.091,
    "tampa": 0.093,        "miami": 0.089,         "orlando": 0.090,
    "nashville": 0.085,    "charlotte": 0.088,     "raleigh": 0.091,
    "atlanta": 0.082,      "jacksonville": 0.086,  "san antonio": 0.072,
    "houston": 0.058,      "las vegas": 0.082,     "riverside": 0.088,
    "los angeles": 0.071,  "san diego": 0.082,     "seattle": 0.073,
    "san francisco": 0.038,"portland": 0.059,      "denver": 0.071,
    "sacramento": 0.079,   "salt lake city": 0.091,
    "new york": 0.062,     "boston": 0.072,        "washington dc": 0.065,
    "philadelphia": 0.071, "baltimore": 0.063,     "hartford": 0.075,
    "providence": 0.082,
    "chicago": 0.055,      "minneapolis": 0.065,   "columbus": 0.074,
    "indianapolis": 0.072, "kansas city": 0.073,   "st louis": 0.058,
    "cleveland": 0.062,    "detroit": 0.068,       "milwaukee": 0.065,
    "cincinnati": 0.070,   "pittsburgh": 0.060,
    "national": 0.068,
}

STATE_APPRECIATION = {
    "AL": 0.072, "AK": 0.041, "AZ": 0.088, "AR": 0.068, "CA": 0.072,
    "CO": 0.071, "CT": 0.075, "DE": 0.071, "FL": 0.089, "GA": 0.082,
    "HI": 0.055, "ID": 0.091, "IL": 0.059, "IN": 0.072, "IA": 0.062,
    "KS": 0.068, "KY": 0.072, "LA": 0.052, "ME": 0.079, "MD": 0.065,
    "MA": 0.074, "MI": 0.068, "MN": 0.065, "MS": 0.062, "MO": 0.065,
    "MT": 0.088, "NE": 0.068, "NV": 0.082, "NH": 0.082, "NJ": 0.069,
    "NM": 0.072, "NY": 0.062, "NC": 0.088, "ND": 0.048, "OH": 0.066,
    "OK": 0.061, "OR": 0.059, "PA": 0.065, "RI": 0.082, "SC": 0.084,
    "SD": 0.068, "TN": 0.085, "TX": 0.072, "UT": 0.091, "VT": 0.079,
    "VA": 0.072, "WA": 0.073, "WV": 0.058, "WI": 0.065, "WY": 0.062,
}

# State-level market liquidity index (0–1, higher = more liquid market)
# Derived from median days-on-market data; urban/coastal markets are more liquid
STATE_LIQUIDITY = {
    "CA": 0.85, "FL": 0.88, "TX": 0.82, "NY": 0.75, "WA": 0.83,
    "CO": 0.84, "AZ": 0.86, "GA": 0.81, "NC": 0.82, "TN": 0.80,
    "VA": 0.79, "MD": 0.78, "MA": 0.82, "NJ": 0.74, "IL": 0.72,
    "MN": 0.76, "OR": 0.80, "NV": 0.84, "UT": 0.85, "SC": 0.79,
    "PA": 0.71, "OH": 0.70, "MI": 0.70, "IN": 0.72, "MO": 0.71,
    "WI": 0.69, "KY": 0.68, "AL": 0.67, "LA": 0.66, "AR": 0.65,
    "MS": 0.62, "WV": 0.60, "MT": 0.72, "ID": 0.78, "NM": 0.68,
    "ND": 0.60, "SD": 0.61, "WY": 0.62, "AK": 0.55, "HI": 0.74,
    "RI": 0.76, "CT": 0.73, "DE": 0.74, "NH": 0.78, "VT": 0.70,
    "ME": 0.68, "KS": 0.67, "NE": 0.68, "IA": 0.65, "OK": 0.64,
}

# Flood zone risk by state (population-weighted exposure, 0–1)
STATE_FLOOD_RISK = {
    "FL": 0.45, "LA": 0.55, "TX": 0.35, "SC": 0.30, "NC": 0.28,
    "VA": 0.22, "MD": 0.25, "NJ": 0.28, "NY": 0.22, "MA": 0.18,
    "CT": 0.20, "RI": 0.22, "GA": 0.18, "AL": 0.25, "MS": 0.30,
    "CA": 0.15, "WA": 0.12, "OR": 0.10, "HI": 0.20, "AK": 0.12,
    # Most interior/mountain states have low flood exposure
    "CO": 0.05, "UT": 0.05, "AZ": 0.08, "NV": 0.05, "MT": 0.06,
    "ID": 0.07, "WY": 0.06, "ND": 0.18, "SD": 0.14, "NE": 0.14,
    "KS": 0.12, "MN": 0.14, "IA": 0.18, "MO": 0.18, "AR": 0.22,
    "TN": 0.12, "KY": 0.12, "OH": 0.08, "IN": 0.10, "IL": 0.12,
    "MI": 0.08, "WI": 0.08, "PA": 0.12, "WV": 0.10, "DE": 0.20,
    "OK": 0.12, "NM": 0.06, "ME": 0.10, "VT": 0.08, "NH": 0.08,
}


def get_appreciation_rate(state: str, metro: str = None) -> float:
    if metro:
        key = metro.lower().strip()
        for k, v in METRO_APPRECIATION.items():
            if k in key or key in k:
                return v
    if state:
        return STATE_APPRECIATION.get(state.upper(), METRO_APPRECIATION["national"])
    return METRO_APPRECIATION["national"]


def get_market_liquidity(state: str) -> float:
    return STATE_LIQUIDITY.get(state.upper(), 0.70)


def get_flood_risk(state: str) -> float:
    return STATE_FLOOD_RISK.get(state.upper(), 0.12)


# ---------------------------------------------------------------------------
# Enumerations / Tiers
# ---------------------------------------------------------------------------

# Property type risk tiers (0 = lowest risk, 3 = highest / often excluded)
PROPERTY_TYPE_OPTIONS = {
    "Single-Family Residence (SFR)": 0,
    "Townhome / Row Home":           1,
    "Condominium":                   2,
    "Manufactured / Mobile Home":    3,
}

# Employment stability tiers
EMPLOYMENT_OPTIONS = {
    "W-2 / Full-Time Employee": 2,
    "Self-Employed / 1099":     1,
    "Retired (Fixed Income)":   2,  # treated same as W2 — stable income
    "Unemployed / Irregular":   0,
}


# ---------------------------------------------------------------------------
# IRR Calculator
# ---------------------------------------------------------------------------

def calculate_irr(
    investment: float,
    equity_share: float,
    home_value: float,
    appreciation_rate: float,
    cap_multiple: float,
    term_years: int,
) -> Tuple[float, float, bool]:
    if investment <= 0 or home_value <= 0:
        return 0.0, 0.0, False

    final_value = home_value * ((1 + appreciation_rate) ** term_years)
    appreciation_gain = max(final_value - home_value, 0.0)
    gross_return = equity_share * appreciation_gain
    cap_amount = cap_multiple * investment

    cap_hit = gross_return > cap_amount
    net_return = min(gross_return, cap_amount)

    if net_return <= 0:
        return -1.0, net_return, cap_hit

    irr = (net_return / investment) ** (1.0 / term_years) - 1.0
    return float(irr), float(net_return), cap_hit


def compute_irr_distribution(
    investment: float,
    equity_share: float,
    home_value: float,
    appreciation_p10: float,
    appreciation_p50: float,
    appreciation_p90: float,
    cap_multiple: float,
    term_years: int,
) -> Dict:
    scenarios = {"bear": appreciation_p10, "base": appreciation_p50, "bull": appreciation_p90}
    results = {}
    cap_probs = []

    for label, rate in scenarios.items():
        irr, net_return, cap_hit = calculate_irr(
            investment, equity_share, home_value, rate, cap_multiple, term_years
        )
        results[label] = {"irr": round(irr * 100, 2), "net_return": round(net_return, 0), "cap_hit": cap_hit}
        cap_probs.append(float(cap_hit))

    cap_exceedance_prob = round(0.16 * cap_probs[0] + 0.68 * cap_probs[1] + 0.16 * cap_probs[2], 2)

    return {
        "scenarios": results,
        "cap_exceedance_prob": cap_exceedance_prob,
        "base_irr": results["base"]["irr"],
    }


# ---------------------------------------------------------------------------
# Feature Engineering — v2 (28 features)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # --- Collateral / LTV ---
    "ltv",
    "cltv",
    "equity_pct",
    "hei_to_equity_ratio",
    "subordinate_lien_count",
    "heloc_utilization",

    # --- Creditworthiness ---
    "credit_tier",
    "foreclosure_flag",
    "bankruptcy_flag",
    "mortgage_delinquency_flag",
    "dti_ratio",
    "employment_stability_tier",

    # --- Property quality ---
    "property_type_risk",
    "property_age_normalized",
    "owner_occupied",
    "flood_zone_risk",
    "arm_flag",

    # --- Market & deal ---
    "log_property_value",
    "appreciation_cagr",
    "appreciation_5yr_total",
    "market_liquidity_score",
    "cap_multiple",
    "equity_share_pct",
    "term_years",

    # --- Return metrics ---
    "expected_irr_base",
    "cap_exceedance_prob",
    "log_investment",
    "investment_to_value_pct",
]


def engineer_features(
    # Collateral
    property_value: float,
    outstanding_mortgage: float,
    heloc_balance: float,
    second_mortgage_balance: float,
    tax_lien_amount: float,
    hoa_lien_amount: float,
    # Credit
    credit_score: int,
    foreclosure_flag: int,
    bankruptcy_flag: int,
    mortgage_delinquency_flag: int,
    dti_ratio: float,
    employment_stability_tier: int,
    # Property
    property_type_risk: int,
    property_age: int,
    owner_occupied: int,
    arm_flag: int,
    # Deal
    hei_amount: float,
    equity_share_pct: float,
    cap_multiple: float,
    term_years: int,
    # Market
    appreciation_cagr: float,
    appreciation_p10: float,
    appreciation_p90: float,
    state: str = "FL",
) -> pd.DataFrame:
    """
    Transform raw deal inputs into the 28-feature vector used by the ML models.
    All features are interpretable and grounded in HEI domain logic.
    """
    # --- Collateral calculations ---
    total_debt = outstanding_mortgage + heloc_balance + second_mortgage_balance + tax_lien_amount + hoa_lien_amount
    equity = max(property_value - outstanding_mortgage, 0.0)
    ltv = outstanding_mortgage / max(property_value, 1.0)
    cltv = total_debt / max(property_value, 1.0)
    equity_pct = equity / max(property_value, 1.0)
    hei_to_equity = hei_amount / max(equity, 1.0)

    subordinate_lien_count = sum([
        1 if heloc_balance > 0 else 0,
        1 if second_mortgage_balance > 0 else 0,
        1 if tax_lien_amount > 0 else 0,
        1 if hoa_lien_amount > 0 else 0,
    ])

    # HELOC utilization: 0 if no HELOC, else balance / typical HELOC limit (20% of equity)
    typical_heloc_limit = max(equity * 0.20, 1.0)
    heloc_utilization = min(heloc_balance / typical_heloc_limit, 1.0) if heloc_balance > 0 else 0.0

    # --- Credit ---
    if credit_score >= 740:
        credit_tier = 3
    elif credit_score >= 680:
        credit_tier = 2
    elif credit_score >= 620:
        credit_tier = 1
    else:
        credit_tier = 0

    # --- Market ---
    appreciation_5yr_total = (1 + appreciation_cagr) ** 5 - 1
    market_liquidity = get_market_liquidity(state)
    flood_risk = get_flood_risk(state)

    # Property age normalized (0 = new, 1 = 80+ years old)
    property_age_normalized = min(property_age / 80.0, 1.0)

    # --- Return metrics ---
    irr_data = compute_irr_distribution(
        hei_amount, equity_share_pct, property_value,
        appreciation_p10, appreciation_cagr, appreciation_p90,
        cap_multiple, term_years,
    )

    features = {
        "ltv":                       ltv,
        "cltv":                      cltv,
        "equity_pct":                equity_pct,
        "hei_to_equity_ratio":       hei_to_equity,
        "subordinate_lien_count":    subordinate_lien_count,
        "heloc_utilization":         heloc_utilization,
        "credit_tier":               credit_tier,
        "foreclosure_flag":          float(foreclosure_flag),
        "bankruptcy_flag":           float(bankruptcy_flag),
        "mortgage_delinquency_flag": float(mortgage_delinquency_flag),
        "dti_ratio":                 dti_ratio,
        "employment_stability_tier": float(employment_stability_tier),
        "property_type_risk":        float(property_type_risk),
        "property_age_normalized":   property_age_normalized,
        "owner_occupied":            float(owner_occupied),
        "flood_zone_risk":           flood_risk,
        "arm_flag":                  float(arm_flag),
        "log_property_value":        np.log1p(property_value),
        "appreciation_cagr":         appreciation_cagr,
        "appreciation_5yr_total":    appreciation_5yr_total,
        "market_liquidity_score":    market_liquidity,
        "cap_multiple":              cap_multiple,
        "equity_share_pct":          equity_share_pct,
        "term_years":                float(term_years),
        "expected_irr_base":         irr_data["base_irr"] / 100,
        "cap_exceedance_prob":       irr_data["cap_exceedance_prob"],
        "log_investment":            np.log1p(hei_amount),
        "investment_to_value_pct":   hei_amount / max(property_value, 1.0),
    }

    return pd.DataFrame([features])[FEATURE_NAMES]


# ---------------------------------------------------------------------------
# Deal Scorer (v2 — updated weights for expanded feature set)
# ---------------------------------------------------------------------------

def compute_deal_score(
    irr_base: float,
    cap_exceedance_prob: float,
    ltv: float,
    cltv: float,
    credit_tier: int,
    equity_pct: float,
    risk_class: str,
    foreclosure_flag: int = 0,
    bankruptcy_flag: int = 0,
    mortgage_delinquency_flag: int = 0,
    property_type_risk: int = 0,
    owner_occupied: int = 1,
    dti_ratio: float = 0.35,
) -> int:
    """
    Compute a 0–100 deal score from key underwriting signals.

    Weights (v2):
      IRR quality:           30 pts
      CLTV safety:           25 pts
      Credit quality:        15 pts
      Credit history:        10 pts  (new — foreclosure/bankruptcy/delinquency)
      Cap efficiency:         8 pts
      Equity cushion:         7 pts
      Property quality:       5 pts  (new — type, owner-occupied)
    """
    # Immediate hard-stop overrides
    if foreclosure_flag or bankruptcy_flag:
        return min(15, int(irr_base * 100))
    if not owner_occupied:
        return min(20, int(irr_base * 100))
    if property_type_risk >= 3:  # manufactured home
        return min(10, int(irr_base * 100))

    # IRR component (0–30)
    irr_score = min(irr_base / 0.20, 1.0) * 30

    # CLTV component (0–25): use combined LTV — stricter than first-lien LTV
    cltv_score = max(0.0, 1.0 - cltv / 0.90) * 25

    # Credit score component (0–15)
    credit_score_pts = (credit_tier / 3.0) * 15

    # Credit history component (0–10)
    history_score = 10.0
    if mortgage_delinquency_flag:
        history_score -= 6
    # Foreclosure/bankruptcy already handled above as hard stops

    # Cap efficiency (0–8)
    cap_score = max(0.0, 1.0 - cap_exceedance_prob) * 8

    # Equity cushion (0–7)
    equity_score = min(equity_pct / 0.50, 1.0) * 7

    # Property quality (0–5)
    prop_score = max(0.0, (3 - property_type_risk) / 3.0) * 5
    if not owner_occupied:
        prop_score = 0

    # DTI penalty
    if dti_ratio > 0.50:
        irr_score *= 0.85
    elif dti_ratio > 0.43:
        irr_score *= 0.92

    total = irr_score + cltv_score + credit_score_pts + history_score + cap_score + equity_score + prop_score

    # Risk class cap
    if risk_class == "REJECT":
        total = min(total, 35)
    elif risk_class == "REVIEW":
        total = min(total, 65)

    return int(round(min(max(total, 0), 100)))


# ---------------------------------------------------------------------------
# Underwriting Checklist (pass/fail for each threshold)
# ---------------------------------------------------------------------------

def generate_checklist(
    ltv: float,
    cltv: float,
    credit_score: int,
    foreclosure_flag: int,
    bankruptcy_flag: int,
    mortgage_delinquency_flag: int,
    owner_occupied: int,
    property_type_risk: int,
    dti_ratio: float,
    irr_base: float,
    equity_pct: float,
    arm_flag: int,
    flood_zone_risk: float,
) -> list:
    """
    Returns a list of (label, pass_bool, detail) tuples for the UI checklist.
    """
    checks = [
        ("First Lien LTV ≤ 80%",        ltv <= 0.80,                  f"{ltv:.1%}"),
        ("Combined LTV (CLTV) ≤ 87%",   cltv <= 0.87,                 f"{cltv:.1%}"),
        ("Credit Score ≥ 620",           credit_score >= 620,          str(credit_score)),
        ("No Prior Foreclosure",         foreclosure_flag == 0,        "Clear" if not foreclosure_flag else "⚠️ Flag"),
        ("No Active Bankruptcy",         bankruptcy_flag == 0,         "Clear" if not bankruptcy_flag else "⚠️ Flag"),
        ("No Recent Delinquency",        mortgage_delinquency_flag==0, "Clean" if not mortgage_delinquency_flag else "30+ day late"),
        ("Owner-Occupied Property",      owner_occupied == 1,          "Yes" if owner_occupied else "No — Investment"),
        ("Eligible Property Type",       property_type_risk <= 2,      ["SFR","Townhome","Condo","Manufactured"][min(property_type_risk,3)]),
        ("DTI ≤ 50%",                    dti_ratio <= 0.50,            f"{dti_ratio:.1%}"),
        ("Equity Cushion ≥ 15%",         equity_pct >= 0.15,           f"{equity_pct:.1%}"),
        ("Viable Base IRR (≥ 5%)",       irr_base >= 5.0,              f"{irr_base:.1f}%"),
        ("Fixed-Rate Mortgage",          arm_flag == 0,                "Fixed" if not arm_flag else "ARM — ⚠️"),
        ("Low Flood Zone Exposure",      flood_zone_risk <= 0.30,      "Low" if flood_zone_risk <= 0.15 else "Moderate" if flood_zone_risk <= 0.30 else "High"),
    ]
    return checks


# ---------------------------------------------------------------------------
# SHAP Feature Labels (human-readable)
# ---------------------------------------------------------------------------

FEATURE_LABELS = {
    "ltv":                       "First-Lien LTV",
    "cltv":                      "Combined LTV (All Liens)",
    "equity_pct":                "Equity as % of Value",
    "hei_to_equity_ratio":       "HEI Amount / Available Equity",
    "subordinate_lien_count":    "Number of Subordinate Liens",
    "heloc_utilization":         "HELOC Utilization Rate",
    "credit_tier":               "Credit Quality Tier",
    "foreclosure_flag":          "Prior Foreclosure Flag",
    "bankruptcy_flag":           "Prior Bankruptcy Flag",
    "mortgage_delinquency_flag": "Mortgage Delinquency History",
    "dti_ratio":                 "Debt-to-Income Ratio",
    "employment_stability_tier": "Employment Stability",
    "property_type_risk":        "Property Type Risk Tier",
    "property_age_normalized":   "Property Age",
    "owner_occupied":            "Owner-Occupied Flag",
    "flood_zone_risk":           "Flood Zone Exposure",
    "arm_flag":                  "Adjustable-Rate Mortgage Flag",
    "log_property_value":        "Log Property Value",
    "appreciation_cagr":         "Market Appreciation (CAGR)",
    "appreciation_5yr_total":    "5-Year Appreciation (Total)",
    "market_liquidity_score":    "Market Liquidity Score",
    "cap_multiple":              "Return Cap Multiple",
    "equity_share_pct":          "Equity Share %",
    "term_years":                "Investment Term",
    "expected_irr_base":         "Expected IRR (Base Case)",
    "cap_exceedance_prob":       "Cap Exceedance Probability",
    "log_investment":            "Log HEI Amount",
    "investment_to_value_pct":   "Investment as % of Value",
}
