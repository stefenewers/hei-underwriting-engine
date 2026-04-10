"""
HEI Underwriting Engine — Core Logic
=====================================
Feature engineering, IRR calculation, deal scoring, and SHAP explainability
for Home Equity Investment (HEI) deal analysis.
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
    # Sun Belt / High Growth
    "austin": 0.092,       "dallas": 0.088,       "phoenix": 0.091,
    "tampa": 0.093,        "miami": 0.089,         "orlando": 0.090,
    "nashville": 0.085,    "charlotte": 0.088,     "raleigh": 0.091,
    "atlanta": 0.082,      "jacksonville": 0.086,  "san antonio": 0.072,
    "houston": 0.058,      "las vegas": 0.082,     "riverside": 0.088,

    # West Coast
    "los angeles": 0.071,  "san diego": 0.082,     "seattle": 0.073,
    "san francisco": 0.038,"portland": 0.059,      "denver": 0.071,
    "sacramento": 0.079,   "salt lake city": 0.091,

    # Northeast
    "new york": 0.062,     "boston": 0.072,        "washington dc": 0.065,
    "philadelphia": 0.071, "baltimore": 0.063,     "hartford": 0.075,
    "providence": 0.082,

    # Midwest
    "chicago": 0.055,      "minneapolis": 0.065,   "columbus": 0.074,
    "indianapolis": 0.072, "kansas city": 0.073,   "st louis": 0.058,
    "cleveland": 0.062,    "detroit": 0.068,       "milwaukee": 0.065,
    "cincinnati": 0.070,   "pittsburgh": 0.060,

    # National fallback
    "national": 0.068,
}

# State-level CAGR fallbacks (for ZIP prefix lookups)
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


def get_appreciation_rate(state: str, metro: str = None) -> float:
    """
    Retrieve 5-year CAGR for a given market.
    Falls back gracefully: metro → state → national average.
    """
    if metro:
        key = metro.lower().strip()
        for k, v in METRO_APPRECIATION.items():
            if k in key or key in k:
                return v
    if state:
        return STATE_APPRECIATION.get(state.upper(), METRO_APPRECIATION["national"])
    return METRO_APPRECIATION["national"]


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
    """
    Compute IRR for an HEI deal under a given appreciation scenario.

    Parameters
    ----------
    investment      : Dollar amount invested by the HEI operator
    equity_share    : Fractional claim on appreciation (e.g., 0.15 = 15%)
    home_value      : Current estimated home value
    appreciation_rate : Annual appreciation rate (CAGR)
    cap_multiple    : Max return expressed as a multiple of investment (e.g., 2.0)
    term_years      : Investment horizon in years

    Returns
    -------
    irr             : Annualized internal rate of return
    net_return      : Nominal dollar return to the operator
    cap_hit         : Whether the cap was binding (True if cap constrained return)
    """
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
    """
    Compute IRR across the appreciation distribution (bear / base / bull scenarios).
    """
    scenarios = {
        "bear":  appreciation_p10,
        "base":  appreciation_p50,
        "bull":  appreciation_p90,
    }
    results = {}
    cap_probs = []

    for label, rate in scenarios.items():
        irr, net_return, cap_hit = calculate_irr(
            investment, equity_share, home_value, rate, cap_multiple, term_years
        )
        results[label] = {
            "irr": round(irr * 100, 2),
            "net_return": round(net_return, 0),
            "cap_hit": cap_hit,
        }
        cap_probs.append(float(cap_hit))

    # Approximate cap exceedance probability: interpolate across scenarios
    # Bear = 16th pct, Base = 50th pct, Bull = 84th pct (±1σ bands)
    # Simple weighted average assuming symmetry
    cap_exceedance_prob = round(
        0.16 * cap_probs[0] + 0.68 * cap_probs[1] + 0.16 * cap_probs[2], 2
    )

    return {
        "scenarios": results,
        "cap_exceedance_prob": cap_exceedance_prob,
        "base_irr": results["base"]["irr"],
    }


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "ltv",
    "equity_pct",
    "hei_to_equity_ratio",
    "credit_tier",
    "log_property_value",
    "appreciation_cagr",
    "appreciation_5yr_total",
    "cap_multiple",
    "equity_share_pct",
    "term_years",
    "expected_irr_base",
    "cap_exceedance_prob",
    "log_investment",
    "investment_to_value_pct",
]


def engineer_features(
    property_value: float,
    outstanding_mortgage: float,
    credit_score: int,
    hei_amount: float,
    equity_share_pct: float,
    cap_multiple: float,
    term_years: int,
    appreciation_cagr: float,
    appreciation_p10: float,
    appreciation_p90: float,
) -> pd.DataFrame:
    """
    Transform raw deal inputs into the feature vector used by the ML models.
    All features are interpretable and grounded in HEI domain logic.
    """
    equity = max(property_value - outstanding_mortgage, 0.0)
    ltv = outstanding_mortgage / max(property_value, 1.0)
    equity_pct = equity / max(property_value, 1.0)
    hei_to_equity = hei_amount / max(equity, 1.0)

    # Credit score → ordinal risk tier (0 = subprime, 3 = prime)
    if credit_score >= 740:
        credit_tier = 3
    elif credit_score >= 680:
        credit_tier = 2
    elif credit_score >= 620:
        credit_tier = 1
    else:
        credit_tier = 0

    appreciation_5yr_total = (1 + appreciation_cagr) ** 5 - 1

    irr_data = compute_irr_distribution(
        hei_amount, equity_share_pct, property_value,
        appreciation_p10, appreciation_cagr, appreciation_p90,
        cap_multiple, term_years,
    )

    features = {
        "ltv": ltv,
        "equity_pct": equity_pct,
        "hei_to_equity_ratio": hei_to_equity,
        "credit_tier": credit_tier,
        "log_property_value": np.log1p(property_value),
        "appreciation_cagr": appreciation_cagr,
        "appreciation_5yr_total": appreciation_5yr_total,
        "cap_multiple": cap_multiple,
        "equity_share_pct": equity_share_pct,
        "term_years": term_years,
        "expected_irr_base": irr_data["base_irr"] / 100,  # back to decimal
        "cap_exceedance_prob": irr_data["cap_exceedance_prob"],
        "log_investment": np.log1p(hei_amount),
        "investment_to_value_pct": hei_amount / max(property_value, 1.0),
    }

    return pd.DataFrame([features])[FEATURE_NAMES]


# ---------------------------------------------------------------------------
# Deal Scorer (deterministic composite score)
# ---------------------------------------------------------------------------

def compute_deal_score(
    irr_base: float,
    cap_exceedance_prob: float,
    ltv: float,
    credit_tier: int,
    equity_pct: float,
    risk_class: str,
) -> int:
    """
    Compute a 0–100 deal score from key underwriting signals.

    Weights:
      - IRR quality:          35 pts
      - LTV safety:           25 pts
      - Credit quality:       20 pts
      - Cap efficiency:       10 pts
      - Equity cushion:       10 pts
    """
    # IRR component (0–35): normalize 0–20% IRR range
    irr_score = min(irr_base / 0.20, 1.0) * 35

    # LTV component (0–25): lower LTV → higher score
    ltv_score = max(0.0, 1.0 - ltv / 0.85) * 25

    # Credit component (0–20)
    credit_score_pts = (credit_tier / 3.0) * 20

    # Cap efficiency (0–10): lower cap exceedance prob is better
    # (hitting the cap means we're leaving money on the table)
    cap_score = max(0.0, 1.0 - cap_exceedance_prob) * 10

    # Equity cushion (0–10): more equity → better collateral
    equity_score = min(equity_pct / 0.50, 1.0) * 10

    total = irr_score + ltv_score + credit_score_pts + cap_score + equity_score

    # Risk class adjustment
    if risk_class == "REJECT":
        total = min(total, 35)
    elif risk_class == "REVIEW":
        total = min(total, 65)

    return int(round(min(max(total, 0), 100)))


# ---------------------------------------------------------------------------
# SHAP Feature Labels (human-readable)
# ---------------------------------------------------------------------------

FEATURE_LABELS = {
    "ltv":                    "Loan-to-Value Ratio",
    "equity_pct":             "Equity as % of Value",
    "hei_to_equity_ratio":    "HEI Amount / Available Equity",
    "credit_tier":            "Credit Quality Tier",
    "log_property_value":     "Log Property Value",
    "appreciation_cagr":      "Market Appreciation (CAGR)",
    "appreciation_5yr_total": "5-Year Appreciation (Total)",
    "cap_multiple":           "Return Cap Multiple",
    "equity_share_pct":       "Equity Share %",
    "term_years":             "Investment Term",
    "expected_irr_base":      "Expected IRR (Base Case)",
    "cap_exceedance_prob":    "Cap Exceedance Probability",
    "log_investment":         "Log HEI Amount",
    "investment_to_value_pct":"Investment as % of Value",
}
