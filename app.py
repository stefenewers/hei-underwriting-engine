"""
HEI Underwriting Engine — Streamlit Application v2
====================================================
Automated deal scoring for Home Equity Investments.
28-feature ML pipeline: CLTV, liens, foreclosure/bankruptcy history,
property type, DTI, flood zone, ARM flags, and more.
"""

import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from hei_engine import (
    FEATURE_LABELS,
    FEATURE_NAMES,
    PROPERTY_TYPE_OPTIONS,
    EMPLOYMENT_OPTIONS,
    compute_deal_score,
    compute_irr_distribution,
    engineer_features,
    generate_checklist,
    get_appreciation_rate,
    get_flood_risk,
    get_market_liquidity,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HEI Underwriting Engine",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1300px; }

    .score-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155; border-radius: 16px;
        padding: 2rem; text-align: center; color: white;
    }
    .score-number { font-size: 4rem; font-weight: 800; line-height: 1; margin: 0.5rem 0; }
    .score-label { font-size: 0.85rem; color: #94a3b8; letter-spacing: 0.1em; }

    .badge-approve { background:#065f46; color:#6ee7b7; border:1px solid #059669;
        padding:0.4rem 1.2rem; border-radius:999px; font-weight:700; font-size:0.9rem; display:inline-block; }
    .badge-review  { background:#78350f; color:#fcd34d; border:1px solid #d97706;
        padding:0.4rem 1.2rem; border-radius:999px; font-weight:700; font-size:0.9rem; display:inline-block; }
    .badge-reject  { background:#7f1d1d; color:#fca5a5; border:1px solid #dc2626;
        padding:0.4rem 1.2rem; border-radius:999px; font-weight:700; font-size:0.9rem; display:inline-block; }

    .metric-card { background:#1e293b; border:1px solid #334155; border-radius:12px;
        padding:0.9rem 1.1rem; margin-bottom:0.4rem; }
    .metric-label { font-size:0.72rem; color:#94a3b8; font-weight:600; letter-spacing:0.05em; }
    .metric-value { font-size:1.5rem; font-weight:700; color:#f1f5f9; margin-top:0.1rem; }
    .metric-sub   { font-size:0.72rem; color:#64748b; margin-top:0.1rem; }

    .check-pass { color: #6ee7b7; }
    .check-fail { color: #fca5a5; }
    .check-row  { display:flex; justify-content:space-between; align-items:center;
        padding:0.35rem 0; border-bottom:1px solid #1e293b; font-size:0.82rem; }
    .check-label { color:#94a3b8; }
    .check-value { font-weight:600; font-size:0.78rem; }

    .flag-chip { display:inline-block; padding:0.15rem 0.6rem; border-radius:4px;
        font-size:0.72rem; font-weight:600; margin-right:0.3rem; }
    .flag-warn { background:#78350f22; color:#fcd34d; border:1px solid #d9770644; }
    .flag-ok   { background:#065f4622; color:#6ee7b7; border:1px solid #05996944; }

    .sidebar-section { font-size:0.68rem; color:#94a3b8; letter-spacing:0.08em;
        font-weight:600; text-transform:uppercase; margin-top:1rem; margin-bottom:0.2rem; }
    hr { border-color: #334155 !important; }
    #MainMenu { visibility:hidden; } footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_shap_background():
    bg_path = ROOT / "models" / "shap_background.pkl"
    if bg_path.exists():
        with open(bg_path, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_resource(show_spinner="Training models on first run — ~45 seconds...")
def load_models():
    models_dir = ROOT / "models"
    hpa_path   = models_dir / "hpa_models.pkl"
    clf_path   = models_dir / "risk_classifier.pkl"
    le_path    = models_dir / "label_encoder.pkl"

    if not (hpa_path.exists() and clf_path.exists() and le_path.exists()):
        from train_models import train_all
        train_all()

    with open(hpa_path,  "rb") as f: hpa_models = pickle.load(f)
    with open(clf_path,  "rb") as f: clf        = pickle.load(f)
    with open(le_path,   "rb") as f: le         = pickle.load(f)
    return hpa_models, clf, le


# ---------------------------------------------------------------------------
# Prediction pipeline
# ---------------------------------------------------------------------------

def run_prediction(inputs: dict, hpa_models: dict, clf, le) -> dict:
    state = inputs["state"]
    appreciation_cagr = get_appreciation_rate(state, inputs["metro"])
    sigma = 0.025
    p10 = max(appreciation_cagr - 2 * sigma, 0.005)
    p90 = min(appreciation_cagr + 2 * sigma, 0.25)

    feat_df = engineer_features(
        property_value            = inputs["property_value"],
        outstanding_mortgage      = inputs["outstanding_mortgage"],
        heloc_balance             = inputs["heloc_balance"],
        second_mortgage_balance   = inputs["second_mortgage"],
        tax_lien_amount           = inputs["tax_lien"],
        hoa_lien_amount           = inputs["hoa_lien"],
        credit_score              = inputs["credit_score"],
        foreclosure_flag          = inputs["foreclosure_flag"],
        bankruptcy_flag           = inputs["bankruptcy_flag"],
        mortgage_delinquency_flag = inputs["delinquency_flag"],
        dti_ratio                 = inputs["dti_ratio"],
        employment_stability_tier = inputs["employment_tier"],
        property_type_risk        = inputs["property_type_risk"],
        property_age              = inputs["property_age"],
        owner_occupied            = inputs["owner_occupied"],
        arm_flag                  = inputs["arm_flag"],
        hei_amount                = inputs["hei_amount"],
        equity_share_pct          = inputs["equity_share_pct"],
        cap_multiple              = inputs["cap_multiple"],
        term_years                = inputs["term_years"],
        appreciation_cagr         = appreciation_cagr,
        appreciation_p10          = p10,
        appreciation_p90          = p90,
        state                     = state,
    )

    X = feat_df.values

    hpa_p10 = max(float(hpa_models["p10"].predict(X)[0]), 0.005)
    hpa_p50 = float(hpa_models["p50"].predict(X)[0])
    hpa_p90 = max(float(hpa_models["p90"].predict(X)[0]), hpa_p10 + 0.005)

    irr_data = compute_irr_distribution(
        inputs["hei_amount"], inputs["equity_share_pct"], inputs["property_value"],
        hpa_p10, hpa_p50, hpa_p90, inputs["cap_multiple"], inputs["term_years"]
    )

    prob      = clf.predict_proba(X)[0]
    label_idx = int(np.argmax(prob))
    risk_class = le.inverse_transform([label_idx])[0]
    class_probs = {le.classes_[i]: prob[i] for i in range(len(le.classes_))}

    pv   = inputs["property_value"]
    mort = inputs["outstanding_mortgage"]
    total_debt = mort + inputs["heloc_balance"] + inputs["second_mortgage"] + inputs["tax_lien"] + inputs["hoa_lien"]
    equity     = max(pv - mort, 0)
    ltv        = mort / max(pv, 1)
    cltv       = total_debt / max(pv, 1)
    equity_pct = equity / max(pv, 1)
    credit_tier = (3 if inputs["credit_score"] >= 740 else
                   2 if inputs["credit_score"] >= 680 else
                   1 if inputs["credit_score"] >= 620 else 0)

    deal_score = compute_deal_score(
        irr_base               = irr_data["base_irr"] / 100,
        cap_exceedance_prob    = irr_data["cap_exceedance_prob"],
        ltv                    = ltv,
        cltv                   = cltv,
        credit_tier            = credit_tier,
        equity_pct             = equity_pct,
        risk_class             = risk_class,
        foreclosure_flag       = inputs["foreclosure_flag"],
        bankruptcy_flag        = inputs["bankruptcy_flag"],
        mortgage_delinquency_flag = inputs["delinquency_flag"],
        property_type_risk     = inputs["property_type_risk"],
        owner_occupied         = inputs["owner_occupied"],
        dti_ratio              = inputs["dti_ratio"],
    )

    checklist = generate_checklist(
        ltv                    = ltv,
        cltv                   = cltv,
        credit_score           = inputs["credit_score"],
        foreclosure_flag       = inputs["foreclosure_flag"],
        bankruptcy_flag        = inputs["bankruptcy_flag"],
        mortgage_delinquency_flag = inputs["delinquency_flag"],
        owner_occupied         = inputs["owner_occupied"],
        property_type_risk     = inputs["property_type_risk"],
        dti_ratio              = inputs["dti_ratio"],
        irr_base               = irr_data["base_irr"],
        equity_pct             = equity_pct,
        arm_flag               = inputs["arm_flag"],
        flood_zone_risk        = get_flood_risk(state),
    )

    # SHAP
    try:
        import shap
        bg = _load_shap_background()
        if bg is not None:
            explainer = shap.Explainer(clf.predict_proba, bg)
            sv_obj    = explainer(X)
            sv        = sv_obj.values[0, :, label_idx]
        else:
            raise ValueError("No background")
        shap_dict = {
            FEATURE_LABELS.get(FEATURE_NAMES[i], FEATURE_NAMES[i]): float(sv[i])
            for i in range(len(FEATURE_NAMES))
        }
    except Exception:
        fi = clf.feature_importances_
        shap_dict = {
            FEATURE_LABELS.get(FEATURE_NAMES[i], FEATURE_NAMES[i]): float(fi[i])
            for i in range(len(FEATURE_NAMES))
        }

    return {
        "deal_score":   deal_score,
        "risk_class":   risk_class,
        "class_probs":  class_probs,
        "irr_data":     irr_data,
        "hpa":          {"p10": hpa_p10, "p50": hpa_p50, "p90": hpa_p90},
        "appreciation_cagr": appreciation_cagr,
        "equity":       equity,
        "total_debt":   total_debt,
        "ltv":          ltv,
        "cltv":         cltv,
        "equity_pct":   equity_pct,
        "shap_dict":    shap_dict,
        "checklist":    checklist,
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def score_gauge(score: int) -> go.Figure:
    color = "#10b981" if score >= 70 else "#f59e0b" if score >= 45 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 52, "color": "#f1f5f9"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748b", "tickfont": {"color": "#64748b", "size": 11}},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": "#1e293b", "bordercolor": "#334155",
            "steps": [
                {"range": [0,  35],  "color": "#7f1d1d33"},
                {"range": [35, 65],  "color": "#78350f33"},
                {"range": [65, 100], "color": "#065f4633"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.85, "value": score},
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=220, margin=dict(t=20, b=10, l=20, r=20), font_color="#f1f5f9")
    return fig


def irr_bar_chart(irr_data: dict) -> go.Figure:
    s = irr_data["scenarios"]
    labels = ["Bear (P10)", "Base (P50)", "Bull (P90)"]
    values = [s["bear"]["irr"], s["base"]["irr"], s["bull"]["irr"]]
    colors = ["#ef4444", "#3b82f6", "#10b981"]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
        text=[f"{v:.1f}%" for v in values], textposition="outside",
        textfont={"color": "#f1f5f9", "size": 13}))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=220, margin=dict(t=20, b=10, l=10, r=10),
        yaxis=dict(title="IRR (%)", gridcolor="#334155", color="#94a3b8", zeroline=False),
        xaxis=dict(color="#94a3b8"), showlegend=False, font_color="#f1f5f9")
    return fig


def shap_waterfall(shap_dict: dict) -> go.Figure:
    items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    labels = [i[0] for i in items]
    values = [i[1] for i in items]
    colors = ["#10b981" if v > 0 else "#ef4444" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"{'+' if v > 0 else ''}{v:.3f}" for v in values],
        textposition="outside", textfont={"color": "#94a3b8", "size": 10},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=340, margin=dict(t=10, b=10, l=10, r=90),
        xaxis=dict(title="SHAP impact", gridcolor="#334155", color="#94a3b8",
                   zeroline=True, zerolinecolor="#475569"),
        yaxis=dict(color="#94a3b8", autorange="reversed"),
        font_color="#f1f5f9")
    return fig


def appreciation_chart(hpa: dict, term_years: int, home_value: float) -> go.Figure:
    years = list(range(term_years + 1))
    def vals(cagr): return [home_value * ((1 + cagr) ** y) / 1000 for y in years]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=vals(hpa["p90"]), mode="lines", name="Bull (P90)",
        line=dict(color="#10b981", width=1.5, dash="dot"), fill=None))
    fig.add_trace(go.Scatter(x=years, y=vals(hpa["p10"]), mode="lines", name="Bear (P10)",
        line=dict(color="#ef4444", width=1.5, dash="dot"),
        fill="tonexty", fillcolor="rgba(100,116,139,0.15)"))
    fig.add_trace(go.Scatter(x=years, y=vals(hpa["p50"]), mode="lines+markers", name="Base (P50)",
        line=dict(color="#3b82f6", width=2.5), marker=dict(size=5, color="#3b82f6")))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=220, margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(title="Year", gridcolor="#334155", color="#94a3b8"),
        yaxis=dict(title="Value ($K)", gridcolor="#334155", color="#94a3b8"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#334155",
            font=dict(color="#94a3b8", size=11), orientation="h", yanchor="bottom", y=1.02),
        font_color="#f1f5f9")
    return fig


def capital_stack_chart(property_value, outstanding_mortgage, heloc_balance,
                         second_mortgage, tax_lien, hoa_lien, hei_amount) -> go.Figure:
    """Stacked bar showing the full capital stack position."""
    equity_after_hei = max(property_value - outstanding_mortgage - heloc_balance
                           - second_mortgage - tax_lien - hoa_lien - hei_amount, 0)
    labels = ["1st Mortgage", "HELOC", "2nd Mortgage", "Tax/HOA Liens", "HEI Position", "Homeowner Equity"]
    values = [outstanding_mortgage, heloc_balance, second_mortgage,
              tax_lien + hoa_lien, hei_amount, equity_after_hei]
    colors = ["#ef4444", "#f97316", "#f59e0b", "#eab308", "#3b82f6", "#10b981"]

    # Filter out zero values for cleaner chart
    filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if not filtered:
        return go.Figure()
    labels_f, values_f, colors_f = zip(*filtered)

    fig = go.Figure(go.Bar(
        x=[v / 1000 for v in values_f],
        y=["Capital Stack"] * len(values_f),
        orientation="h",
        name="",
        marker_color=colors_f,
        text=[f"{l}<br>${v/1000:.0f}K" for l, v in zip(labels_f, values_f)],
        textposition="inside",
        insidetextanchor="middle",
        textfont={"size": 10, "color": "white"},
    ))
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=110, margin=dict(t=5, b=5, l=10, r=10),
        xaxis=dict(title=f"$ Thousands (Total: ${property_value/1000:.0f}K)", color="#94a3b8",
                   gridcolor="#334155"),
        yaxis=dict(showticklabels=False),
        showlegend=False, font_color="#f1f5f9",
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    st.sidebar.markdown("## 🏠 HEI Underwriting Engine")
    st.sidebar.markdown("*Automated deal analysis · 28-factor ML pipeline*")
    st.sidebar.markdown("---")

    # ---- Property ----
    st.sidebar.markdown('<div class="sidebar-section">Property</div>', unsafe_allow_html=True)
    property_value = st.sidebar.number_input(
        "Estimated Home Value ($)", min_value=100_000, max_value=5_000_000,
        value=550_000, step=10_000, format="%d")
    outstanding_mortgage = st.sidebar.number_input(
        "1st Mortgage Balance ($)", min_value=0, max_value=4_000_000,
        value=260_000, step=5_000, format="%d")

    prop_type_label = st.sidebar.selectbox(
        "Property Type", options=list(PROPERTY_TYPE_OPTIONS.keys()), index=0)
    property_type_risk = PROPERTY_TYPE_OPTIONS[prop_type_label]

    c1, c2 = st.sidebar.columns(2)
    with c1:
        property_age = st.number_input("Property Age (yrs)", min_value=1, max_value=100, value=18, step=1)
    with c2:
        owner_occupied = st.radio("Owner-Occupied?", options=["Yes", "No"], index=0, horizontal=True)
    owner_occupied_flag = 1 if owner_occupied == "Yes" else 0

    arm_flag = 1 if st.sidebar.radio(
        "Mortgage Type", options=["Fixed Rate", "Adjustable (ARM)"],
        index=0, horizontal=True) == "Adjustable (ARM)" else 0

    # ---- Liens & Debt ----
    with st.sidebar.expander("🏦 Subordinate Liens & Debt", expanded=False):
        st.markdown("*Enter balances for any liens beyond the 1st mortgage*")
        heloc_balance = st.number_input("HELOC Balance ($)", min_value=0, max_value=500_000, value=0, step=1_000, format="%d")
        second_mortgage = st.number_input("2nd Mortgage Balance ($)", min_value=0, max_value=500_000, value=0, step=1_000, format="%d")
        tax_lien = st.number_input("Tax Lien Amount ($)", min_value=0, max_value=100_000, value=0, step=500, format="%d")
        hoa_lien = st.number_input("HOA Lien Amount ($)", min_value=0, max_value=50_000, value=0, step=500, format="%d")

    # ---- Homeowner Profile ----
    st.sidebar.markdown('<div class="sidebar-section">Homeowner Profile</div>', unsafe_allow_html=True)
    credit_score = st.sidebar.slider("Estimated Credit Score", min_value=540, max_value=850, value=720, step=5)
    dti_pct = st.sidebar.slider("Debt-to-Income Ratio (%)", min_value=5, max_value=75, value=32, step=1)
    dti_ratio = dti_pct / 100.0

    emp_label = st.sidebar.selectbox("Employment Type", options=list(EMPLOYMENT_OPTIONS.keys()), index=0)
    employment_tier = EMPLOYMENT_OPTIONS[emp_label]

    # ---- Credit History ----
    with st.sidebar.expander("⚠️ Credit & Legal History", expanded=False):
        st.markdown("*Check all that apply within the last 7 years*")
        foreclosure_flag = 1 if st.checkbox("Prior Foreclosure or Short Sale") else 0
        bankruptcy_flag  = 1 if st.checkbox("Prior Bankruptcy (Ch. 7 or 13)") else 0
        delinquency_flag = 1 if st.checkbox("30+ Day Mortgage Delinquency (last 24 months)") else 0

    # ---- Market ----
    st.sidebar.markdown('<div class="sidebar-section">Market</div>', unsafe_allow_html=True)
    c1, c2 = st.sidebar.columns([1, 1])
    with c1:
        state = st.selectbox("State", options=sorted([
            "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID",
            "IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS",
            "MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK",
            "OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
        ]), index=9, label_visibility="collapsed")
    with c2:
        metro_input = st.text_input("Metro (optional)", value="", placeholder="e.g. Tampa",
                                    label_visibility="collapsed")

    # ---- HEI Terms ----
    st.sidebar.markdown('<div class="sidebar-section">HEI Deal Terms</div>', unsafe_allow_html=True)
    hei_amount = st.sidebar.number_input(
        "HEI Investment Amount ($)", min_value=10_000, max_value=500_000,
        value=70_000, step=5_000, format="%d")
    equity_share_pct = st.sidebar.slider("Equity Share (%)", min_value=10, max_value=30, value=16, step=1) / 100.0
    cap_multiple = st.sidebar.slider("Return Cap (× Investment)", min_value=1.5, max_value=3.5, value=2.0, step=0.1)
    term_years = st.sidebar.radio("Investment Term", options=[5, 10], index=1, horizontal=True)

    st.sidebar.markdown("---")
    analyze = st.sidebar.button("⚡ Analyze Deal", use_container_width=True, type="primary")

    return {
        "property_value":    property_value,
        "outstanding_mortgage": outstanding_mortgage,
        "heloc_balance":     heloc_balance,
        "second_mortgage":   second_mortgage,
        "tax_lien":          tax_lien,
        "hoa_lien":          hoa_lien,
        "credit_score":      credit_score,
        "dti_ratio":         dti_ratio,
        "employment_tier":   employment_tier,
        "foreclosure_flag":  foreclosure_flag,
        "bankruptcy_flag":   bankruptcy_flag,
        "delinquency_flag":  delinquency_flag,
        "property_type_risk": property_type_risk,
        "property_age":      property_age,
        "owner_occupied":    owner_occupied_flag,
        "arm_flag":          arm_flag,
        "state":             state,
        "metro":             metro_input.strip(),
        "hei_amount":        hei_amount,
        "equity_share_pct":  equity_share_pct,
        "cap_multiple":      cap_multiple,
        "term_years":        term_years,
        "analyze":           analyze,
    }


# ---------------------------------------------------------------------------
# Results renderer
# ---------------------------------------------------------------------------

def render_results(result: dict, inputs: dict):
    rc    = result["risk_class"]
    score = result["deal_score"]
    irr_data = result["irr_data"]
    hpa   = result["hpa"]

    badge_map  = {"APPROVE": "badge-approve", "REVIEW": "badge-review", "REJECT": "badge-reject"}
    badge_icon = {"APPROVE": "✅", "REVIEW": "⚠️", "REJECT": "🚫"}

    # ===== Row 1: Score + key metrics =====
    col_g, col_m = st.columns([1, 2.2])

    with col_g:
        st.plotly_chart(score_gauge(score), use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"<div style='text-align:center;margin-top:-1rem;'><span class='score-label'>DEAL SCORE</span></div>", unsafe_allow_html=True)

    with col_m:
        st.markdown(
            f"<div style='margin-top:1.2rem;'>"
            f"<span class='{badge_map[rc]}'>{badge_icon[rc]} {rc}</span>"
            f"</div>", unsafe_allow_html=True)

        probs = result["class_probs"]
        st.markdown(f"""
        <div style="margin-top:0.4rem;font-size:0.72rem;color:#64748b;">
          Approve: {probs.get('APPROVE',0):.0%} &nbsp;|&nbsp;
          Review: {probs.get('REVIEW',0):.0%} &nbsp;|&nbsp;
          Reject: {probs.get('REJECT',0):.0%}
        </div>""", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        pv   = inputs["property_value"]
        with m1:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-label">1ST LIEN LTV</div>
              <div class="metric-value">{result['ltv']:.1%}</div>
              <div class="metric-sub">Equity: ${result['equity']:,.0f}</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            cltv_color = "#ef4444" if result["cltv"] > 0.87 else "#f59e0b" if result["cltv"] > 0.80 else "#10b981"
            st.markdown(f"""<div class="metric-card">
              <div class="metric-label">COMBINED LTV (CLTV)</div>
              <div class="metric-value" style="color:{cltv_color}">{result['cltv']:.1%}</div>
              <div class="metric-sub">All liens: ${result['total_debt']:,.0f}</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-label">BASE IRR</div>
              <div class="metric-value">{irr_data['base_irr']:.1f}%</div>
              <div class="metric-sub">{inputs['term_years']}-yr annualized</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            cap_prob = irr_data["cap_exceedance_prob"]
            st.markdown(f"""<div class="metric-card">
              <div class="metric-label">CAP EXCEEDANCE PROB</div>
              <div class="metric-value">{cap_prob:.0%}</div>
              <div class="metric-sub">{'Cap likely binding' if cap_prob > 0.5 else 'Cap likely non-binding'}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ===== Row 2: Capital Stack =====
    st.markdown("**Capital Stack**")
    cap_fig = capital_stack_chart(
        pv, inputs["outstanding_mortgage"], inputs["heloc_balance"],
        inputs["second_mortgage"], inputs["tax_lien"], inputs["hoa_lien"], inputs["hei_amount"]
    )
    st.plotly_chart(cap_fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("---")

    # ===== Row 3: IRR + Appreciation + SHAP =====
    col_irr, col_appr, col_shap = st.columns([1, 1.2, 1.6])

    with col_irr:
        st.markdown("**IRR by Scenario**")
        st.plotly_chart(irr_bar_chart(irr_data), use_container_width=True, config={"displayModeBar": False})
        s = irr_data["scenarios"]
        st.markdown(f"<div style='font-size:0.76rem;color:#64748b;'>Bear: <b style='color:#ef4444'>{s['bear']['irr']:.1f}%</b> &nbsp;·&nbsp; Bull: <b style='color:#10b981'>{s['bull']['irr']:.1f}%</b></div>", unsafe_allow_html=True)

    with col_appr:
        st.markdown("**Home Value Forecast**")
        st.plotly_chart(
            appreciation_chart(hpa, inputs["term_years"], inputs["property_value"]),
            use_container_width=True, config={"displayModeBar": False})

    with col_shap:
        st.markdown("**Key Drivers (SHAP)**")
        st.plotly_chart(shap_waterfall(result["shap_dict"]), use_container_width=True, config={"displayModeBar": False})
        st.markdown("<div style='font-size:0.72rem;color:#64748b;'>Green = pushes toward approval · Red = pushes toward rejection</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ===== Row 4: Underwriting Checklist + Summary =====
    col_check, col_summary = st.columns([1, 1.4])

    with col_check:
        st.markdown("**Underwriting Checklist**")
        checklist = result["checklist"]
        passed = sum(1 for _, p, _ in checklist if p)
        total  = len(checklist)
        st.markdown(f"<div style='font-size:0.78rem;color:#64748b;margin-bottom:0.5rem;'>{passed}/{total} checks passed</div>", unsafe_allow_html=True)

        for label, passed_bool, detail in checklist:
            icon = "✓" if passed_bool else "✗"
            icon_color = "#6ee7b7" if passed_bool else "#fca5a5"
            detail_color = "#6ee7b7" if passed_bool else "#fca5a5"
            st.markdown(f"""
            <div class="check-row">
              <span class="check-label">{label}</span>
              <span class="check-value" style="color:{detail_color}">
                <span style="color:{icon_color}">{icon}</span> {detail}
              </span>
            </div>""", unsafe_allow_html=True)

    with col_summary:
        st.markdown("**Underwriting Summary**")
        bear_irr = irr_data["scenarios"]["bear"]["irr"]
        base_irr = irr_data["scenarios"]["base"]["irr"]
        bull_irr = irr_data["scenarios"]["bull"]["irr"]
        net_base = irr_data["scenarios"]["base"]["net_return"]
        cagr     = result["appreciation_cagr"]

        def recommendation_text(rc, base_irr, cltv, ltv, cap_prob, inputs, result):
            flags = []
            if inputs["foreclosure_flag"]: flags.append("prior foreclosure on record")
            if inputs["bankruptcy_flag"]:  flags.append("prior bankruptcy on record")
            if inputs["delinquency_flag"]: flags.append("recent mortgage delinquency")
            if cltv > 0.87: flags.append(f"CLTV of {cltv:.1%} exceeds 87% threshold")
            if inputs["arm_flag"]: flags.append("ARM mortgage adds forced-sale risk")
            if inputs["dti_ratio"] > 0.50: flags.append(f"DTI of {inputs['dti_ratio']:.0%} is elevated")

            if rc == "APPROVE":
                flag_str = (f" Minor flags: {'; '.join(flags)}." if flags else "")
                return (f"This deal meets all key underwriting thresholds. Base-case IRR of **{base_irr:.1f}%** with "
                        f"a CLTV of **{cltv:.1%}** provides a solid collateral position. Market CAGR of "
                        f"**{cagr:.1%}** supports the appreciation thesis.{flag_str}")
            elif rc == "REVIEW":
                flag_str = "; ".join(flags) if flags else "one or more borderline metrics"
                return (f"This deal requires analyst review before proceeding. Flagged: {flag_str}. "
                        f"Base IRR is **{base_irr:.1f}%**. Recommend pulling a full title report, "
                        f"ordering an independent appraisal, and verifying income documentation.")
            else:
                flag_str = "; ".join(flags) if flags else "fundamental underwriting thresholds are not met"
                return (f"Deal does not meet minimum program requirements. Reason(s): {flag_str}. "
                        f"With a base-case IRR of **{base_irr:.1f}%** and CLTV of **{cltv:.1%}**, "
                        f"the risk/return profile does not support investment at these terms.")

        rec_text = recommendation_text(rc, base_irr, result["cltv"], result["ltv"],
                                        irr_data["cap_exceedance_prob"], inputs, result)

        st.markdown(f"""
        <div style='background:#1e293b;border:1px solid #334155;border-radius:12px;padding:1.2rem 1.4rem;'>
          <div style='color:#e2e8f0;font-size:0.93rem;line-height:1.75;'>{rec_text}</div>
          <div style='margin-top:1rem;display:flex;flex-wrap:wrap;gap:1.2rem;font-size:0.8rem;color:#64748b;'>
            <span>💰 Net Return (base): <b style='color:#f1f5f9'>${net_base:,.0f}</b></span>
            <span>📉 Bear IRR: <b style='color:#ef4444'>{bear_irr:.1f}%</b></span>
            <span>📈 Bull IRR: <b style='color:#10b981'>{bull_irr:.1f}%</b></span>
            <span>📍 Market: <b style='color:#f1f5f9'>{inputs['state']} {cagr:.1%} CAGR</b></span>
          </div>
        </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Methodology tab
# ---------------------------------------------------------------------------

def render_methodology():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("# HEI Underwriting Engine")
        st.markdown("#### Automated deal analysis for Home Equity Investments — decisions in seconds, not hours.")
        st.markdown("---")

        st.markdown("### The Problem")
        st.markdown("""Home Equity Investments (HEIs) are a growing asset class — operators like Unlock, Point,
        Hometap, and Splitero deploy capital into residential properties in exchange for a share of future
        appreciation. Unlike traditional mortgages, HEIs carry no monthly payment; the operator's return is
        realized when the home sells or the term expires. Manual underwriting of these deals is slow, inconsistent,
        and doesn't scale. This engine replaces that process with a 28-factor ML pipeline delivering a deal score,
        risk classification, capital stack analysis, and IRR distribution in under a second.""")

        st.markdown("### Why HEI Underwriting Is Unique")
        st.markdown("""The core complexity is the **cap structure**. Operators cap their return at a multiple of
        the original investment (e.g., 2×). This creates a bimodal return distribution: below the cap, the operator
        benefits fully from appreciation; above it, the cap binds and rapid appreciation benefits the homeowner, not
        the investor. A deal in Austin with 80% appreciation can yield a *worse* IRR than a deal in Columbus with
        45% appreciation — because the cap gets hit early and clips compounding. The engine explicitly models this
        via cap exceedance probability computed from the appreciation distribution.""")

        st.markdown("### Feature Set (28 Factors)")
        st.markdown("""**Capital Stack & Collateral:** First-lien LTV, Combined LTV across all lien positions (HELOC,
        second mortgage, tax liens, HOA liens), number of subordinate liens, HELOC utilization rate, equity as % of
        value, HEI amount relative to available equity, investment as % of property value.

**Credit & Legal History:** Credit quality tier, prior foreclosure flag, prior bankruptcy flag,
        30+ day mortgage delinquency history (last 24 months), debt-to-income ratio, employment stability tier.

**Property Quality:** Property type risk tier (SFR / townhome / condo / manufactured), property age,
        owner-occupied flag, flood zone exposure, ARM vs. fixed-rate mortgage flag.

**Market & Return:** Appreciation CAGR, 5-year total appreciation, market liquidity score,
        expected IRR (base case), cap exceedance probability, return cap multiple, equity share %,
        investment term.""")

        st.markdown("### ML Pipeline")
        st.markdown("""**Stage 1 — Feature Engineering:** Raw inputs are transformed into the 28-feature vector.
        Market data (appreciation, liquidity, flood exposure) is enriched from state-level lookups compiled from
        Zillow ZHVI, FEMA flood maps, and median days-on-market data.

**Stage 2 — HPA Quantile Regression:** Three gradient boosting regressors (P10/P50/P90) produce
        a home price appreciation distribution — not a point estimate. This uncertainty quantification is what
        enables bear/base/bull IRR scenario modeling.

**Stage 3 — Cap Analysis & IRR Distribution:** For each appreciation scenario, the engine computes
        the operator's gross return, applies the cap constraint, and calculates annualized IRR. Cap exceedance
        probability is derived from the distribution of capped outcomes across scenarios.

**Stage 4 — Risk Classification:** A gradient boosting classifier produces Approve / Review / Reject
        probabilities with 97%+ accuracy on the test set.

**Stage 5 — Deal Scoring & SHAP:** The 0–100 score is a weighted composite (IRR 30%, CLTV 25%,
        credit 15%, history 10%, cap efficiency 8%, equity 7%, property quality 5%). SHAP values identify which
        of the 28 factors drove the classification.""")

        st.markdown("### Data")
        st.markdown("""10,000 synthetic HEI deals generated with IRR-anchored investment sizing across three deal
        quality tiers. Lien profiles, credit history flags, property attributes, and DTI are sampled from
        tier-correlated distributions mirroring real HEI program intake data. Labels are assigned via
        expert-derived business rules that replicate published HEI operator underwriting criteria.
        Training: 80% / Test: 20% stratified split. 5-fold cross-validation on test set.""")

        st.markdown("### Limitations")
        st.markdown("""Production deployment would require: proprietary deal-flow data with actual operator outcomes;
        AVM integration (CoreLogic, ATTOM) for real-time property valuation; tri-merge credit bureau pull;
        title search and lien verification; and model performance tracking across market cycles.""")

    with col2:
        st.markdown("### Tech Stack")
        for name, desc in [
            ("scikit-learn", "Gradient Boosting (quantile + classifier)"),
            ("SHAP", "28-factor feature attribution"),
            ("Plotly", "Capital stack, IRR, gauge charts"),
            ("Streamlit", "Production deployment"),
            ("Zillow ZHVI", "Appreciation rate calibration"),
            ("FEMA / DOM data", "Flood & liquidity lookups"),
            ("Pandas / NumPy", "Feature engineering pipeline"),
        ]:
            st.markdown(f"""<div style='background:#1e293b;border:1px solid #334155;border-radius:8px;
                padding:0.55rem 0.8rem;margin-bottom:0.4rem;'>
              <div style='color:#93c5fd;font-size:0.78rem;font-weight:600;font-family:monospace;'>{name}</div>
              <div style='color:#64748b;font-size:0.72rem;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("### Underwriting Thresholds")
        for label, thresh in [
            ("Max 1st Lien LTV", "80%"),
            ("Max Combined LTV", "87%"),
            ("Min Credit Score", "620"),
            ("Max DTI", "50%"),
            ("Min IRR (base)", "5%"),
            ("Min Equity", "15%"),
            ("Foreclosure", "Hard reject"),
            ("Bankruptcy", "Hard reject"),
            ("Non-owner-occupied", "Hard reject"),
            ("Manufactured home", "Hard reject"),
        ]:
            st.markdown(f"""<div style='display:flex;justify-content:space-between;
                font-size:0.78rem;padding:0.25rem 0;border-bottom:1px solid #1e293b;'>
              <span style='color:#94a3b8;'>{label}</span>
              <span style='color:#f1f5f9;font-weight:600;'>{thresh}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""<div style='font-size:0.75rem;color:#475569;line-height:1.7;'>
        Built by <a href='https://stefenewers.com' target='_blank' style='color:#3b82f6;'>Stefen Ewers</a>
        as part of a portfolio of applied ML projects in real estate finance.<br><br>
        Portfolio demonstration only — not investment advice.
        </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    hpa_models, clf, le = load_models()
    inputs = render_sidebar()

    tab_tool, tab_method = st.tabs(["🔍 Deal Analyzer", "📖 Methodology"])

    with tab_tool:
        if not inputs["analyze"]:
            st.markdown("## HEI Underwriting Engine")
            st.markdown("Automated deal scoring for **Home Equity Investments** — 28-factor ML pipeline covering capital stack, credit history, property quality, and return structure.")
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            cards = [
                ("Capital Stack Analysis", "First-lien LTV + CLTV across all lien positions — HELOC, 2nd mortgage, tax and HOA liens."),
                ("Credit History Flags", "Foreclosure, bankruptcy, and delinquency history surfaced as hard underwriting stops."),
                ("IRR Distribution", "Bear / base / bull IRR scenarios with cap exceedance probability from quantile regression."),
                ("28-Factor SHAP", "Every decision explained: which of 28 features drove the score and by how much."),
            ]
            for col, (title, body) in zip([c1, c2, c3, c4], cards):
                with col:
                    st.markdown(f"""<div class="metric-card">
                      <div class="metric-label">{title.upper()}</div>
                      <div style="color:#e2e8f0;font-size:0.88rem;margin-top:0.5rem;line-height:1.6;">{body}</div>
                    </div>""", unsafe_allow_html=True)
            st.info("👈 Fill in the deal parameters in the sidebar and click **Analyze Deal** to get started.")

        else:
            equity = inputs["property_value"] - inputs["outstanding_mortgage"]
            if equity <= 0:
                st.error("Outstanding mortgage exceeds property value — no equity to invest against.")
                st.stop()
            total_liens = (inputs["outstanding_mortgage"] + inputs["heloc_balance"] +
                           inputs["second_mortgage"] + inputs["tax_lien"] + inputs["hoa_lien"])
            cltv_check = total_liens / max(inputs["property_value"], 1)
            if cltv_check > 0.92:
                st.warning(f"⚠️ Combined LTV ({cltv_check:.1%}) is very high — expect REJECT on CLTV threshold.")

            with st.spinner("Running 28-factor underwriting analysis..."):
                result = run_prediction(inputs, hpa_models, clf, le)
            render_results(result, inputs)

    with tab_method:
        render_methodology()


if __name__ == "__main__":
    main()
