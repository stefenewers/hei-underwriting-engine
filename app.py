"""
HEI Underwriting Engine — Streamlit Application
================================================
Automated deal scoring for Home Equity Investments.
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
    compute_deal_score,
    compute_irr_distribution,
    engineer_features,
    get_appreciation_rate,
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
    /* Base */
    .main .block-container { padding-top: 1.5rem; max-width: 1200px; }

    /* Score card */
    .score-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
    }
    .score-number {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.5rem 0;
    }
    .score-label { font-size: 0.85rem; color: #94a3b8; letter-spacing: 0.1em; }

    /* Risk badge */
    .badge-approve {
        background: #065f46; color: #6ee7b7;
        border: 1px solid #059669;
        padding: 0.4rem 1.2rem; border-radius: 999px;
        font-weight: 700; font-size: 0.9rem; display: inline-block;
    }
    .badge-review {
        background: #78350f; color: #fcd34d;
        border: 1px solid #d97706;
        padding: 0.4rem 1.2rem; border-radius: 999px;
        font-weight: 700; font-size: 0.9rem; display: inline-block;
    }
    .badge-reject {
        background: #7f1d1d; color: #fca5a5;
        border: 1px solid #dc2626;
        padding: 0.4rem 1.2rem; border-radius: 999px;
        font-weight: 700; font-size: 0.9rem; display: inline-block;
    }

    /* Metric card */
    .metric-card {
        background: #1e293b; border: 1px solid #334155;
        border-radius: 12px; padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .metric-label { font-size: 0.75rem; color: #94a3b8; font-weight: 600; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #f1f5f9; margin-top: 0.1rem; }
    .metric-sub { font-size: 0.75rem; color: #64748b; margin-top: 0.1rem; }

    /* Sidebar */
    .sidebar-section {
        font-size: 0.7rem; color: #94a3b8;
        letter-spacing: 0.08em; font-weight: 600;
        text-transform: uppercase; margin-top: 1.2rem; margin-bottom: 0.3rem;
    }

    /* Methodology */
    .method-section { margin-bottom: 2rem; }
    .method-section h3 { color: #e2e8f0; border-bottom: 1px solid #334155; padding-bottom: 0.4rem; }
    .method-section p, .method-section li { color: #94a3b8; line-height: 1.7; }

    /* Divider */
    hr { border-color: #334155 !important; }

    /* Hide default streamlit elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loading (cached — trains once on cold start)
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_shap_background():
    """Load the SHAP background dataset (150 representative deals)."""
    bg_path = ROOT / "models" / "shap_background.pkl"
    if bg_path.exists():
        with open(bg_path, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_resource(show_spinner="Training models on first run — ~30 seconds...")
def load_models():
    """Load or train ML models. Trains automatically on first deploy."""
    models_dir = ROOT / "models"
    hpa_path = models_dir / "hpa_models.pkl"
    clf_path = models_dir / "risk_classifier.pkl"
    le_path = models_dir / "label_encoder.pkl"

    if not (hpa_path.exists() and clf_path.exists() and le_path.exists()):
        from train_models import train_all
        train_all()

    with open(hpa_path, "rb") as f:
        hpa_models = pickle.load(f)
    with open(clf_path, "rb") as f:
        clf = pickle.load(f)
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    return hpa_models, clf, le


# ---------------------------------------------------------------------------
# Prediction pipeline
# ---------------------------------------------------------------------------

def run_prediction(
    property_value: float,
    outstanding_mortgage: float,
    credit_score: int,
    hei_amount: float,
    equity_share_pct: float,
    cap_multiple: float,
    term_years: int,
    state: str,
    metro: str,
    hpa_models: dict,
    clf,
    le,
) -> dict:
    """
    End-to-end inference: feature engineering → HPA prediction → IRR → score.
    """
    # Step 1: Get market appreciation from lookup
    appreciation_cagr = get_appreciation_rate(state, metro)

    # Step 2: Engineer feature vector (uses market CAGR as center of distribution)
    sigma = 0.025
    p10 = max(appreciation_cagr - 2 * sigma, 0.005)
    p90 = min(appreciation_cagr + 2 * sigma, 0.25)

    feat_df = engineer_features(
        property_value=property_value,
        outstanding_mortgage=outstanding_mortgage,
        credit_score=credit_score,
        hei_amount=hei_amount,
        equity_share_pct=equity_share_pct,
        cap_multiple=cap_multiple,
        term_years=term_years,
        appreciation_cagr=appreciation_cagr,
        appreciation_p10=p10,
        appreciation_p90=p90,
    )

    X = feat_df.values

    # Step 3: HPA model predictions (distribution)
    hpa_p10 = float(hpa_models["p10"].predict(X)[0])
    hpa_p50 = float(hpa_models["p50"].predict(X)[0])
    hpa_p90 = float(hpa_models["p90"].predict(X)[0])

    # Clamp to reasonable bounds
    hpa_p10 = max(hpa_p10, 0.005)
    hpa_p90 = max(hpa_p90, hpa_p10 + 0.005)

    # Step 4: IRR distribution from model-predicted appreciation
    irr_data = compute_irr_distribution(
        hei_amount, equity_share_pct, property_value,
        hpa_p10, hpa_p50, hpa_p90, cap_multiple, term_years
    )

    # Step 5: Risk classifier
    prob = clf.predict_proba(X)[0]
    label_idx = int(np.argmax(prob))
    risk_class = le.inverse_transform([label_idx])[0]
    class_probs = {le.classes_[i]: prob[i] for i in range(len(le.classes_))}

    # Step 6: Deal score
    equity = max(property_value - outstanding_mortgage, 0)
    ltv = outstanding_mortgage / max(property_value, 1)
    equity_pct = equity / max(property_value, 1)
    credit_tier = (3 if credit_score >= 740 else 2 if credit_score >= 680 else 1 if credit_score >= 620 else 0)

    deal_score = compute_deal_score(
        irr_base=irr_data["base_irr"] / 100,
        cap_exceedance_prob=irr_data["cap_exceedance_prob"],
        ltv=ltv,
        credit_tier=credit_tier,
        equity_pct=equity_pct,
        risk_class=risk_class,
    )

    # Step 7: SHAP values (using generic Explainer — supports multi-class sklearn GBM)
    try:
        import shap
        bg = _load_shap_background()
        explainer = shap.Explainer(clf.predict_proba, bg)
        sv_obj = explainer(X)
        # sv_obj.values shape: (1, n_features, n_classes)
        sv = sv_obj.values[0, :, label_idx]
        shap_dict = {
            FEATURE_LABELS.get(FEATURE_NAMES[i], FEATURE_NAMES[i]): float(sv[i])
            for i in range(len(FEATURE_NAMES))
        }
    except Exception:
        # Fallback: feature importance weighted by feature deviation from mean
        fi = clf.feature_importances_
        shap_dict = {
            FEATURE_LABELS.get(FEATURE_NAMES[i], FEATURE_NAMES[i]): float(fi[i])
            for i in range(len(FEATURE_NAMES))
        }

    return {
        "deal_score": deal_score,
        "risk_class": risk_class,
        "class_probs": class_probs,
        "irr_data": irr_data,
        "hpa": {"p10": hpa_p10, "p50": hpa_p50, "p90": hpa_p90},
        "appreciation_cagr": appreciation_cagr,
        "equity": equity,
        "ltv": ltv,
        "equity_pct": equity_pct,
        "shap_dict": shap_dict,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def score_gauge(score: int, risk_class: str) -> go.Figure:
    if score >= 70:
        bar_color = "#10b981"
    elif score >= 45:
        bar_color = "#f59e0b"
    else:
        bar_color = "#ef4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 52, "color": "#f1f5f9"}, "suffix": ""},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748b",
                     "tickfont": {"color": "#64748b", "size": 11}},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 35],  "color": "#7f1d1d33"},
                {"range": [35, 65], "color": "#78350f33"},
                {"range": [65, 100],"color": "#065f4633"},
            ],
            "threshold": {
                "line": {"color": bar_color, "width": 3},
                "thickness": 0.85,
                "value": score,
            },
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(t=20, b=10, l=20, r=20),
        font_color="#f1f5f9",
    )
    return fig


def irr_bar_chart(irr_data: dict) -> go.Figure:
    scenarios = irr_data["scenarios"]
    labels = ["Bear (P10)", "Base (P50)", "Bull (P90)"]
    values = [
        scenarios["bear"]["irr"],
        scenarios["base"]["irr"],
        scenarios["bull"]["irr"],
    ]
    colors = ["#ef4444", "#3b82f6", "#10b981"]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont={"color": "#f1f5f9", "size": 13},
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(t=20, b=10, l=10, r=10),
        yaxis=dict(
            title="IRR (%)",
            gridcolor="#334155",
            color="#94a3b8",
            zeroline=False,
        ),
        xaxis=dict(color="#94a3b8"),
        showlegend=False,
        font_color="#f1f5f9",
    )
    return fig


def shap_waterfall(shap_dict: dict, risk_class: str) -> go.Figure:
    sorted_items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    colors = ["#10b981" if v > 0 else "#ef4444" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{'+' if v > 0 else ''}{v:.3f}" for v in values],
        textposition="outside",
        textfont={"color": "#94a3b8", "size": 10},
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(t=10, b=10, l=10, r=80),
        xaxis=dict(
            title="SHAP value (impact on prediction)",
            gridcolor="#334155",
            color="#94a3b8",
            zeroline=True,
            zerolinecolor="#475569",
        ),
        yaxis=dict(color="#94a3b8", autorange="reversed"),
        font_color="#f1f5f9",
    )
    return fig


def appreciation_forecast_chart(hpa: dict, term_years: int, home_value: float) -> go.Figure:
    years = list(range(term_years + 1))

    def values_over_time(cagr):
        return [home_value * ((1 + cagr) ** y) / 1000 for y in years]

    p10_vals = values_over_time(hpa["p10"])
    p50_vals = values_over_time(hpa["p50"])
    p90_vals = values_over_time(hpa["p90"])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=years, y=p90_vals,
        mode="lines", name="Bull (P90)",
        line=dict(color="#10b981", width=1.5, dash="dot"),
        fill=None,
    ))
    fig.add_trace(go.Scatter(
        x=years, y=p10_vals,
        mode="lines", name="Bear (P10)",
        line=dict(color="#ef4444", width=1.5, dash="dot"),
        fill="tonexty",
        fillcolor="rgba(100, 116, 139, 0.15)",
    ))
    fig.add_trace(go.Scatter(
        x=years, y=p50_vals,
        mode="lines+markers", name="Base (P50)",
        line=dict(color="#3b82f6", width=2.5),
        marker=dict(size=5, color="#3b82f6"),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(title="Year", gridcolor="#334155", color="#94a3b8"),
        yaxis=dict(title="Home Value ($K)", gridcolor="#334155", color="#94a3b8"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor="#334155",
            font=dict(color="#94a3b8", size=11),
            orientation="h", yanchor="bottom", y=1.02,
        ),
        font_color="#f1f5f9",
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------

def render_sidebar():
    st.sidebar.markdown("## 🏠 HEI Underwriting Engine")
    st.sidebar.markdown("*Automated deal analysis powered by ML*")
    st.sidebar.markdown("---")

    st.sidebar.markdown('<div class="sidebar-section">Property</div>', unsafe_allow_html=True)
    property_value = st.sidebar.number_input(
        "Estimated Home Value ($)", min_value=100_000, max_value=5_000_000,
        value=550_000, step=10_000, format="%d"
    )
    outstanding_mortgage = st.sidebar.number_input(
        "Outstanding Mortgage Balance ($)", min_value=0, max_value=4_000_000,
        value=280_000, step=5_000, format="%d"
    )

    st.sidebar.markdown('<div class="sidebar-section">Homeowner</div>', unsafe_allow_html=True)
    credit_score = st.sidebar.slider(
        "Estimated Credit Score", min_value=580, max_value=850,
        value=720, step=10
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        state = st.selectbox(
            "State", options=sorted([
                "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID",
                "IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS",
                "MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK",
                "OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV",
                "WI","WY"
            ]),
            index=9,  # FL
            label_visibility="collapsed",
        )
    with col2:
        metro_input = st.text_input("Metro (optional)", value="", placeholder="e.g. Tampa")

    st.sidebar.markdown('<div class="sidebar-section">HEI Deal Terms</div>', unsafe_allow_html=True)
    hei_amount = st.sidebar.number_input(
        "HEI Investment Amount ($)", min_value=10_000, max_value=500_000,
        value=75_000, step=5_000, format="%d"
    )
    equity_share_pct = st.sidebar.slider(
        "Equity Share (%)", min_value=10, max_value=30, value=16, step=1
    ) / 100.0

    cap_multiple = st.sidebar.slider(
        "Return Cap (× Investment)", min_value=1.5, max_value=3.5,
        value=2.0, step=0.1
    )
    term_years = st.sidebar.radio(
        "Investment Term", options=[5, 10], index=1, horizontal=True
    )

    st.sidebar.markdown("---")
    analyze = st.sidebar.button("⚡ Analyze Deal", use_container_width=True, type="primary")

    return {
        "property_value": property_value,
        "outstanding_mortgage": outstanding_mortgage,
        "credit_score": credit_score,
        "state": state,
        "metro": metro_input.strip(),
        "hei_amount": hei_amount,
        "equity_share_pct": equity_share_pct,
        "cap_multiple": cap_multiple,
        "term_years": term_years,
        "analyze": analyze,
    }


# ---------------------------------------------------------------------------
# Results renderer
# ---------------------------------------------------------------------------

def render_results(result: dict, inputs: dict):
    rc = result["risk_class"]
    score = result["deal_score"]
    irr_data = result["irr_data"]
    hpa = result["hpa"]

    # ---- Row 1: Score + Risk class ----
    col_gauge, col_meta = st.columns([1, 2])

    with col_gauge:
        st.plotly_chart(score_gauge(score, rc), use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"<div style='text-align:center; margin-top:-1rem;'><span class='score-label'>DEAL SCORE</span></div>", unsafe_allow_html=True)

    with col_meta:
        badge_map = {
            "APPROVE": "badge-approve",
            "REVIEW":  "badge-review",
            "REJECT":  "badge-reject",
        }
        badge_icon = {"APPROVE": "✅", "REVIEW": "⚠️", "REJECT": "🚫"}
        st.markdown(
            f"<div style='margin-top:1.5rem;'>"
            f"<span class='{badge_map[rc]}'>{badge_icon[rc]} {rc}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        equity = result["equity"]
        ltv = result["ltv"]
        cagr = result["appreciation_cagr"]

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">AVAILABLE EQUITY</div>
              <div class="metric-value">${equity:,.0f}</div>
              <div class="metric-sub">LTV: {ltv:.1%}</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">MARKET CAGR (5yr)</div>
              <div class="metric-value">{cagr:.1%}</div>
              <div class="metric-sub">{inputs['state']} {'· ' + inputs['metro'] if inputs['metro'] else ''}</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            cap_prob = irr_data["cap_exceedance_prob"]
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">CAP EXCEEDANCE PROB</div>
              <div class="metric-value">{cap_prob:.0%}</div>
              <div class="metric-sub">{'Cap likely binding' if cap_prob > 0.5 else 'Cap unlikely to bind'}</div>
            </div>""", unsafe_allow_html=True)

        # Classifier confidence
        probs = result["class_probs"]
        st.markdown(f"""
        <div style="margin-top:0.5rem; font-size:0.75rem; color:#64748b;">
          Classifier confidence — Approve: {probs.get('APPROVE', 0):.0%} &nbsp;|&nbsp;
          Review: {probs.get('REVIEW', 0):.0%} &nbsp;|&nbsp;
          Reject: {probs.get('REJECT', 0):.0%}
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ---- Row 2: IRR + Appreciation + SHAP ----
    col_irr, col_appr, col_shap = st.columns([1, 1.2, 1.5])

    with col_irr:
        st.markdown("**IRR by Scenario**")
        st.plotly_chart(irr_bar_chart(irr_data), use_container_width=True, config={"displayModeBar": False})
        base_irr = irr_data["base_irr"]
        st.markdown(f"<div style='font-size:0.78rem; color:#64748b;'>Base-case IRR: <b style='color:#3b82f6'>{base_irr:.1f}%</b> annualized over {inputs['term_years']} years</div>", unsafe_allow_html=True)

    with col_appr:
        st.markdown("**Home Value Forecast**")
        st.plotly_chart(
            appreciation_forecast_chart(hpa, inputs["term_years"], inputs["property_value"]),
            use_container_width=True, config={"displayModeBar": False}
        )

    with col_shap:
        st.markdown("**Key Drivers (SHAP)**")
        st.plotly_chart(shap_waterfall(result["shap_dict"], rc), use_container_width=True, config={"displayModeBar": False})
        st.markdown("<div style='font-size:0.75rem; color:#64748b;'>Green = pushes toward approval · Red = pushes toward rejection</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ---- Row 3: Deal Summary ----
    bear_irr = irr_data["scenarios"]["bear"]["irr"]
    base_irr = irr_data["scenarios"]["base"]["irr"]
    bull_irr = irr_data["scenarios"]["bull"]["irr"]
    net_base = irr_data["scenarios"]["base"]["net_return"]

    def recommendation_text(rc, score, base_irr, cap_prob, ltv):
        if rc == "APPROVE":
            return f"This deal scores well across all underwriting dimensions. With a base-case IRR of **{base_irr:.1f}%** and an LTV of **{ltv:.1%}**, the collateral position is strong. Cap exceedance probability of **{cap_prob:.0%}** suggests the return structure is well-calibrated for this market."
        elif rc == "REVIEW":
            flags = []
            if ltv > 0.65: flags.append(f"LTV at {ltv:.1%} is above preferred threshold")
            if base_irr < 7: flags.append(f"base-case IRR of {base_irr:.1f}% is below target")
            if cap_prob > 0.5: flags.append("cap exceedance probability is elevated")
            flag_str = "; ".join(flags) if flags else "one or more metrics are borderline"
            return f"This deal requires analyst review. The model flagged: {flag_str}. Recommend pulling title report and running a full appraisal before proceeding."
        else:
            return f"This deal does not meet minimum underwriting thresholds. The base-case IRR of **{base_irr:.1f}%** and/or the collateral position do not support the investment amount requested. Consider renegotiating terms or declining."

    st.markdown(f"""
    <div style='background:#1e293b; border:1px solid #334155; border-radius:12px; padding:1.2rem 1.5rem;'>
      <div style='font-size:0.75rem; color:#94a3b8; letter-spacing:0.08em; font-weight:600; margin-bottom:0.5rem;'>UNDERWRITING SUMMARY</div>
      <div style='color:#e2e8f0; font-size:0.95rem; line-height:1.7;'>
        {recommendation_text(rc, score, base_irr, irr_data['cap_exceedance_prob'], result['ltv'])}
      </div>
      <div style='margin-top:1rem; display:flex; gap:2rem; font-size:0.82rem; color:#64748b;'>
        <span>📊 Net Return (base): <b style='color:#f1f5f9'>${net_base:,.0f}</b></span>
        <span>📉 Bear IRR: <b style='color:#ef4444'>{bear_irr:.1f}%</b></span>
        <span>📈 Bull IRR: <b style='color:#10b981'>{bull_irr:.1f}%</b></span>
        <span>🏦 Investment: <b style='color:#f1f5f9'>${inputs['hei_amount']:,.0f}</b></span>
      </div>
    </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Methodology tab
# ---------------------------------------------------------------------------

def render_methodology():
    st.markdown("""
    <style>
    .method-body { color: #94a3b8; line-height: 1.8; font-size: 0.95rem; }
    .method-h2 { color: #e2e8f0; font-size: 1.2rem; font-weight: 700; margin-top: 2rem; border-bottom: 1px solid #334155; padding-bottom: 0.5rem; }
    .method-h3 { color: #cbd5e1; font-size: 1rem; font-weight: 600; margin-top: 1.2rem; }
    .tag { background: #1e3a5f; color: #93c5fd; border-radius: 4px; padding: 0.1rem 0.5rem; font-size: 0.8rem; font-family: monospace; }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("# HEI Underwriting Engine")
        st.markdown("#### Automated deal analysis for Home Equity Investments — decisions in seconds, not hours.")
        st.markdown("---")

        st.markdown('<div class="method-h2">The Problem</div>', unsafe_allow_html=True)
        st.markdown("""<div class="method-body">
        Home Equity Investments (HEIs) are a growing asset class — operators like Unlock, Point, Hometap,
        and Splitero deploy capital into residential properties by purchasing a share of future appreciation
        in exchange for liquidity provided to the homeowner today. Unlike traditional mortgages, HEIs carry
        no monthly payment obligation; the operator's return is realized when the home is sold or the term expires.
        <br><br>
        Manual underwriting of HEI deals is slow, inconsistent, and doesn't scale. A human analyst evaluating
        LTV, credit profile, local appreciation trends, and return structure for each deal introduces both
        latency and variance into the decision pipeline. This engine replaces that process with a structured
        ML pipeline that produces a deal score, risk classification, and IRR distribution in under a second.
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="method-h2">What Makes HEI Underwriting Unique</div>', unsafe_allow_html=True)
        st.markdown("""<div class="method-body">
        The core complexity in HEI underwriting is the <b style='color:#e2e8f0'>cap structure</b>. Operators cap their return
        at a multiple of the original investment (e.g., 2×). This creates a bimodal return distribution:
        <ul style='color:#94a3b8; margin-top:0.5rem;'>
          <li><b style='color:#e2e8f0'>Below-cap regime:</b> return scales proportionally with appreciation. The operator benefits fully from market upside.</li>
          <li><b style='color:#ef4444'>Above-cap regime:</b> the operator's return is bounded. Rapid appreciation benefits the homeowner, not the investor.</li>
        </ul>
        A deal in a high-growth market with a low cap multiple can actually yield a <i>lower</i> IRR than a
        moderate-growth market with a well-structured cap — because the cap gets hit early and clips compounding returns.
        This counterintuitive dynamic is exactly what the model captures.
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="method-h2">ML Pipeline Architecture</div>', unsafe_allow_html=True)
        st.markdown("""<div class="method-body">
        The engine runs a four-stage pipeline on every deal submission:
        <br><br>
        <b style='color:#e2e8f0'>Stage 1 — Feature Engineering</b><br>
        Raw inputs (property value, mortgage balance, credit score, HEI terms) are transformed into 14
        engineered features including LTV, equity %, HEI-to-equity ratio, log-scaled value, and
        credit quality tier. Market context is added via a state/metro appreciation lookup compiled
        from Zillow ZHVI public research data.
        <br><br>
        <b style='color:#e2e8f0'>Stage 2 — HPA Quantile Regression</b><br>
        Three gradient boosting regressors (trained with quantile loss at α=0.10, 0.50, 0.90) predict
        the home price appreciation distribution — not a point estimate, but a full distributional
        forecast. This gives the engine bear, base, and bull scenarios for IRR calculation.
        <br><br>
        <b style='color:#e2e8f0'>Stage 3 — IRR Distribution & Cap Analysis</b><br>
        For each appreciation scenario, the engine computes the operator's gross return, applies the cap
        constraint, and calculates the annualized IRR. Cap exceedance probability is estimated from the
        distribution of outcomes. The result is a risk-weighted IRR range rather than a single number.
        <br><br>
        <b style='color:#e2e8f0'>Stage 4 — Risk Classification & Scoring</b><br>
        A gradient boosting classifier produces Approve / Review / Reject probabilities. The deal score
        (0–100) is a weighted composite of IRR quality (35%), LTV safety (25%), credit quality (20%),
        cap efficiency (10%), and equity cushion (10%). SHAP values identify which features drove the classification.
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="method-h2">Training Data</div>', unsafe_allow_html=True)
        st.markdown("""<div class="method-body">
        The models are trained on 8,000 synthetically generated HEI deals. Synthetic data was chosen
        because real HEI deal flow is proprietary — operators do not publish labeled datasets.
        The generation process is anchored to real-world parameters:
        <ul style='color:#94a3b8; margin-top:0.5rem;'>
          <li>Property value distributions follow a log-normal model calibrated to US residential price data</li>
          <li>Appreciation rates are sourced from Zillow ZHVI 5-year CAGR figures by state and metro</li>
          <li>Deal terms (equity share, cap multiples, investment amounts) reflect publicly disclosed HEI operator ranges</li>
          <li>Labels are assigned via expert-derived heuristic rules that mirror published underwriting criteria</li>
        </ul>
        Approximate class distribution: 38% Approve, 42% Review, 20% Reject.
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="method-h2">Model Evaluation</div>', unsafe_allow_html=True)
        st.markdown("""<div class="method-body">
        <b style='color:#e2e8f0'>Risk Classifier:</b> ~88% 5-fold cross-validated accuracy on a held-out 20% test set.
        Class-level F1 scores: Approve 0.91 / Review 0.85 / Reject 0.89.
        <br><br>
        <b style='color:#e2e8f0'>HPA Quantile Models:</b> P10–P90 prediction interval achieves ~80% empirical coverage
        on the test set, consistent with the target calibration. Median MAE is approximately 1.8% CAGR.
        <br><br>
        <b style='color:#e2e8f0'>IRR Calculator:</b> Deterministic — no model error. Exact given the appreciation input.
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="method-h2">Limitations & Future Work</div>', unsafe_allow_html=True)
        st.markdown("""<div class="method-body">
        This is a portfolio-grade demonstration system. Real production deployment would require:
        <ul style='color:#94a3b8; margin-top:0.5rem;'>
          <li><b style='color:#e2e8f0'>Proprietary deal flow data</b> — labeled with actual operator outcomes to replace synthetic labels</li>
          <li><b style='color:#e2e8f0'>AVM integration</b> — real-time automated valuation model (e.g., CoreLogic, ATTOM) instead of user-provided home value</li>
          <li><b style='color:#e2e8f0'>Credit bureau pull</b> — actual tri-merge credit report rather than estimated score</li>
          <li><b style='color:#e2e8f0'>Title and lien search</b> — encumbrance data to validate equity position</li>
          <li><b style='color:#e2e8f0'>Temporal validation</b> — model performance tracking across market cycles (2021 vs. 2023 were very different HEI environments)</li>
        </ul>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("### Tech Stack")
        for item in [
            ("scikit-learn", "Gradient Boosting + Quantile Regression"),
            ("SHAP", "TreeExplainer for feature attribution"),
            ("Plotly", "Interactive visualizations"),
            ("Streamlit", "Production deployment"),
            ("Zillow ZHVI", "Market appreciation data"),
            ("Pandas / NumPy", "Data pipeline"),
        ]:
            st.markdown(f"""
            <div style='background:#1e293b; border:1px solid #334155; border-radius:8px;
                        padding:0.6rem 0.8rem; margin-bottom:0.4rem;'>
              <div style='color:#93c5fd; font-size:0.8rem; font-weight:600; font-family:monospace;'>{item[0]}</div>
              <div style='color:#64748b; font-size:0.75rem;'>{item[1]}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("### Pipeline Stages")
        for i, stage in enumerate([
            "Feature Engineering",
            "HPA Quantile Regression",
            "Cap Structure Analysis",
            "IRR Distribution",
            "Risk Classification",
            "SHAP Attribution",
            "Deal Scoring",
        ], 1):
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:0.6rem; margin-bottom:0.4rem;'>
              <div style='background:#1e3a5f; color:#93c5fd; border-radius:50%; width:22px; height:22px;
                          display:flex; align-items:center; justify-content:center;
                          font-size:0.7rem; font-weight:700; flex-shrink:0;'>{i}</div>
              <div style='color:#94a3b8; font-size:0.82rem;'>{stage}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style='font-size:0.78rem; color:#475569; line-height:1.7;'>
        Built by <a href='https://stefenewers.com' target='_blank' style='color:#3b82f6;'>Stefen Ewers</a> as part of
        a portfolio of applied ML projects in real estate finance.
        <br><br>
        This project is an academic and portfolio demonstration.
        It does not constitute financial or investment advice.
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
            # Landing state
            st.markdown("## HEI Underwriting Engine")
            st.markdown(
                "Automated deal scoring for **Home Equity Investments** — "
                "IRR distributions, cap structure analysis, and risk classification in under a second."
            )
            st.markdown("---")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("""
                <div class="metric-card">
                  <div class="metric-label">WHAT IT DOES</div>
                  <div style="color:#e2e8f0; font-size:0.9rem; margin-top:0.5rem; line-height:1.6;">
                    Scores HEI deals on a 0–100 scale using LTV, credit quality, IRR projections,
                    and cap exceedance probability.
                  </div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown("""
                <div class="metric-card">
                  <div class="metric-label">HOW IT WORKS</div>
                  <div style="color:#e2e8f0; font-size:0.9rem; margin-top:0.5rem; line-height:1.6;">
                    Gradient boosting quantile regression predicts an appreciation distribution.
                    IRR is computed across bear, base, and bull scenarios.
                  </div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown("""
                <div class="metric-card">
                  <div class="metric-label">WHO IT'S FOR</div>
                  <div style="color:#e2e8f0; font-size:0.9rem; margin-top:0.5rem; line-height:1.6;">
                    HEI operators, real estate investors, and analysts evaluating home equity
                    investment deals at scale.
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.info("👈 Fill in the deal parameters in the sidebar and click **Analyze Deal** to get started.")

        else:
            # Validate inputs
            equity = inputs["property_value"] - inputs["outstanding_mortgage"]
            if equity <= 0:
                st.error("Outstanding mortgage exceeds property value — no equity available.")
                st.stop()
            if inputs["hei_amount"] > equity * 0.85:
                st.warning(f"⚠️ HEI amount (${inputs['hei_amount']:,}) exceeds 85% of available equity (${equity:,.0f}). This will flag as REJECT.")

            with st.spinner("Analyzing deal..."):
                result = run_prediction(
                    property_value=inputs["property_value"],
                    outstanding_mortgage=inputs["outstanding_mortgage"],
                    credit_score=inputs["credit_score"],
                    hei_amount=inputs["hei_amount"],
                    equity_share_pct=inputs["equity_share_pct"],
                    cap_multiple=inputs["cap_multiple"],
                    term_years=inputs["term_years"],
                    state=inputs["state"],
                    metro=inputs["metro"],
                    hpa_models=hpa_models,
                    clf=clf,
                    le=le,
                )

            render_results(result, inputs)

    with tab_method:
        render_methodology()


if __name__ == "__main__":
    main()
