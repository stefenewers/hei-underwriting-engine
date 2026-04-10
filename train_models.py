"""
HEI Underwriting Engine — Model Training
==========================================
Trains and persists the two core ML models:

  1. HPA Model (Gradient Boosting Quantile Regression)
     Predicts home price appreciation at P10, P50, and P90 percentiles,
     giving the engine a distribution rather than a point estimate.

  2. Risk Classifier (Gradient Boosting Classifier)
     Produces Approve / Review / Reject classification with
     calibrated probabilities and SHAP-ready feature importance.

All models are saved to ./models/ for loading by the Streamlit app.
"""

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    # Collateral / LTV
    "ltv",
    "cltv",
    "equity_pct",
    "hei_to_equity_ratio",
    "subordinate_lien_count",
    "heloc_utilization",
    # Creditworthiness
    "credit_tier",
    "foreclosure_flag",
    "bankruptcy_flag",
    "mortgage_delinquency_flag",
    "dti_ratio",
    "employment_stability_tier",
    # Property quality
    "property_type_risk",
    "property_age_normalized",
    "owner_occupied",
    "flood_zone_risk",
    "arm_flag",
    # Market & deal
    "log_property_value",
    "appreciation_cagr",
    "appreciation_5yr_total",
    "market_liquidity_score",
    "cap_multiple",
    "equity_share_pct",
    "term_years",
    # Return metrics
    "expected_irr_base",
    "cap_exceedance_prob",
    "log_investment",
    "investment_to_value_pct",
]

LABEL_ORDER = ["APPROVE", "REVIEW", "REJECT"]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_or_generate_data() -> pd.DataFrame:
    csv_path = DATA_DIR / "synthetic_deals.csv"
    if not csv_path.exists():
        print("Generating synthetic dataset...")
        import sys
        sys.path.insert(0, str(ROOT))
        from generate_data import generate_dataset
        df = generate_dataset()
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)

    print(f"Dataset loaded: {len(df)} records")
    print(df["label"].value_counts())
    return df


# ---------------------------------------------------------------------------
# HPA Quantile Model
# ---------------------------------------------------------------------------

def train_hpa_model(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Train three gradient boosting regressors at P10, P50, P90 quantiles.
    Returns a dict of {quantile: fitted_model}.
    """
    quantiles = {"p10": 0.10, "p50": 0.50, "p90": 0.90}
    models = {}

    for name, alpha in quantiles.items():
        print(f"  Training HPA model [{name}] (alpha={alpha})...")
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha,
            n_estimators=80,
            max_depth=4,
            learning_rate=0.10,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        )
        model.fit(X_train, y_train)
        models[name] = model

    return models


# ---------------------------------------------------------------------------
# Risk Classifier
# ---------------------------------------------------------------------------

def train_risk_classifier(X_train: np.ndarray, y_train: np.ndarray) -> GradientBoostingClassifier:
    """
    Train a gradient boosting multi-class classifier for deal risk.
    Produces calibrated probability scores used by the deal scorer.
    """
    print("  Training risk classifier...")
    clf = GradientBoostingClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.10,
        subsample=0.8,
        min_samples_leaf=8,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_risk_classifier(clf, X_test, y_test, label_encoder):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nRisk Classifier Accuracy: {acc:.3f}")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    ))

    # 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_test, y_test, cv=cv, scoring="accuracy")
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")


def evaluate_hpa_models(hpa_models, X_test, y_test):
    print("\nHPA Model MAE by quantile:")
    for name, model in hpa_models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"  {name}: MAE = {mae:.4f} ({mae*100:.2f}% CAGR)")

    # Prediction interval coverage
    p10_preds = hpa_models["p10"].predict(X_test)
    p90_preds = hpa_models["p90"].predict(X_test)
    coverage = np.mean((y_test >= p10_preds) & (y_test <= p90_preds))
    print(f"  P10–P90 coverage: {coverage:.2%} (target: ~80%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_all():
    print("=" * 60)
    print("HEI Underwriting Engine — Model Training")
    print("=" * 60)

    df = load_or_generate_data()

    # Feature matrix
    X = df[FEATURE_COLS].values.astype(float)

    # HPA target: use appreciation_cagr as the label
    y_hpa = df["appreciation_cagr"].values.astype(float)

    # Risk label
    le = LabelEncoder()
    le.fit(LABEL_ORDER)  # enforce consistent ordering
    y_risk = le.transform(df["label"].values)

    # Train / test split (stratified on risk label)
    X_train, X_test, y_train_hpa, y_test_hpa, y_train_risk, y_test_risk = (
        train_test_split(X, y_hpa, y_risk, test_size=0.2, stratify=y_risk, random_state=42)
    )

    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    # --------------- HPA Models ---------------
    print("\n[1/2] Training HPA Quantile Models...")
    hpa_models = train_hpa_model(X_train, y_train_hpa)
    evaluate_hpa_models(hpa_models, X_test, y_test_hpa)

    # --------------- Risk Classifier ---------------
    print("\n[2/2] Training Risk Classifier...")
    clf = train_risk_classifier(X_train, y_train_risk)
    evaluate_risk_classifier(clf, X_test, y_test_risk, le)

    # --------------- Save Artifacts ---------------
    print("\nSaving model artifacts...")

    with open(MODELS_DIR / "hpa_models.pkl", "wb") as f:
        pickle.dump(hpa_models, f, protocol=5)

    with open(MODELS_DIR / "risk_classifier.pkl", "wb") as f:
        pickle.dump(clf, f, protocol=5)

    with open(MODELS_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f, protocol=5)

    # Save SHAP background dataset (150 representative deals for KernelExplainer)
    bg = X_train[:150].astype(float)
    with open(MODELS_DIR / "shap_background.pkl", "wb") as f:
        pickle.dump(bg, f, protocol=5)

    # Save feature importance for SHAP fallback
    importances = clf.feature_importances_
    fi_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(MODELS_DIR / "feature_importance.csv", index=False)

    print("\nArtifacts saved:")
    for p in MODELS_DIR.iterdir():
        print(f"  {p.name}  ({p.stat().st_size / 1024:.1f} KB)")

    print("\nTraining complete.")
    return hpa_models, clf, le


if __name__ == "__main__":
    train_all()
