# HEI Underwriting Engine — Deployment Guide

## Stack
- **App:** Streamlit (Python)
- **Hosting:** Streamlit Community Cloud (free)
- **Link from:** stefenewers.com → iframe or button → Streamlit URL

---

## Step 1 — Push to GitHub

Create a new GitHub repo (e.g., `hei-underwriting-engine`) and push the full
`hei_underwriting_engine/` folder as the **root** of the repo.

The repo root should look like:
```
app.py
hei_engine.py
generate_data.py
train_models.py
requirements.txt
.streamlit/
  config.toml
models/
  hpa_models.pkl
  risk_classifier.pkl
  label_encoder.pkl
  shap_background.pkl
  feature_importance.csv
data/
  synthetic_deals.csv
```

**Important:** Commit the `models/` directory. The pkl files are ~1.2 MB total
and well within GitHub's limits. This avoids the cold-start training delay on
each deploy.

---

## Step 2 — Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in with GitHub
2. Click **New app**
3. Select your repo → branch `main` → main file: `app.py`
4. Click **Deploy**

Streamlit will install `requirements.txt` automatically. First deploy takes
~2–3 minutes. Subsequent deploys are instant.

Your app URL will be:
```
https://[your-username]-hei-underwriting-engine-app-[hash].streamlit.app
```

You can set a custom subdomain in Streamlit Cloud settings:
```
https://hei-underwriting.streamlit.app
```

---

## Step 3 — Link from stefenewers.com

Add a project card to your portfolio site pointing to the Streamlit URL.
Suggested link text / button copy:

```
Live Demo →  https://hei-underwriting.streamlit.app
```

If your site supports iframes (most do), you can embed it directly:
```html
<iframe
  src="https://hei-underwriting.streamlit.app?embed=true"
  height="900"
  style="width:100%; border:none; border-radius:12px;"
></iframe>
```

The `?embed=true` param hides the Streamlit toolbar for a cleaner look.

---

## Step 4 — LinkedIn Post Tips

This project has a strong narrative arc for LinkedIn:

- **Hook:** "Manual HEI underwriting takes hours. I automated it."
- **Technical credibility:** Quantile regression for appreciation uncertainty,
  cap exceedance probability, SHAP explainability
- **Business grounding:** Splitero, Unlock, Point, Hometap as real-world
  context; Zillow ZHVI data as the data source
- **Call to action:** Link to live demo + methodology write-up (in-app)

Suggested format: 3-paragraph post + 1 screenshot of the deal analyzer showing
a strong deal (APPROVE, score ~75, IRR chart + SHAP waterfall).

---

## Local Development

```bash
cd hei_underwriting_engine
pip install -r requirements.txt
streamlit run app.py
```

To retrain models from scratch:
```bash
python train_models.py
```

To regenerate synthetic data:
```bash
python generate_data.py
```
