# Karachi 72-Hour AQI Forecaster

**Author:** Muskan Pawan | IBA Karachi | 10Pearls Shine Internship 2026  
**Live Dashboard:** https://karachiaqipredictor-hdeggxz4ambqcjvhfdnqvt.streamlit.app/  
**Hopsowork:** https://eu-west.cloud.hopsworks.ai/p/33022/view
**Domain:** Data Sciences | Air Quality Index (AQI) Forecasting  

---

## What This Project Does

This is a **100% serverless, end-to-end MLOps pipeline** that predicts Karachi's Air Quality Index (AQI) 72 hours into the future. There is no always-on server. Three GitHub Actions workflows run automatically in the cloud, keeping the system alive and self-improving without any manual intervention.

The system:
1. **Fetches** live atmospheric data from the Open-Meteo Copernicus satellite API every hour
2. **Stores** engineered features in a Hopsworks cloud Feature Store
3. **Retrains** all candidate ML models daily and promotes the best one to a Model Registry
4. **Monitors** live prediction accuracy daily to detect concept drift
5. **Serves** a public Streamlit dashboard that shows a 72-hour AQI forecast with EPA health alerts

The target variable is **PM2.5 concentration 24 hours forward** (`pm2_5[t+24]`), converted to AQI using the US EPA piecewise linear formula.

---

## Repository Structure

```
KarachiAQIPredictor/
│
├── .devcontainer/
│   └── devcontainer.json          # VS Code dev container config (Python 3.11)
│
├── .github/
│   └── workflows/
│       ├── hourly_pipeline.yml    # Runs every hour — fetches & stores features
│       ├── daily_monitor.yml      # Runs every day — checks model health (MAE/MSE)
│       └── daily_training.yml     # Runs every day — retrains & promotes best model
│
├── EDA/
│   └── EDA10PearlsAQI.ipynb      # 8-step exploratory data analysis notebook
│
├── ModelTraining/
│   ├── TrainingModels10Pearls.ipynb   # 7-phase training, tuning & SHAP notebook
│   └── trainingmodels10pearls.py      # Script version of the training pipeline
│
├── WebApp/
│   └── app.py                     # Streamlit dashboard (production web app)
│
├── model_cache/                   # Auto-created at runtime; stores downloaded .pkl
│
├── .env                           # LOCAL ONLY — never committed (holds API key)
├── .gitignore                     # Excludes .env, venv/, model_cache/, __pycache__
├── feature_pipeline.py            # Hourly: fetch → engineer features → push to Hopsworks
├── fetch_data.py                  # One-time historical backfill (2021-01-01 → 2026-05-27)
├── karachi_raw_aqi.csv            # Raw backfill output (33,432 clean hourly records)
├── model_monitor.py               # Daily: fetch actuals → predict → compute MAE/MSE
├── training_pipeline.py           # Daily: retrain all models → champion-challenger → promote
└── requirements.txt               # Streamlit Cloud production dependencies
```

---

## Environment Setup (Local Development)

### Prerequisites

- Python 3.11
- A free [Hopsworks account](https://app.hopsworks.ai) — grab your API key from Project Settings → API Keys
- Git

### Step 1 — Clone the repository

```bash
git clone https://github.com/Musknn/KarachiAQIPredictor.git
cd KarachiAQIPredictor
```

### Step 2 — Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt once activated.

### Step 3 — Install all dependencies

```bash
pip install --upgrade pip
pip install hopsworks pandas scikit-learn lightgbm torch joblib pyarrow requests python-dotenv streamlit shap notebook
```

> **Note for Streamlit Cloud deployment only:** the `requirements.txt` in the repo root is a minimal subset used by Streamlit Cloud (`streamlit hopsworks pandas requests scikit-learn lightgbm joblib python-dotenv`). For full local development including training and EDA, use the command above.

### Step 4 — Create your `.env` file

Create a file named `.env` in the **repository root** (same level as `feature_pipeline.py`):

```
HOPSWORKS_API_KEY=your_actual_api_key_here
```

This file is listed in `.gitignore` and will never be committed. All scripts read this key via:

```python
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.environ["HOPSWORKS_API_KEY"]
```

---

## How to Run Each Component

### 1. Historical Data Backfill (`fetch_data.py`)

Run **once** to populate `karachi_raw_aqi.csv` with historical data from 2021 to present. This is the training dataset source.

```bash
python fetch_data.py
```

**What it does:**  
Calls the Open-Meteo Air Quality Historical API for Karachi (`lat=24.8607, lon=67.0011`) for these 9 variables: `pm10`, `pm2_5`, `carbon_monoxide`, `nitrogen_dioxide`, `sulphur_dioxide`, `ozone`, `aerosol_optical_depth`, `dust`, `uv_index`. Saves raw output to `karachi_raw_aqi.csv` (47,352 rows before cleaning).

---

### 2. EDA Notebook (`EDA/EDA10PearlsAQI.ipynb`)

Open in Jupyter and run all cells sequentially.

```bash
cd EDA
jupyter notebook EDA10PearlsAQI.ipynb
```

**What it does (8 steps in order):**

| Step | Action | Outcome |
|------|--------|---------|
| 1 | Drop features with 0 valid records | 18 → 11 features (removes 7 pollen columns never measured in Karachi) |
| 2–3 | Two-dimensional missing data audit + drop systemic outage rows | 47,352 → 33,432 records |
| 4 | Drop `methane` (58.4% missing) | Prevents imputation bias |
| 5 | Forward-fill minor sensor gaps (<2% per feature) | Dataset mathematically complete |
| 6 | Drop `uv_index_clear_sky` (corr=0.98 with `uv_index`) | Removes multicollinearity |
| 7 | Engineer time features: `hour`, `day`, `month`, `day_of_week` | +4 features |
| 8 | Engineer momentum deltas: `*_change` for each of the 9 pollutants via `.diff().fillna(0)` | +9 features → **22 predictors total** |
After EDA, the clean feature dataset is uploaded to the **Hopsworks Feature Store** (Feature Group: `karachi_aqi_features`, version 2).

---

### 3. Feature Pipeline — Manual Run (`feature_pipeline.py`)

This script is also triggered **every hour** automatically by `hourly_pipeline.yml`. To run it locally:

```bash
python feature_pipeline.py
```

**What it does:**

1. GET request to Open-Meteo Air Quality API with `past_days=1&forecast_days=0` for Karachi coordinates
2. Applies the same feature engineering as EDA (temporal features + momentum deltas)
3. Authenticates with Hopsworks using `HOPSWORKS_API_KEY` from `.env`
4. Calls `.insert(df)` on Feature Group `karachi_aqi_features` v2 to append the 24 new rows
5. Wrapped in `try/except` — errors are logged without crashing

**Cron schedule in GitHub Actions:** `18 * * * *` (runs at minute 18 of every hour)

---

### 4. Model Training (`ModelTraining/TrainingModels10Pearls.ipynb`)

Open in Jupyter and run all cells sequentially. This is the full 7-phase training pipeline.

```bash
cd ModelTraining
jupyter notebook TrainingModels10Pearls.ipynb
```

**7-phase protocol:**

| Phase | What Happens |
|-------|-------------|
| 1 | Fetch features from Hopsworks Feature Store; sort chronologically; create target `y = pm2_5[t+24]`; 80/20 chronological split (no shuffle) → 26,726 train / 6,682 vaulted test |
| 2 | 10-fold TimeSeriesSplit baseline tournament: Ridge (RMSE 12.47), Random Forest (12.48), LightGBM (12.35), PyTorch (14.81) |
| 3 | RandomizedSearchCV over wide parameter grids (30 configs Ridge, 30 LightGBM, 20 RF, 15 PyTorch) |
| 4 | GridSearchCV over narrow micro-grids around Phase 3 winners → final configs |
| 5 | VotingRegressor ensemble = Ridge + LightGBM + Random Forest. *(Note: PyTorch DNN architecture was also trained and pushed to the Model Registry for completeness, but excluded from the final production ensemble to optimize inference speed and variance).* |
| 6 | Evaluate all 4 models on vaulted test set → Ensemble: RMSE=11.02, MAE=7.98, R²=0.35 |
| 7 | SHAP explainability — TreeExplainer on 2,000-sample subset; blended importance weighted by inverse RMSE |

**Final optimised hyperparameters:**
- **Ridge:** `alpha=900.0`
- **LightGBM:** `n_estimators=220, learning_rate=0.015, max_depth=2, num_leaves=12, subsample=0.65, colsample_bytree=0.75`
- **Random Forest:** `n_estimators=500, max_depth=3, min_samples_split=18, min_samples_leaf=12`

**Models stored in Hopsworks Model Registry (from notebook):**
- `karachi_ridge_aqi_final` — version 1
- `karachi_lgb_aqi_final` — version 1
- `karachi_ensemble_aqi_final` — version 1

The daily `training_pipeline.py` additionally registers versioned models under `karachi_ridge_aqi_daily`, `karachi_lgbm_aqi_daily`, `karachi_rf_aqi_daily`, `karachi_ensemble_aqi_daily`, and promotes the lowest-MSE champion to `karachi_aqi_production` — the alias loaded by the Streamlit dashboard.

---

### 5. Daily Training & Best-Model Selection (`training_pipeline.py`)

This script is triggered **every day at 02:00 UTC** by `daily_training.yml`. To run locally:

```bash
python training_pipeline.py
```

**Champion selection protocol:**

1. Fetch all records from Hopsworks Feature Store (includes latest hourly appends)
2. Sort chronologically; create 24-hour forward-shifted target `y = pm2_5[t+24]`
3. Apply an **80/20 chronological split** (no shuffling) → 26,726 train rows / 6,682 test rows
4. Train all four architectures (Ridge, LightGBM, Random Forest, VotingEnsemble) on the training split
5. Evaluate all four on the held-out 20% test set → MSE, RMSE, MAE, R²
6. Select the model with the **lowest MSE** as today's champion
7. Register all four models under their own registry names; additionally register the champion under `karachi_aqi_production` (latest version)
8. Save a structured JSON log `model_artifacts/daily_log_YYYY-MM-DD.json` with all metrics

**Cron schedule in GitHub Actions:** `0 2 * * *` (02:00 UTC = 07:00 PKT)

---

### 6. Daily Model Monitor (`model_monitor.py`)

This script is triggered **every day at 19:18 UTC** by `daily_monitor.yml`. To run locally:

```bash
python model_monitor.py
```

**What it does:**

1. GET request to Open-Meteo with `past_days=1&forecast_days=0` → 24 actual hourly PM2.5 readings for yesterday
2. Applies identical feature engineering (temporal + momentum deltas); drops first row with `.iloc[1:]`
3. Downloads latest version of `karachi_aqi_production` from Hopsworks Model Registry → `model_cache/*.pkl`
4. Calls `ensemble_model.predict(X_test)` on yesterday's features
5. Computes `MSE` and `MAE` vs. yesterday's actual PM2.5
6. Prints structured health report to Actions log
7. If `MAE > 10 µg/m³` → prints concept drift warning; otherwise prints health confirmation
8. Validates feature schema via `ensemble_model.feature_names_in_` — mismatch causes explicit failure

**Drift alert threshold:** MAE > 10 µg/m³  
**Cron schedule in GitHub Actions:** `18 19 * * *` (19:18 UTC)

---

### 7. Streamlit Dashboard (`WebApp/app.py`)

To run the dashboard locally:

```bash
cd WebApp
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

**What happens on each button press ("Run 3-Day Forecast"):**

1. Authenticate with Hopsworks → resolve latest version of `karachi_aqi_production` → cache today-stamped `.pkl` to `model_cache/`
2. GET request to Open-Meteo with `past_days=1&forecast_days=3&timezone=Asia%2FKarachi` → 96 hours of data (past 24h sliced off after momentum calculations to leave 72 future hours).
3. Apply feature engineering (same pipeline as training)
4. `ensemble_model.predict(X_inference)` → 72 raw PM2.5 predictions
5. Pass each through `calculate_epa_aqi()` (piecewise linear EPA formula) → 72 AQI scores
6. Compute `max_aqi` across all 72 predictions → display alert banner:
   - `max_aqi > 150` → `st.error()` — red banner, hazardous, mask advisory
   - `max_aqi > 50` → `st.warning()` — amber banner, moderate conditions
   - Otherwise → `st.success()` — green banner, clean air
7. Two-column layout: "Right Now" card (current AQI + PM2.5) | "72-Hour Outlook" (3 daily summary cards)
8. `st.line_chart()` — 72-hour AQI trend line chart
9. `st.expander()` — full raw inference dataframe with all 72 rows and all 22 feature columns

**EPA AQI Breakpoints implemented in `calculate_epa_aqi()`:**

| PM2.5 (µg/m³) | AQI Range | Category |
|---|---|---|
| 0.0 – 12.0 | 0 – 50 | Good |
| 12.1 – 35.4 | 51 – 100 | Moderate |
| 35.5 – 55.4 | 101 – 150 | Unhealthy for Sensitive Groups |
| 55.5 – 150.4 | 151 – 200 | Unhealthy |
| 150.5 – 250.4 | 201 – 300 | Very Unhealthy |
| 250.5 – 500.4 | 301 – 500 | Hazardous |

---

## GitHub Actions — Automated Workflows

All three workflows live in `.github/workflows/`. They require one GitHub Repository Secret:

**Setting up the secret:**  
`Repository → Settings → Secrets and variables → Actions → New repository secret`  
Name: `HOPSWORKS_API_KEY` | Value: your Hopsworks API key

Each workflow injects it at runtime via:
```yaml
env:
  HOPSWORKS_API_KEY: ${{ secrets.HOPSWORKS_API_KEY }}
```
And the Python scripts read it via `os.environ["HOPSWORKS_API_KEY"]`. The value is **never** printed or stored in logs — GitHub masks it automatically.

### Workflow Summary

| Property | `hourly_pipeline.yml` | `daily_monitor.yml` | `daily_training.yml` |
|---|---|---|---|
| **Script** | `feature_pipeline.py` | `model_monitor.py` | `training_pipeline.py` |
| **Cron** | `18 * * * *` | `18 19 * * *` | `0 2 * * *` |
| **Role** | Data ingestion | Drift detection | Model improvement |
| **Reads from** | Open-Meteo API | Open-Meteo API | Hopsworks Feature Store |
| **Writes to** | Hopsworks Feature Store | Actions log | Hopsworks Model Registry |
| **Decision** | — | Alert if MAE > 10 | Promote lowest-MSE champion daily |
| **Runner** | `ubuntu-latest` | `ubuntu-latest` | `ubuntu-latest` |
| **Manual trigger** | `workflow_dispatch` | `workflow_dispatch` | `workflow_dispatch` |

### Manually triggering a workflow

Go to `GitHub → Actions tab → select workflow → Run workflow button → Run workflow`.

### Install steps inside each workflow (for reference)

```yaml
# hourly_pipeline.yml
pip install hopsworks pandas requests pyarrow

# daily_monitor.yml
pip install hopsworks pandas requests scikit-learn lightgbm pyarrow

# daily_training.yml
pip install hopsworks pandas scikit-learn lightgbm joblib pyarrow
```

---

## Full System Data Flow

```
Open-Meteo Copernicus Satellite API
         │
         │  every hour (GitHub Actions: hourly_pipeline.yml)
         ▼
feature_pipeline.py
  → fetches past 24h of raw pollutant data
  → engineers temporal + momentum features
  → pushes to Hopsworks Feature Store
         │
         ├──────────────────────────────────────────────────────┐
         │  daily at 02:00 UTC (daily_training.yml)             │  daily at 19:18 UTC
         ▼                                                       │  (daily_monitor.yml)
training_pipeline.py                                             ▼
  → fetches all Feature Store data               model_monitor.py
  → trains Ridge / LightGBM / RF / Ensemble       → fetches yesterday's actuals
  → evaluates on 80/20 chronological split         → loads karachi_aqi_production
  → promotes lowest-MSE model to                  → computes MAE / MSE
    karachi_aqi_production in Registry             → alerts if MAE > 10 µg/m³
         │
         │  (on button press, Streamlit Cloud)
         ▼
WebApp/app.py
  → loads latest champion model from Registry
  → fetches 72h forward forecast from Open-Meteo
  → engineers features → predicts PM2.5 × 72
  → converts to EPA AQI → displays dashboard
         │
         ▼
  Public URL: karachiaqipredictor-hdeggxz4ambqcjvhfdnqvt.streamlit.app
```

---

## Hopsworks Cloud Resources

| Resource | Name | Version |
|---|---|---|
| Feature Group | `karachi_aqi_features` | v2 |
| Registered Model | `karachi_ridge_aqi_final` | v1 (notebook baseline) |
| Registered Model | `karachi_lgb_aqi_final` | v1 (notebook baseline) |
| Registered Model | `karachi_ensemble_aqi_final` | v1 (notebook baseline) |
| Registered Model | `karachi_pytorch_aqi_final` | v1 (evaluated; excluded from ensemble) |
| Daily Model | `karachi_ridge_aqi_daily` | latest (auto-updated daily) |
| Daily Model | `karachi_lgbm_aqi_daily` | latest (auto-updated daily) |
| Daily Model | `karachi_rf_aqi_daily` | latest (auto-updated daily) |
| Daily Model | `karachi_ensemble_aqi_daily` | latest (auto-updated daily) |
| Production alias | `karachi_aqi_production` | latest (today's lowest-MSE champion) |

The Streamlit dashboard always loads `karachi_aqi_production` latest version.

---

## Feature Schema (22 features used by all models)

**Raw sensor features (9):** `pm10`, `pm2_5`, `carbon_monoxide`, `nitrogen_dioxide`, `sulphur_dioxide`, `ozone`, `aerosol_optical_depth`, `dust`, `uv_index`

**Temporal features (4):** `hour`, `day`, `month`, `day_of_week`

**Momentum delta features (9) — computed via `.diff().fillna(0)`:** `pm10_change`, `pm2_5_change`, `carbon_monoxide_change`, `nitrogen_dioxide_change`, `sulphur_dioxide_change`, `ozone_change`, `aerosol_optical_depth_change`, `dust_change`, `uv_index_change`

**Target variable (not a feature):** `pm2_5` shifted forward 24 hours: `y_t = pm2_5[t+24]`

The model's expected column order is enforced via `ensemble_model.feature_names_in_` at inference time.

---

## Open-Meteo API Reference

**Base URL:** `https://air-quality-api.open-meteo.com/v1/air-quality`

**Karachi coordinates:** `latitude=24.8607&longitude=67.0011&timezone=Asia%2FKarachi`

**Historical / feature pipeline call:**
```text
?latitude=24.8607&longitude=67.0011&past_days=1&forecast_days=0
&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,
        ozone,aerosol_optical_depth,dust,uv_index
&timezone=Asia%2FKarachi
```

Inference (dashboard) call:

```text
?latitude=24.8607&longitude=67.0011&past_days=1&forecast_days=3
&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,
        ozone,aerosol_optical_depth,dust,uv_index
&timezone=Asia%2FKarachi
```
(Note: `past_days=1` provides a 24-hour baseline so `.diff()` momentum calculations have a valid prior row for the very first forecast hour; after feature engineering the past 24 rows are sliced off, leaving 72 future rows for prediction).

No API key required for Open-Meteo.

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `KeyError: HOPSWORKS_API_KEY` | `.env` file missing or key name wrong | Create `.env` in repo root with `HOPSWORKS_API_KEY=...` |
| `ModuleNotFoundError: hopsworks` | venv not activated or packages not installed | Run `source venv/bin/activate` then `pip install hopsworks` |
| GitHub Action fails on secret | Secret not set in repo | Go to Settings → Secrets → Actions → add `HOPSWORKS_API_KEY` |
| Dashboard shows stale model | Model file cached locally | Delete `model_cache/` folder and re-run |
| `feature_names_in_` mismatch error | API column schema changed | Re-run `feature_pipeline.py` and retrain |
| Streamlit Cloud import error | `lightgbm` missing from `requirements.txt` | Add `lightgbm` to `requirements.txt` in repo root |
| Actions workflow shows 0 runs | Cron not yet triggered | Click "Run workflow" manually in the Actions tab |

---

## Deployment to Streamlit Cloud

The dashboard is already deployed. If you need to redeploy from scratch:

1. Push all code to GitHub (main branch)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select repo: `Musknn/KarachiAQIPredictor` | Branch: `main` | Main file path: `WebApp/app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   HOPSWORKS_API_KEY = "your_api_key_here"
   ```
5. Click Deploy. Streamlit Cloud reads `requirements.txt` from the repo root automatically.

---

*10Pearls Shine Internship 2026 — Karachi AQI Predictor*
