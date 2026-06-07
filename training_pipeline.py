import os
import json
import joblib
import datetime
import numpy as np
import pandas as pd
import hopsworks

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

# ─────────────────────────────────────────────
# 0. BOOT
# ─────────────────────────────────────────────
RUN_DATE = datetime.date.today().isoformat()
print(f"🚀 Daily Training Pipeline — {RUN_DATE}")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. CONNECT TO HOPSWORKS
# ─────────────────────────────────────────────
print("\n🔐 Authenticating with Hopsworks...")
project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
fs = project.get_feature_store()
mr = project.get_model_registry()
print("✅ Connected.")

# ─────────────────────────────────────────────
# 2. FETCH FEATURES FROM FEATURE STORE
# ─────────────────────────────────────────────
print("\n📦 Fetching features from Feature Store...")
aqi_fg = fs.get_feature_group(name="karachi_aqi_features", version=1)
df = aqi_fg.read()
print(f"   Raw rows fetched: {len(df)}")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING  (mirrors feature_pipeline.py)
# ─────────────────────────────────────────────
print("\n🔧 Engineering features...")

df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

# Drop rows where ALL sensor columns are NaN (systemic outages)
sensor_cols = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
               "sulphur_dioxide", "ozone", "aerosol_optical_depth",
               "dust", "uv_index"]
df = df.dropna(subset=sensor_cols, how="all")

# Forward-fill minor sensor glitches (<2% per feature)
df[sensor_cols] = df[sensor_cols].ffill()
df = df.dropna(subset=sensor_cols)

# Temporal features
df["hour"]        = df["time"].dt.hour
df["day"]         = df["time"].dt.day
df["month"]       = df["time"].dt.month
df["day_of_week"] = df["time"].dt.dayofweek

# Momentum delta features
for col in sensor_cols:
    df[f"{col}_change"] = df[col].diff().fillna(0)

print(f"   Rows after cleaning: {len(df)}")

# ─────────────────────────────────────────────
# 4. BUILD TRAIN/TEST SPLIT (chronological 80/20)
# ─────────────────────────────────────────────
print("\n✂️  Splitting data chronologically (80 / 20)...")

FEATURE_COLS = (
    sensor_cols
    + ["hour", "day", "month", "day_of_week"]
    + [f"{c}_change" for c in sensor_cols]
)
TARGET_COL = "pm2_5"

# 24-hour forward-shift target
df["target"] = df[TARGET_COL].shift(-24)
df = df.dropna(subset=["target"])

X = df[FEATURE_COLS].values
y = df["target"].values

split_idx = int(len(X) * 0.80)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"   Train: {len(X_train)} rows | Test: {len(X_test)} rows")

# ─────────────────────────────────────────────
# 5. TRAIN 3 MODELS WITH BEST KNOWN HYPERPARAMS
#    (tuned in the Jupyter notebook via two-round
#     RandomizedSearchCV → GridSearchCV)
# ─────────────────────────────────────────────
print("\n🏋️  Training three models with optimised hyperparameters...")

# --- 5a. Ridge (alpha = 900) ---
print("   [1/3] Ridge Regression  (alpha=900)...")
ridge_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  Ridge(alpha=900.0))
])
ridge_pipe.fit(X_train, y_train)

# --- 5b. LightGBM ---
print("   [2/3] LightGBM  (n_est=220, lr=0.015, depth=2, leaves=12, sub=0.65, col=0.75)...")
lgbm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  lgb.LGBMRegressor(
        n_estimators=220,
        learning_rate=0.015,
        max_depth=2,
        num_leaves=12,
        subsample=0.65,
        colsample_bytree=0.75,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    ))
])
lgbm_pipe.fit(X_train, y_train)

# --- 5c. Random Forest ---
print("   [3/3] Random Forest  (n_est=500, depth=3, min_split=18, min_leaf=12)...")
rf_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  RandomForestRegressor(
        n_estimators=500,
        max_depth=3,
        min_samples_split=18,
        min_samples_leaf=12,
        random_state=42,
        n_jobs=-1,
    ))
])
rf_pipe.fit(X_train, y_train)

# --- 5d. VotingRegressor Ensemble ---
print("   [+] Fitting VotingRegressor ensemble over the three models...")
# VotingRegressor needs estimators that are not already pipelines with a scaler
# so we pass the already-fitted pipelines as-is (they include their own scalers)
ensemble = VotingRegressor(
    estimators=[
        ("ridge",  ridge_pipe),
        ("lgbm",   lgbm_pipe),
        ("rf",     rf_pipe),
    ],
    n_jobs=-1,
)
ensemble.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 6. EVALUATE ALL 4 ON HELD-OUT TEST SET
# ─────────────────────────────────────────────
print("\n📊 Evaluating all models on held-out test set...")

def evaluate(name, model, X, y):
    preds = model.predict(X)
    mse   = mean_squared_error(y, preds)
    rmse  = np.sqrt(mse)
    mae   = mean_absolute_error(y, preds)
    r2    = r2_score(y, preds)
    print(f"   {name:<22}  MSE={mse:8.3f}  RMSE={rmse:6.3f}  MAE={mae:6.3f}  R²={r2:5.3f}")
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

results = {}
results["Ridge"]    = evaluate("Ridge",         ridge_pipe, X_test, y_test)
results["LightGBM"] = evaluate("LightGBM",      lgbm_pipe,  X_test, y_test)
results["RF"]       = evaluate("Random Forest", rf_pipe,    X_test, y_test)
results["Ensemble"] = evaluate("VotingEnsemble",ensemble,   X_test, y_test)

# ─────────────────────────────────────────────
# 7. DECIDE CHAMPION  (lowest MSE wins)
# ─────────────────────────────────────────────
model_map = {
    "Ridge":    (ridge_pipe,  "karachi_ridge_aqi_daily"),
    "LightGBM": (lgbm_pipe,   "karachi_lgbm_aqi_daily"),
    "RF":       (rf_pipe,     "karachi_rf_aqi_daily"),
    "Ensemble": (ensemble,    "karachi_ensemble_aqi_daily"),
}

champion_key = min(results, key=lambda k: results[k]["mse"])
champion_model, champion_name = model_map[champion_key]

print(f"\n🏆 Champion today: {champion_key}  (MSE = {results[champion_key]['mse']:.3f})")

# ─────────────────────────────────────────────
# 8. SAVE ALL 4 MODELS + REGISTER CHAMPION
#    Hopsworks v2 uses mr.python.create_model()
# ─────────────────────────────────────────────
print("\n💾 Saving models to disk and registering with Hopsworks Model Registry...")

os.makedirs("model_artifacts", exist_ok=True)

def save_and_register(key, model, reg_name, metrics, is_champion=False):
    path = f"model_artifacts/{reg_name}.pkl"
    joblib.dump(model, path)

    hw_model = mr.python.create_model(
        name=reg_name,
        metrics=metrics,
        description=(
            f"{'[CHAMPION] ' if is_champion else ''}"
            f"Karachi AQI daily model — {key} — trained {RUN_DATE}"
        ),
    )
    hw_model.save(path)
    print(f"   ✅ Registered: {reg_name}  (champion={is_champion})")

for key, (model, reg_name) in model_map.items():
    save_and_register(
        key, model, reg_name,
        metrics=results[key],
        is_champion=(key == champion_key),
    )

# Also register the champion under the fixed "production" name so the
# Streamlit dashboard always loads the same model name regardless of
# which architecture won today.
PROD_MODEL_NAME = "karachi_aqi_production"
prod_path = f"model_artifacts/{PROD_MODEL_NAME}.pkl"
joblib.dump(champion_model, prod_path)

hw_prod = mr.python.create_model(
    name=PROD_MODEL_NAME,
    metrics=results[champion_key],
    description=(
        f"Production model for Streamlit dashboard. "
        f"Today's champion: {champion_key} trained on {RUN_DATE}."
    ),
)
hw_prod.save(prod_path)
print(f"   ✅ Production alias registered: {PROD_MODEL_NAME} → {champion_key}")

# ─────────────────────────────────────────────
# 9. WRITE DAILY RESULTS LOG
# ─────────────────────────────────────────────
log = {
    "run_date":    RUN_DATE,
    "champion":    champion_key,
    "train_rows":  int(len(X_train)),
    "test_rows":   int(len(X_test)),
    "metrics":     results,
}

log_path = f"model_artifacts/daily_log_{RUN_DATE}.json"
with open(log_path, "w") as f:
    json.dump(log, f, indent=2)

print(f"\n📝 Daily log saved → {log_path}")
print("\n✅ Training pipeline complete.")
print("=" * 60)