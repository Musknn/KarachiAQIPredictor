import os
import pandas as pd
import requests
import hopsworks
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
from dotenv import load_dotenv
import os
load_dotenv()
print("🔍 Waking up Model Monitoring Robot...")
print(f"   Run date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# ── 1. Fetch yesterday's actual data ──────────────────────────────────────────
url = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
    "?latitude=24.8607&longitude=67.0011"
    "&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,"
    "ozone,aerosol_optical_depth,dust,uv_index"
    "&timezone=Asia%2FKarachi&past_days=1&forecast_days=0"
)
response = requests.get(url).json()
df = pd.DataFrame(response["hourly"])
df["time"] = pd.to_datetime(df["time"])

# ── 2. Engineer features — mirrors training_pipeline.py exactly ───────────────
df["hour"]        = df["time"].dt.hour
df["day"]         = df["time"].dt.day
df["month"]       = df["time"].dt.month
df["day_of_week"] = df["time"].dt.dayofweek

sensor_cols = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide",
    "ozone", "aerosol_optical_depth", "dust", "uv_index",
]
for col in sensor_cols:
    df[f"{col}_change"] = df[col].diff().fillna(0)

# Forward-fill any NaNs in sensor readings (mirrors training cleanup)
df[sensor_cols] = df[sensor_cols].ffill().bfill()
df = df.dropna(subset=sensor_cols)

# ── 3. Connect to Hopsworks and load production model ─────────────────────────
print("🔐 Connecting to Hopsworks Model Registry...")
project  = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
mr       = project.get_model_registry()

# Always grab the highest version (same logic as app.py fix)
all_versions  = mr.get_models("karachi_aqi_production")
model_meta    = max(all_versions, key=lambda m: m.version)
print(f"   Loading production model v{model_meta.version}...")

save_folder = "model_cache"
os.makedirs(save_folder, exist_ok=True)
model_meta.download(save_folder)
ensemble_model = joblib.load(os.path.join(save_folder, "karachi_aqi_production.pkl"))

# ── 4. Retrieve expected feature columns from the trained model ───────────────
# Walk into VotingRegressor estimators if needed
def get_feature_names(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "estimators_"):
        for _, est in model.estimators_:
            names = get_feature_names(est)
            if names:
                return names
    return None

expected_columns = get_feature_names(ensemble_model)
if expected_columns is None:
    raise ValueError("Could not retrieve feature names from model. Ensure it was trained on a DataFrame.")

print(f"   Model expects {len(expected_columns)} features.")

# ── 5. Align inference data to model's expected columns ───────────────────────
missing_cols = [c for c in expected_columns if c not in df.columns]
if missing_cols:
    raise ValueError(f"Feature mismatch — columns missing from API data: {missing_cols}")

X_test   = df[expected_columns]
y_actual = df["pm2_5"].values
y_pred   = ensemble_model.predict(X_test)

# ── 6. Calculate and report metrics ───────────────────────────────────────────
mse  = mean_squared_error(y_actual, y_pred)
mae  = mean_absolute_error(y_actual, y_pred)

print("-" * 40)
print(f"📊 MODEL HEALTH REPORT — Last 24 Hours")
print(f"   Samples evaluated : {len(y_actual)}")
print(f"   Mean Squared Error: {mse:.2f}")
print(f"   Mean Absolute Error: {mae:.2f} µg/m³")
print("-" * 40)

# Drift thresholds (MAE on raw PM2.5 µg/m³)
if mae > 15:
    print("🚨 CRITICAL: Severe drift detected — consider immediate retraining.")
    exit(1)
elif mae > 10:
    print("⚠️  WARNING: Moderate drift detected — model accuracy is dropping.")
    exit(1)
else:
    print("✅ Model is healthy and performing well.")