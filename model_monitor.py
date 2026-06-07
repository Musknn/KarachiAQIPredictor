import os
import pandas as pd
import requests
import hopsworks
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("🔍 Waking up Model Monitoring Robot...")

# 1. Fetch yesterday's ACTUAL data
url = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=24.8607&longitude=67.0011&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,aerosol_optical_depth,dust,uv_index&timezone=Asia%2FKarachi&past_days=1&forecast_days=0"
response = requests.get(url).json()
df = pd.DataFrame(response['hourly'])
df['time'] = pd.to_datetime(df['time'])

# 2. Engineer the exact same features
df['hour'] = df['time'].dt.hour
df['day'] = df['time'].dt.day
df['month'] = df['time'].dt.month
df['day_of_week'] = df['time'].dt.dayofweek

pollutants = ['pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'aerosol_optical_depth', 'dust', 'uv_index']
for p in pollutants:
    df[f'{p}_change'] = df[p].diff().fillna(0)

# Drop the first row since diff() creates a zero for the first calculation
df = df.iloc[1:].reset_index(drop=True)

# 3. Connect to Hopsworks and download the model
print("🔐 Connecting to Hopsworks Model Registry...")
project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
mr = project.get_model_registry()
model_file = mr.get_model("karachi_ensemble_aqi_final", version=2)

save_folder = "model_cache"
os.makedirs(save_folder, exist_ok=True)
model_file.download(save_folder)
ensemble_model = joblib.load(os.path.join(save_folder, "karachi_ensemble_aqi_final.pkl"))

# 4. Make Predictions vs. Actuals
expected_columns = list(ensemble_model.feature_names_in_)
X_test = df[expected_columns]
y_actual = df['pm2_5']  # The true value
y_pred = ensemble_model.predict(X_test) # The model's guess

# 5. Calculate Metrics
mse = mean_squared_error(y_actual, y_pred)
mae = mean_absolute_error(y_actual, y_pred)

print("-" * 30)
print(f"📊 MODEL HEALTH REPORT (Last 24 Hours)")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print("-" * 30)

# Optional safety alert: If MAE spikes above 10, the model might be breaking!
if mae > 10:
    print("⚠️ WARNING: Concept drift detected! Model accuracy is dropping.")
else:
    print("✅ Model is healthy and performing well.")