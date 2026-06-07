import os
import pandas as pd
import requests
import hopsworks

print("🚀 Waking up GitHub Actions Robot...")
print("📡 Fetching latest Karachi AQI data from Open-Meteo...")

# 1. Fetch the last 24 hours of actual data
url = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=24.8607&longitude=67.0011&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,aerosol_optical_depth,dust,uv_index&timezone=Asia%2FKarachi&past_days=1&forecast_days=0"
response = requests.get(url).json()
df = pd.DataFrame(response['hourly'])
df['time'] = pd.to_datetime(df['time'])

# 2. Connect to Hopsworks using the secure GitHub Secret
print("🔐 Authenticating with Hopsworks Cloud...")
project = hopsworks.login(api_key_value=os.environ["HOPSWORKS_API_KEY"])
fs = project.get_feature_store()

# 3. Push to Feature Store
print("🗄️ Pushing fresh data to Feature Store...")
try:
    aqi_fg = fs.get_feature_group(name="karachi_aqi_features", version=2)
    aqi_fg.insert(df)
    print("✅ Successfully updated Hopsworks Feature Store!")
except Exception as e:
    print(f"⚠️ Note: Feature group connection issue (Check group name): {e}")