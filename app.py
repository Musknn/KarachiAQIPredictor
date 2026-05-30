import streamlit as st
import hopsworks
import pandas as pd
import requests
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Load the secret securely in the background
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")

st.set_page_config(page_title="Karachi AQI Forecaster", page_icon="🌫️", layout="wide")

st.title("🌫️ Karachi 72-Hour Air Quality Forecast")
st.markdown("Powered by a Stacking Ensemble (LightGBM + Random Forest + Ridge)")

st.sidebar.header("⚙️ System Configuration")
st.sidebar.info("Secure connection to Cloud Registry enabled.")

# Run the pipeline
if st.sidebar.button("Run 3-Day Forecast"):
    # Check if the API key was successfully loaded from the .env file
    if not api_key:
        st.error("🚨 Configuration Error: API key not found in server environment. Please check your .env file.")
    else:
        with st.spinner("Connecting to Cloud Model Registry..."):
            try:
                # 1. Connect to Hopsworks and load the model intelligently
                project = hopsworks.login(api_key_value=api_key)
                mr = project.get_model_registry()
                model_file = mr.get_model("karachi_ensemble_aqi_final", version=1)
                
                save_folder = os.path.join(os.getcwd(), "model_cache")
                model_path = os.path.join(save_folder, "karachi_ensemble_aqi_final.pkl")
                
                # THE SMART LOAD: Only download if it doesn't exist locally
                if not os.path.exists(model_path):
                    os.makedirs(save_folder, exist_ok=True)
                    model_file.download(save_folder)
                
                ensemble_model = joblib.load(model_path)
                st.sidebar.success("✅ Model Loaded Successfully!")

                # 2. Fetch the next 72 hours of Weather & Pollutants from Open-Meteo
                st.info("📡 Fetching real-time satellite data for Karachi...")
                
                url = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=24.8607&longitude=67.0011&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,aerosol_optical_depth,dust,uv_index&timezone=Asia%2FKarachi&forecast_days=3"
                response = requests.get(url).json()
                
                # Parse the API response into a DataFrame
                future_data = pd.DataFrame(response['hourly'])
                future_data['time'] = pd.to_datetime(future_data['time'])
                
                # 3. Engineer the exact features the model expects
                future_data['hour'] = future_data['time'].dt.hour
                future_data['day'] = future_data['time'].dt.day
                future_data['month'] = future_data['time'].dt.month
                future_data['day_of_week'] = future_data['time'].dt.dayofweek
                
                # Calculate Momentum (_change) features (Simple diff for inference)
                pollutants = ['pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'aerosol_optical_depth', 'dust', 'uv_index']
                for p in pollutants:
                    future_data[f'{p}_change'] = future_data[p].diff().fillna(0)
                    
                # Ensure columns match training EXACTLY (Order matters in ML!)
                # THE FIX: Dynamically ask the model for its memorized column order
                expected_columns = list(ensemble_model.feature_names_in_)
                
                X_inference = future_data[expected_columns]

                # 4. Generate the 72-Hour Predictions!
                future_data['Predicted_PM2_5'] = ensemble_model.predict(X_inference)
                
                # 5. Dashboard UI: Metrics & Alerts
                st.subheader("🚨 Live Alert System")
                max_pm25 = future_data['Predicted_PM2_5'].max()
                
                # The "Hazardous" threshold logic
                if max_pm25 > 150:
                    st.error(f"⚠️ HAZARDOUS AQI DETECTED: Peak PM2.5 expected to hit {max_pm25:.2f} µg/m³. Advise wearing N95 masks.")
                elif max_pm25 > 50:
                    st.warning(f"😷 MODERATE/POOR AQI: Peak PM2.5 expected to hit {max_pm25:.2f} µg/m³. Sensitive groups should limit outdoor activity.")
                else:
                    st.success(f"🍃 GOOD AQI: Peak PM2.5 will remain safe at {max_pm25:.2f} µg/m³.")

                # 6. Dashboard UI: The 3-Day Trend Graph
                st.subheader("📈 72-Hour PM2.5 Forecasting Trend")
                st.line_chart(data=future_data, x='time', y='Predicted_PM2_5', color="#FF4B4B")

                # Show the raw data table for the evaluator
                with st.expander("View Raw Inference Data"):
                    st.dataframe(future_data[['time', 'Predicted_PM2_5'] + expected_columns])

            except Exception as e:
                st.error(f"Pipeline Error: {e}")