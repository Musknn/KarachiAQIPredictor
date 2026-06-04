import streamlit as st
import hopsworks
import pandas as pd
import requests
import joblib
from datetime import datetime, timedelta
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")

st.set_page_config(page_title="Karachi AQI Forecaster", page_icon="🌫️", layout="wide")

st.title("🌫️ Karachi 72-Hour Air Quality Forecast")
st.markdown("Powered by a Stacking Ensemble (LightGBM + Random Forest + Ridge)")

st.sidebar.header("⚙️ System Configuration")
st.sidebar.info("Secure connection to Cloud Registry enabled.")

def calculate_epa_aqi(pm25_concentration):
    """Converts raw PM2.5 concentration (µg/m³) to standard US EPA AQI"""
    c = float(pm25_concentration)
    if c < 0:
        c = 0
    
    # EPA Breakpoints (C_low, C_high, I_low, I_high)
    if c <= 12.0:
        return round(((50 - 0) / (12.0 - 0.0)) * (c - 0.0) + 0)
    elif c <= 35.4:
        return round(((100 - 51) / (35.4 - 12.1)) * (c - 12.1) + 51)
    elif c <= 55.4:
        return round(((150 - 101) / (55.4 - 35.5)) * (c - 35.5) + 101)
    elif c <= 150.4:
        return round(((200 - 151) / (150.4 - 55.5)) * (c - 55.5) + 151)
    elif c <= 250.4:
        return round(((300 - 201) / (250.4 - 150.5)) * (c - 150.5) + 201)
    elif c <= 500.4:
        return round(((500 - 301) / (500.4 - 250.5)) * (c - 250.5) + 301)
    else:
        return 500 # Hazardous max

def get_aqi_status(aqi_val):
    """Returns the health text category and color based on AQI value"""
    if aqi_val <= 50:
        return "Good", "🍃"
    elif aqi_val <= 100:
        return "Moderate", "🟡"
    elif aqi_val <= 150:
        return "Unhealthy for Sensitive Groups", "🟠"
    elif aqi_val <= 200:
        return "Unhealthy", "🔴"
    elif aqi_val <= 300:
        return "Very Unhealthy", "🟣"
    else:
        return "Hazardous", "🟤"

# Run the pipeline
if st.sidebar.button("Run 3-Day Forecast"):
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
                
                if not os.path.exists(model_path):
                    os.makedirs(save_folder, exist_ok=True)
                    model_file.download(save_folder)
                
                ensemble_model = joblib.load(model_path)
                st.sidebar.success("✅ Model Loaded Successfully!")

                # 2. Fetch the next 72 hours of Weather & Pollutants from Open-Meteo
                st.info("📡 Fetching real-time satellite data for Karachi...")
                
                url = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=24.8607&longitude=67.0011&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,aerosol_optical_depth,dust,uv_index&timezone=Asia%2FKarachi&forecast_days=3"
                response = requests.get(url).json()
                
                future_data = pd.DataFrame(response['hourly'])
                future_data['time'] = pd.to_datetime(future_data['time'])
                
                # 3. Engineer the exact features the model expects
                future_data['hour'] = future_data['time'].dt.hour
                future_data['day'] = future_data['time'].dt.day
                future_data['month'] = future_data['time'].dt.month
                future_data['day_of_week'] = future_data['time'].dt.dayofweek
                
                pollutants = ['pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'aerosol_optical_depth', 'dust', 'uv_index']
                for p in pollutants:
                    future_data[f'{p}_change'] = future_data[p].diff().fillna(0)
                    
                expected_columns = list(ensemble_model.feature_names_in_)
                X_inference = future_data[expected_columns]

                # 4. Generate the 72-Hour Predictions & Translate to EPA AQI
                future_data['Predicted_PM2_5'] = ensemble_model.predict(X_inference)
                future_data['Predicted_AQI'] = future_data['Predicted_PM2_5'].apply(calculate_epa_aqi)
                
                # 5. Summary Grid Layout
                st.write("---")
                col_now, col_forecast = st.columns([1, 2])
                
                with col_now:
                    st.subheader("📍 Right Now")
                    current_row = future_data.iloc[0]
                    curr_aqi = int(current_row['Predicted_AQI'])
                    curr_status, emoji = get_aqi_status(curr_aqi)
                    
                    st.metric(label=f"Air Quality Index (AQI)", value=f"{curr_aqi}")
                    st.markdown(f"### {emoji} {curr_status}")
                    st.caption(f"Raw Prediction: {current_row['Predicted_PM2_5']:.1f} µg/m³ PM2.5")
                
                with col_forecast:
                    st.subheader("🗓️ What the Next 3 Days Look Like")
                    # Group by date to extract maximum expected pollution daily
                    future_data['date_label'] = future_data['time'].dt.strftime('%A (%d %b)')
                    daily_summary = future_data.groupby('date_label').agg({
                        'Predicted_AQI': 'max',
                        'Predicted_PM2_5': 'max'
                    }).reindex(future_data['date_label'].unique())
                    
                    card_cols = st.columns(len(daily_summary))
                    for idx, (date_str, row) in enumerate(daily_summary.iterrows()):
                        with card_cols[idx]:
                            day_aqi = int(row['Predicted_AQI'])
                            day_status, day_emoji = get_aqi_status(day_aqi)
                            st.metric(label=date_str, value=f"{day_aqi}", delta=day_status, delta_color="inverse")
                            st.caption(f"Peak: {row['Predicted_PM2_5']:.1f} µg/m³")

                # 6. Dashboard UI: Live Alert System
                st.write("---")
                st.subheader("🚨 Live Alert System")
                max_aqi = future_data['Predicted_AQI'].max()
                max_pm25 = future_data['Predicted_PM2_5'].max()
                
                if max_aqi > 150:
                    st.error(f"⚠️ HAZARDOUS REGIONAL AIR QUALITY EXPECTED: Peak AQI will cross into unhealthy thresholds at {max_aqi} (Raw PM2.5: {max_pm25:.1f} µg/m³). Advise wearing protective masks outdoors.")
                elif max_aqi > 50:
                    st.warning(f"😷 MODERATE METROPOLITAN BACKGROUND: Peak AQI is expected to settle around {max_aqi} (Raw PM2.5: {max_pm25:.1f} µg/m³). Sensitive populations should monitor physical activity levels.")
                else:
                    st.success(f"🍃 CLEAN ATMOSPHERIC CONDITIONS: Max baseline AQI will remain stable at {max_aqi}. Safe ambient levels across coordinates.")

                # 7. Dashboard UI: The 3-Day Trend Graph
                st.subheader("📈 72-Hour AQI Forecasting Trend")
                # Creating a graph cleanly tracking the standard 0-500 scale
                chart_df = future_data[['time', 'Predicted_AQI']].copy()
                chart_df = chart_df.rename(columns={'Predicted_AQI': 'Air Quality Index (AQI)'})
                st.line_chart(data=chart_df, x='time', y='Air Quality Index (AQI)', color="#FF4B4B")

                # Show the raw data table for the evaluator
                with st.expander("View Raw Inference Data"):
                    st.dataframe(future_data[['time', 'Predicted_AQI', 'Predicted_PM2_5'] + expected_columns])

            except Exception as e:
                st.error(f"Pipeline Error: {e}")