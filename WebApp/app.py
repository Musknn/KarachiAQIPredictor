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

# ==========================================
# 🎨 CUSTOM CSS FOR PROFESSIONAL UI
# ==========================================
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.8rem !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌫️ Karachi 72-Hour Air Quality Forecast")

st.sidebar.header("⚙️ System Configuration")
st.sidebar.info("Secure connection to Cloud Registry enabled.")

def calculate_epa_aqi(pm25_concentration):
    """Converts raw PM2.5 concentration (µg/m³) to standard US EPA AQI"""
    c = float(pm25_concentration)
    if c < 0:
        c = 0
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
        return 500

def get_aqi_status(aqi_val):
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

def load_latest_model(mr):
    """
    Always fetches the latest version of karachi_aqi_production from the
    Model Registry. Caches it locally with today's date in the filename so
    yesterday's stale .pkl is never reused after the daily training run.
    """
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    save_folder = os.path.join(os.getcwd(), "model_cache")
    os.makedirs(save_folder, exist_ok=True)
    cached_path = os.path.join(save_folder, f"karachi_aqi_production_{today_str}.pkl")

    if os.path.exists(cached_path):
        # Validate the cache is not a numpy-array-trained model with Column_0 names
        _m = joblib.load(cached_path)
        _bad = False
        def _check_names(m):
            if hasattr(m, 'feature_names_in_'):
                names = list(m.feature_names_in_)
                if names and str(names[0]).startswith('Column_'):
                    return True
            if hasattr(m, 'estimators_'):
                for e in m.estimators_:
                    if _check_names(e): return True
            return False
        _bad = _check_names(_m)
        if not _bad:
            st.sidebar.caption("📂 Using today's cached model.")
            return _m
        # Cache is corrupt — delete it and re-download
        os.remove(cached_path)
        st.sidebar.caption("🔄 Re-downloading model (cached version was incompatible)...")

    # No cache for today → fetch latest version from registry.
    # get_best_model picks the version with the lowest MSE (matches what
    # training_pipeline.py logs). Falls back to highest version number if
    # the Hopsworks tier does not support get_best_model.
    try:
        model_meta = mr.get_best_model("karachi_aqi_production", metric="mse", direction="min")
    except Exception:
        all_versions = mr.get_models("karachi_aqi_production")
        model_meta = max(all_versions, key=lambda m: m.version)

    st.sidebar.caption(f"☁️ Downloading model v{model_meta.version} from registry...")
    model_meta.download(save_folder)

    # Rename to date-stamped path so tomorrow's run re-downloads fresh
    raw_path = os.path.join(save_folder, "karachi_aqi_production.pkl")
    if os.path.exists(raw_path):
        os.rename(raw_path, cached_path)

    return joblib.load(cached_path)

# ==========================================
# RUN THE PIPELINE
# ==========================================
if st.sidebar.button("Run 3-Day Forecast", type="primary"):
    if not api_key:
        st.error("🚨 Configuration Error: API key not found. Please check your .env file or Streamlit secrets.")
    else:
        with st.spinner("Connecting to Cloud Model Registry..."):
            try:
                # 1. Connect to Hopsworks and load today's best model
                project = hopsworks.login(api_key_value=api_key)
                mr = project.get_model_registry()
                ensemble_model = load_latest_model(mr)
                st.sidebar.success("✅ Model Loaded Successfully!")

                # 2. Fetch 72-hour forecast data from Open-Meteo.
                # NOTE: sulphur_dioxide is excluded — it returned all-NaN for
                # Karachi and was never stored in the Feature Store, so the
                # trained model does not expect it as a feature.
                # 2. Fetch 72-hour forecast data PLUS 24-hours of past data
                # We need the past data so the .diff() function has a baseline
                # for the very first hour of the forecast.
                url = (
                    "https://air-quality-api.open-meteo.com/v1/air-quality"
                    "?latitude=24.8607&longitude=67.0011"
                    "&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,"
                    "ozone,aerosol_optical_depth,dust,uv_index"
                    "&timezone=Asia%2FKarachi&past_days=1&forecast_days=3"
                )
                response = requests.get(url).json()
                future_data = pd.DataFrame(response['hourly'])
                future_data['time'] = pd.to_datetime(future_data['time'])

                # 3. Engineer features — must exactly mirror training_pipeline.py.
                # Temporal features
                future_data['hour']        = future_data['time'].dt.hour
                future_data['day']         = future_data['time'].dt.day
                future_data['month']       = future_data['time'].dt.month
                future_data['day_of_week'] = future_data['time'].dt.dayofweek

                # Momentum delta features (8 pollutants, no sulphur_dioxide)
                pollutants = [
                    'pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide',
                    'ozone', 'aerosol_optical_depth', 'dust', 'uv_index',
                ]
                for p in pollutants:
                    future_data[f'{p}_change'] = future_data[p].diff().fillna(0)

            
                future_data = future_data.iloc[-72:].reset_index(drop=True)

    
                _sensor   = ['pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide',
                             'ozone', 'aerosol_optical_depth', 'dust', 'uv_index']
                _temporal = ['hour', 'day', 'month', 'day_of_week']
                _change   = sorted([f'{c}_change' for c in _sensor])
                expected_columns = _sensor + _temporal + _change

                X_inference = future_data[expected_columns]

                # 4. Predict
                future_data['Predicted_PM2_5'] = ensemble_model.predict(X_inference)
                future_data['Predicted_AQI']   = future_data['Predicted_PM2_5'].apply(calculate_epa_aqi)

                # ==========================================
                # 📊 DASHBOARD
                # ==========================================
                st.write("")

                # Alert banner
                max_aqi  = future_data['Predicted_AQI'].max()
                max_pm25 = future_data['Predicted_PM2_5'].max()

                if max_aqi > 150:
                    st.error(
                        f"**⚠️ HAZARDOUS REGIONAL AIR QUALITY EXPECTED:** "
                        f"Peak AQI will cross into unhealthy thresholds at **{max_aqi}** "
                        f"(Raw PM2.5: {max_pm25:.1f} µg/m³). Advise wearing protective masks outdoors."
                    )
                elif max_aqi > 50:
                    st.warning(
                        f"**😷 MODERATE METROPOLITAN BACKGROUND:** "
                        f"Peak AQI is expected to settle around **{max_aqi}** "
                        f"(Raw PM2.5: {max_pm25:.1f} µg/m³). Sensitive populations should monitor activity levels."
                    )
                else:
                    st.success(
                        f"**🍃 CLEAN ATMOSPHERIC CONDITIONS:** "
                        f"Max baseline AQI will remain stable at **{max_aqi}**. "
                        f"Safe ambient levels across coordinates."
                    )

                st.write("")

                # Right Now + 72-Hour Outlook
                col_now, col_forecast = st.columns([1.2, 3])

                with col_now:
                    with st.container(border=True):
                        st.subheader("📍 Right Now")
                        current_row = future_data.iloc[0]
                        curr_aqi = int(current_row['Predicted_AQI'])
                        curr_status, emoji = get_aqi_status(curr_aqi)
                        st.metric(label="Current Air Quality Index", value=f"{curr_aqi}")
                        st.markdown(f"#### {emoji} {curr_status}")
                        st.divider()
                        st.caption(f"**Raw PM2.5 Concentration:**\n{current_row['Predicted_PM2_5']:.2f} µg/m³")

                with col_forecast:
                    with st.container(border=True):
                        st.subheader("🗓️ 72-Hour Outlook")
                        future_data['date_label'] = future_data['time'].dt.strftime('%A (%d %b)')

                        daily_summary = future_data.groupby('date_label', sort=False).agg(
                            Peak_AQI  = ('Predicted_AQI',   'max'),
                            Avg_AQI   = ('Predicted_AQI',   'mean'),
                            Peak_PM25 = ('Predicted_PM2_5', 'max'),
                            Avg_PM25  = ('Predicted_PM2_5', 'mean'),
                        ).reindex(future_data['date_label'].unique())

                        card_cols = st.columns(len(daily_summary))
                        for idx, (date_str, row) in enumerate(daily_summary.iterrows()):
                            with card_cols[idx]:
                                with st.container(border=True):
                                    peak_aqi   = int(row['Peak_AQI'])
                                    avg_aqi    = int(row['Avg_AQI'])
                                    day_status, day_emoji = get_aqi_status(peak_aqi)
                                    st.markdown(f"**{date_str}**")
                                    st.metric(
                                        label="Peak AQI",
                                        value=f"{peak_aqi}",
                                        delta=day_status,
                                        delta_color="inverse",
                                    )
                                    st.markdown(f"*Avg AQI: {avg_aqi}*")
                                    st.caption(f"Raw Peak: {row['Peak_PM25']:.1f} µg/m³")

                # Trend chart
                with st.container(border=True):
                    st.subheader("📈 Environmental AQI Forecasting Trend")
                    chart_df = (
                        future_data[['time', 'Predicted_AQI']]
                        .copy()
                        .rename(columns={'time': 'Time', 'Predicted_AQI': 'Air Quality Index (AQI)'})
                        .set_index('Time')
                    )
                    st.line_chart(data=chart_df, color="#FF4B4B", height=300)

                # Raw data expander
                with st.expander("🔍 System Logs & Raw Inference Data"):
                    st.dataframe(
                        future_data[['time', 'Predicted_AQI', 'Predicted_PM2_5'] + expected_columns],
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Pipeline Error: {e}")
                st.exception(e)