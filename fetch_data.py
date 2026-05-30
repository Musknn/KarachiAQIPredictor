import requests
import pandas as pd

def fetch_expanded_aqi(lat=24.8607, lon=67.0011): 
    # The new, massive URL with ammonia, methane, sulphur dioxide, and weather variables!
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&start_date=2021-01-01&end_date=2026-05-27&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,aerosol_optical_depth,dust,uv_index,uv_index_clear_sky,ammonia,methane,alder_pollen,birch_pollen,grass_pollen,mugwort_pollen,olive_pollen,ragweed_pollen"    
    print("Fetching expanded data from Open-Meteo...")
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        return df
    else:
        print(f"Failed to fetch data. Status Code: {response.status_code}")
        return None

# Execute the function
aqi_df = fetch_expanded_aqi()

if aqi_df is not None:
    # Save the expanded raw data to the same CSV file name
    aqi_df.to_csv("karachi_raw_aqi.csv", index=False)
    print(f"Success! Data saved with {len(aqi_df)} rows and {len(aqi_df.columns)} columns.")
    print(aqi_df.head())