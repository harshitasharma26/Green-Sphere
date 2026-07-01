import streamlit as st
import pandas as pd
import requests
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

AQI_API_URL = "http://localhost:8000"


# --- Live air quality data, fetched directly from the internet (Open-Meteo). ---
# No separate server needed for this part -- it runs right here in Streamlit.
def get_coordinates(city_name):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city_name, "count": 1}
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    if "results" not in data:
        raise ValueError(f"Could not find a location named '{city_name}'")
    result = data["results"][0]
    return result["latitude"], result["longitude"], result["name"], result.get("country")


def get_live_air_quality(lat, lon):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
    }
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    return data["current"]


def safe_open_image(path, caption=""):
    """Load an image, showing a friendly warning instead of crashing if the file is missing."""
    try:
        return Image.open(path)
    except FileNotFoundError:
        st.warning(f"⚠️ Image file not found: `{path}` — make sure it's in the same folder as this script.")
        return None


def safe_read_csv(path):
    """Load a CSV, stopping with a clear message instead of a raw traceback if the file is missing."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"⚠️ Data file not found: `{path}` — make sure it's in the same folder as this script.")
        st.stop()
st.set_page_config(
    page_title="GreenSphere - Air Quality, Water Quality & Soil Analysis Tool",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    
    .stApp {
        background-color: #f5f7f6;
    }

    .stApp p, .stApp li, .stApp label {
        color: #1a1a1a !important;
    }

    h1, h2, h3 {
        color: #1b5e20 !important;
    }

   
    section[data-testid="stSidebar"] {
        background-color: #2e7d32 !important;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    [data-testid="stSelectbox"] > div > div {
        background-color: white !important;
        color: #1a1a1a !important;
    }

    [data-testid="stNumberInput"] input {
        background-color: white !important;
        color: #1a1a1a !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {
        color: #1a1a1a !important;
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🌿 GreenSphere")
    st.markdown("---")
    menu = st.radio("Go to", ["🏠 Introduction", "🏭 Air Quality", "💧 Water Quality", "🌱 Soil & Crop"])

st.title("🌍 GreenSphere")
st.header("ML-Powered Environmental Monitoring — Air, Water & Soil Analysis")
st.markdown(
    "<p style='color:#1a1a1a; font-size:16px;'>"
    "GreenSphere uses machine learning to predict Air Quality Index (AQI), "
    "analyze water safety using pH, Dissolved Oxygen and BOD levels, and recommend "
    "the best crop based on soil nutrients like Nitrogen, Phosphorous and Potassium. "
    "Free environmental monitoring and crop recommendation tool for India."
    "</p>",
    unsafe_allow_html=True
)

if menu == "🏠 Introduction":
    st.markdown("<h6 style='font-size: 25px;'>Analyzing Air, Water and Soil Conditions for a Sustainable Environment</h6>", unsafe_allow_html=True)
    intro_image = safe_open_image("environmental-protection-326923_1280.jpg")
    if intro_image:
        st.image(intro_image, width=800, caption="Protecting our environment with ML")
    st.subheader("About GreenSphere")
    st.write( """
    This system monitors environmental conditions using machine learning to support better decision-making.
             
    It analyzes:
             
    • 🌫 Air Quality  
    • 💧 Water Safety  
    • 🌱 Soil Fertility & Crop Recommendation  
    """
    )
    st.info("Use the sidebar to explore each module.")
elif menu == "🏭 Air Quality":
    st.markdown("<h2 style='font-size: 25px;'>🏭 Air Quality Monitoring</h2>",unsafe_allow_html=True)
    air_image = safe_open_image("aqiimage.jpeg")
    if air_image:
        st.image(air_image, caption="Air Quality Monitoring", width=800)
    st.markdown("""
     Air quality tells us how clean or polluted the air is.
     We mainly look at pollutants like PM2.5, PM10, CO, NO₂, SO₂ and O₃.
     Here, we use past data and machine learning to predict AQI.
    """)
    st.divider()

    # --- City list loaded directly from the CSV (no API needed for this part) ---
    aqi_data_local = safe_read_csv("cleaned_aqi_data.csv")
    cities = sorted(aqi_data_local["City"].unique().tolist())

    city = st.selectbox("Select City", cities)
    year = st.number_input(
        "Enter Year",
        min_value=2015,
        max_value=2100,
        step=1
    )

    prediction = None
    try:
        predict_resp = requests.post(
            f"{AQI_API_URL}/predict",
            json={"city": city, "year": int(year)},
            timeout=60,
        )
        predict_resp.raise_for_status()
        result = predict_resp.json()
        prediction = result["predicted_aqi"]
        rmse = result["rmse"]
        category = result["category"]
    except requests.exceptions.ConnectionError:
        st.info(
            "ℹ️ Future-year prediction needs the API running "
            "(`uvicorn api:app --reload --port 8000`). "
            "Skipping prediction — Current Conditions below still works without it."
        )
    except requests.exceptions.Timeout:
        st.warning(
            "⏳ The prediction is taking longer than expected (first request per "
            "city trains the model and fetches live data). Try again in a moment."
        )
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Prediction request failed: {e}")

    if prediction is not None:
        st.caption(
            "📊 This prediction uses historical average pollutant levels for "
            "this city, so Year genuinely affects the result. For today's real "
            "pollutant readings, see 'Current Conditions (Live)' below."
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Predicted AQI")
            st.write(f"{prediction:.0f}")

        with col2:
            st.subheader("Model RMSE")
            st.write(f"{rmse:.2f}")

        st.caption(
            f"Model used: **{result.get('model_name', 'RandomForest')}** "
            f"(chosen automatically for lowest error) · "
            f"Cross-validated RMSE: {result.get('cv_rmse', rmse):.2f}"
        )

        category_display = {
            "Good": ("success", "Good — Air quality is satisfactory"),
            "Moderate": ("info", "Moderate — Acceptable quality"),
            "Unhealthy for Sensitive Groups": ("warning", "Unhealthy for Sensitive Groups"),
            "Unhealthy": ("warning", "Unhealthy — Everyone may be affected"),
            "Very Unhealthy": ("error", "Very Unhealthy — Health alert"),
            "Hazardous": ("error", "Hazardous — Emergency conditions"),
        }
        level, message = category_display.get(category, ("info", category))
        getattr(st, level)(message)

       
        try:
            trend_resp = requests.get(f"{AQI_API_URL}/trend/{city}", timeout=10)
            trend_resp.raise_for_status()
            trend_data = pd.DataFrame(trend_resp.json()["trend"]).sort_values("Year")

            last_historical_year = int(trend_data["Year"].max())
            forecast_start = last_historical_year
            forecast_end = max(int(year), last_historical_year + 5)

            forecast_resp = requests.get(
                f"{AQI_API_URL}/forecast/{city}",
                params={"start_year": forecast_start, "end_year": forecast_end},
                timeout=60,
            )
            forecast_resp.raise_for_status()
            forecast_data = pd.DataFrame(forecast_resp.json()["forecast"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_data["Year"], y=trend_data["AQI"],
                mode="lines+markers", name="Historical AQI",
                line=dict(color="#1b5e20")
            ))
            fig.add_trace(go.Scatter(
                x=forecast_data["Year"], y=forecast_data["AQI"],
                mode="lines+markers", name="Forecast (model)",
                line=dict(color="#e65100", dash="dash")
            ))
            fig.add_trace(go.Scatter(
                x=[year], y=[prediction],
                mode="markers", name="Your Selected Year",
                marker=dict(size=14, symbol="star", color="#c62828")
            ))
            fig.update_layout(
                title=f"AQI: Historical Trend + Forecast for {city}",
                xaxis_title="Year", yaxis_title="AQI",
            )
            st.plotly_chart(fig)
            st.caption(
                "Solid line = actual historical AQI. Dashed line = model forecast "
                "using this city's historical average pollutant levels."
            )
        except requests.exceptions.RequestException as e:
            st.warning(f"Could not load trend/forecast chart: {e}")

  
    st.divider()
    st.markdown("## 🌐 Current Conditions (Live)")

    try:
        lat, lon, matched_name, country = get_coordinates(city)
        current = get_live_air_quality(lat, lon)

        c1, c2, c3 = st.columns(3)
        c1.metric("PM2.5", f"{current['pm2_5']} µg/m³")
        c2.metric("PM10", f"{current['pm10']} µg/m³")
        c3.metric("O3", f"{current['ozone']} µg/m³")

        c4, c5, c6 = st.columns(3)
        c4.metric("NO2", f"{current['nitrogen_dioxide']} µg/m³")
        c5.metric("CO", f"{current['carbon_monoxide']} µg/m³")
        c6.metric("SO2", f"{current['sulphur_dioxide']} µg/m³")
    except (requests.exceptions.RequestException, ValueError, KeyError) as e:
        st.warning(f"Could not load live conditions: {e}")

elif menu == "💧 Water Quality":
 st.markdown("<h2 style='font-size: 25px;'>💧 Water Quality Analysis</h2>",unsafe_allow_html=True)
 water_image = safe_open_image("water_image.jpeg")
 if water_image:
  st.image(water_image, caption="Water Quality Monitoring", width=800)
 st.markdown("""
 Water quality depends on a few important factors like pH, dissolved oxygen and BOD.
 - pH shows if water is acidic or basic  
 - DO tells how much oxygen is available  
 - BOD indicates pollution level  
 Based on these values, we classify water as good, moderate or poor.
 """)
 st.divider()
 water_data = safe_read_csv("cleaned_water_data.csv")
 num_cols = ["Min pH", "Max pH", "Min Dissolved Oxygen", "Max Dissolved Oxygen", "Min BOD", "Max BOD"]
 for col in num_cols:
    water_data[col] = pd.to_numeric(water_data[col], errors="coerce")
 water_data["pH_avg"] = (water_data["Min pH"] + water_data["Max pH"]) / 2
 water_data["DO_avg"] = (water_data["Min Dissolved Oxygen"] + water_data["Max Dissolved Oxygen"]) / 2
 water_data["BOD_avg"] = (water_data["Min BOD"] + water_data["Max BOD"]) / 2

 states = water_data["State Name"].unique()
 selected_state = st.selectbox("Select State", states)
 state_data = water_data[water_data["State Name"] == selected_state]
 locations = state_data["Name of Monitoring Location"].unique()
 selected_location = st.selectbox("Select Monitoring Location", locations)
 loc_row = state_data[state_data["Name of Monitoring Location"] == selected_location].iloc[0]

 pH_avg  = loc_row["pH_avg"]
 DO_avg  =loc_row["DO_avg"]
 BOD_avg = loc_row["BOD_avg"]

 if pd.isna(DO_avg) or pd.isna(BOD_avg):
        quality = "No data"
 elif DO_avg >= 6 and BOD_avg <= 3:
        quality = "Good"
 elif DO_avg >= 5 and BOD_avg <= 3:
        quality = "Moderate"
 else:
        quality = "Poor"
 st.divider()
 st.markdown("### 💧 Results")
 st.write("State:", loc_row["State Name"])
 st.write("Average pH:", "N/A" if pd.isna(pH_avg) else round(pH_avg, 2))
 st.write("Average Dissolved Oxygen:", "N/A" if pd.isna(DO_avg) else round(DO_avg, 2))
 st.write("Average BOD:", "N/A" if pd.isna(BOD_avg) else round(BOD_avg, 2))
    
 if quality == "Good":
        st.success("GOOD Water Quality ")
 elif quality == "Moderate":
        st.warning("MODERATE Water Quality ")
 elif quality == "No data":
        st.info("No data available for this location to determine water quality.")
 else:
        st.error("POOR Water Quality ")

elif menu == "🌱 Soil & Crop":
 st.markdown("<h2 style='font-size: 25px;'>🌱 Soil Fertility & Crop Recommendation</h2>", unsafe_allow_html=True)
 soil_image = safe_open_image("soil_image.jpeg")
 if soil_image:
  st.image(soil_image, caption="Healthy soil, healthy crops", width=800)
 st.markdown("""
 Soil fertility depends on nutrients like nitrogen, phosphorous and potassium.
 Other factors like pH, temperature, humidity and rainfall also matter.
 Using these values, the system suggests the best crop using machine learning.
 """)
 st.divider()
 soil_data = safe_read_csv("crop_recommendation_dataset.csv")
 X = soil_data[["Nitrogen", "Phosphorous", "Potassium", "PH","Temperature", "Humidity", "Rainfall"]]   
 y = soil_data["Crop"]  
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 model = RandomForestClassifier(n_estimators=300, random_state=42)
 model.fit(X_train, y_train)
 prediction = model.predict(X_test)
 accuracy = accuracy_score(y_test,prediction)
 st.markdown("##  Enter Soil Conditions")
 col1, col2 = st.columns(2)
 with col1:
      Nitrogen = st.slider("Nitrogen (N)", 0, 150, 50)
      Phosphorous = st.slider("Phosphorous (P)", 0, 150, 50)
      Temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
 with col2:
      Potassium = st.slider("Potassium (K)", 0, 150, 50)
      PH = st.slider("Soil PH", 0.0, 14.0, 7.0)
      Humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)
 Rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)
 user_input = [[Nitrogen, Phosphorous, Potassium, PH, Temperature, Humidity, Rainfall]]
 prediction = model.predict(user_input)
 st.markdown("## 🌾 Recommended Crop")
 st.success(f"Best crop for this soil:  {prediction[0].upper()}")
 st.metric(label="Soil Model Accuracy", value=f"{accuracy * 100:.1f}%")
 st.markdown("## 🌱 Soil Fertility")
 if Nitrogen > 80 and Phosphorous > 80 and Potassium > 80:
     st.success("Soil fertility is HIGH ")
 elif Nitrogen > 40 and Phosphorous > 40 and Potassium > 40:
     st.warning("Soil fertility is MODERATE ")
 else:
    st.error("Soil fertility is LOW ")
 st.markdown("## 📊 Your Soil vs. Recommended Crop's Ideal Profile")
 radar_features = ["Nitrogen", "Phosphorous", "Potassium", "PH", "Temperature", "Humidity", "Rainfall"]
 user_values = [Nitrogen, Phosphorous, Potassium, PH, Temperature, Humidity, Rainfall]

 
 crop_rows = soil_data[soil_data["Crop"] == prediction[0]]
 crop_avg = crop_rows[radar_features].mean()

 
 feature_min = soil_data[radar_features].min()
 feature_max = soil_data[radar_features].max()

 def normalize(values):
     return [
         100 * (v - feature_min[f]) / (feature_max[f] - feature_min[f])
         for f, v in zip(radar_features, values)
     ]

 user_normalized = normalize(user_values)
 crop_normalized = normalize(crop_avg.tolist())

 
 categories_closed = radar_features + [radar_features[0]]
 user_closed = user_normalized + [user_normalized[0]]
 crop_closed = crop_normalized + [crop_normalized[0]]

 fig = go.Figure()
 fig.add_trace(go.Scatterpolar(
     r=crop_closed,
     theta=categories_closed,
     fill='toself',
     name=f"Ideal for {prediction[0].title()}",
     opacity=0.5
 ))
 fig.add_trace(go.Scatterpolar(
     r=user_closed,
     theta=categories_closed,
     fill='toself',
     name="Your Soil",
     opacity=0.5
 ))
 fig.update_layout(
     polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
     title=f"Your Soil Profile vs. {prediction[0].title()}'s Ideal Range",
     showlegend=True
 )
 st.plotly_chart(fig)
 st.caption(
     "Each axis is scaled 0-100 based on the dataset's min/max for that feature. "
     "The closer the two shapes overlap, the better your soil matches the recommended crop."
 )