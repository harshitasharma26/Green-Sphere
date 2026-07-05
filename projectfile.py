import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score

AQI_FEATURES = ["Year", "PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]


def safe_open_image(path, caption=""):
    try:
        return Image.open(path)
    except FileNotFoundError:
        st.warning(f"⚠️ Image file not found: `{path}` — make sure it's in the same folder as this script.")
        return None


def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"⚠️ Data file not found: `{path}` — make sure it's in the same folder as this script.")
        st.stop()


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


def classify_water_quality(do_avg, bod_avg):
    if pd.isna(do_avg) or pd.isna(bod_avg):
        return "No data"
    elif do_avg >= 6 and bod_avg <= 3:
        return "Good"
    elif do_avg >= 5 and bod_avg <= 3:
        return "Moderate"
    else:
        return "Poor"


def main_water_concern(pH_avg, DO_avg, BOD_avg):
    if pd.isna(pH_avg) or pd.isna(DO_avg) or pd.isna(BOD_avg):
        return "Insufficient data"

    pH_deviation = 0 if 6.5 <= pH_avg <= 8.5 else min(abs(pH_avg - 6.5), abs(pH_avg - 8.5)) / 2
    do_deviation = max(0, (6 - DO_avg) / 6)
    bod_deviation = max(0, (BOD_avg - 3) / 3)

    deviations = {"pH": pH_deviation, "Dissolved Oxygen": do_deviation, "BOD": bod_deviation}
    top_param = max(deviations, key=deviations.get)

    if deviations[top_param] <= 0:
        return "None — all parameters within healthy range"
    return top_param


WATER_USAGE_GUIDANCE = {
    "Good": "Safe for drinking after standard treatment; suitable for aquatic life and irrigation.",
    "Moderate": "Suitable for irrigation and industrial use; not recommended for direct drinking without treatment.",
    "Poor": "Not recommended for drinking, bathing, or irrigation without significant treatment.",
    "No data": "Not enough data available to determine safe usage.",
}


def build_water_summary(location, quality, pH_avg, DO_avg, BOD_avg, concern):
    if quality == "No data":
        return f"Not enough data is available for {location} to determine water quality."

    summary = (
        f"Water quality at {location} is rated {quality}"
    )
    if concern not in ("None — all parameters within healthy range", "Insufficient data"):
        summary += f", primarily due to {concern} levels"
    summary += ". "
    summary += WATER_USAGE_GUIDANCE.get(quality, "")
    return summary


def get_pollutant_forecast(lat, lon, days=5):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "forecast_days": days,
    }
    response = requests.get(url, params=params, timeout=15)
    data = response.json()
    hourly = data["hourly"]

    df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "pm2_5": hourly["pm2_5"],
        "pm10": hourly["pm10"],
        "nitrogen_dioxide": hourly["nitrogen_dioxide"],
        "carbon_monoxide": hourly["carbon_monoxide"],
        "sulphur_dioxide": hourly["sulphur_dioxide"],
        "ozone": hourly["ozone"],
    })
    df["date"] = df["time"].dt.date
    daily = df.groupby("date").mean(numeric_only=True).reset_index()
    return daily


@st.cache_resource
def get_or_train_aqi_model(city: str, _aqi_data: pd.DataFrame):
    city_data = _aqi_data[_aqi_data["City"] == city]
    required_cols = AQI_FEATURES + ["AQI"]
    city_data = city_data.dropna(subset=required_cols)

    if len(city_data) < 10:
        return None

    X = city_data[AQI_FEATURES]
    y = city_data["AQI"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_rows = len(city_data)
    cv_folds = min(5, max(2, n_rows // 10)) if n_rows >= 10 else None

    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }

    best_name, best_model, best_cv_rmse = None, None, float("inf")
    for name, candidate in candidates.items():
        if cv_folds:
            scores = cross_val_score(candidate, X, y, cv=cv_folds, scoring="neg_mean_squared_error")
            cv_rmse = float(np.sqrt(-scores.mean()))
        else:
            candidate.fit(X_train, y_train)
            cv_rmse = float(np.sqrt(mean_squared_error(y_test, candidate.predict(X_test))))

        if cv_rmse < best_cv_rmse:
            best_name, best_model, best_cv_rmse = name, candidate, cv_rmse

    if best_model is None:
        return None

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    return {
        "model": best_model,
        "model_name": best_name,
        "rmse": rmse,
        "cv_rmse": round(best_cv_rmse, 2),
        "last_year": int(city_data["Year"].max()),
        "trend": city_data[["Year", "AQI"]].to_dict(orient="records"),
    }


def categorize_aqi(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


HEALTH_TIPS = {
    "Good": "Great day for outdoor activities.",
    "Moderate": "Acceptable air quality; sensitive individuals should limit prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups": "Children, elderly, and people with asthma should limit outdoor activity.",
    "Unhealthy": "Everyone should reduce outdoor exertion; consider a mask outside.",
    "Very Unhealthy": "Avoid outdoor activity; keep windows closed; use an air purifier if available.",
    "Hazardous": "Stay indoors; avoid all outdoor exertion; keep windows closed.",
}

POLLUTANT_LIMITS = {
    "pm2_5": ("PM2.5", 15),
    "pm10": ("PM10", 45),
    "nitrogen_dioxide": ("NO2", 25),
    "carbon_monoxide": ("CO", 4000),
    "sulphur_dioxide": ("SO2", 40),
    "ozone": ("O3", 100),
}


def dominant_pollutant(row):
    ratios = {key: row[key] / limit for key, (_, limit) in POLLUTANT_LIMITS.items()}
    top_key = max(ratios, key=ratios.get)
    return POLLUTANT_LIMITS[top_key][0]


def predict_daily_aqi(entry, daily_row, year):
    model = entry["model"]
    user_input = [[
        year,
        daily_row["pm2_5"],
        daily_row["pm10"],
        daily_row["nitrogen_dioxide"],
        daily_row["carbon_monoxide"],
        daily_row["sulphur_dioxide"],
        daily_row["ozone"],
    ]]
    return float(model.predict(user_input)[0])


def build_forecast_summary(city, forecast_df):
    best_row = forecast_df.loc[forecast_df["AQI"].idxmin()]
    worst_row = forecast_df.loc[forecast_df["AQI"].idxmax()]
    overall_dominant = forecast_df["Dominant Pollutant"].mode().iloc[0]
    sensitive_days = forecast_df[forecast_df["AQI"] > 100]["Date"].tolist()

    summary = (
        f"Air quality in {city} is expected to range from {categorize_aqi(best_row['AQI'])} "
        f"to {categorize_aqi(worst_row['AQI'])} over the next {len(forecast_df)} days. "
        f"The best day looks to be {best_row['Date']} (AQI {best_row['AQI']:.0f}), while "
        f"{worst_row['Date']} looks worst (AQI {worst_row['AQI']:.0f}). "
        f"{overall_dominant} is the main pollutant of concern overall."
    )
    if sensitive_days:
        days_text = ", ".join(str(d) for d in sensitive_days)
        summary += f" Sensitive groups should limit outdoor activity on: {days_text}."
    return summary


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
    st.write("""
    This system monitors environmental conditions using machine learning to support better decision-making.

    It analyzes:

    • 🌫 Air Quality  
    • 💧 Water Safety  
    • 🌱 Soil Fertility & Crop Recommendation  
    """)
    st.info("Use the sidebar to explore each module.")

elif menu == "🏭 Air Quality":
    st.markdown("<h2 style='font-size: 25px;'>🏭 Air Quality Monitoring</h2>", unsafe_allow_html=True)
    air_image = safe_open_image("aqiimage.jpeg")
    if air_image:
        st.image(air_image, caption="Air Quality Monitoring", width=800)
    st.markdown("""
     Air quality tells us how clean or polluted the air is.
     We mainly look at pollutants like PM2.5, PM10, CO, NO₂, SO₂ and O₃.
     Here, we use real forecasted pollutant levels and a trained ML model
     to predict AQI for the next 5 days.
    """)
    st.divider()

    aqi_data_local = safe_read_csv("cleaned_aqi_data.csv")
    aqi_data_local = aqi_data_local[aqi_data_local["AQI"] <= 500]
    cities = sorted(aqi_data_local["City"].unique().tolist())

    city = st.selectbox("Select City", cities)

    if city is None:
        st.error("No city selected.")
        st.stop()

    entry = get_or_train_aqi_model(city, aqi_data_local)

    forecast_daily = None
    forecast_error = None
    try:
        lat, lon, matched_name, country = get_coordinates(city)
        forecast_daily = get_pollutant_forecast(lat, lon, days=5)
    except (requests.exceptions.RequestException, ValueError, KeyError) as e:
        forecast_error = str(e)

    if entry is None:
        st.error(f"⚠️ Not enough clean data for '{city}' to train a model.")
    elif forecast_daily is None:
        st.warning(f"Could not load forecast data: {forecast_error}")
    else:
        current_year = entry["last_year"]
        rows = []
        for _, day_row in forecast_daily.iterrows():
            predicted_aqi = predict_daily_aqi(entry, day_row, current_year)
            category = categorize_aqi(predicted_aqi)
            rows.append({
                "Date": day_row["date"],
                "AQI": round(predicted_aqi, 1),
                "Category": category,
                "Dominant Pollutant": dominant_pollutant(day_row),
            })
        forecast_table = pd.DataFrame(rows)

        st.markdown("## 📅 Next 5 Days AQI Forecast")
        day_cols = st.columns(len(forecast_table))
        category_display = {
            "Good": "success",
            "Moderate": "info",
            "Unhealthy for Sensitive Groups": "warning",
            "Unhealthy": "warning",
            "Very Unhealthy": "error",
            "Hazardous": "error",
        }
        for col, (_, row) in zip(day_cols, forecast_table.iterrows()):
            with col:
                st.markdown(f"**{row['Date']}**")
                st.metric("AQI", f"{row['AQI']:.0f}")
                level = category_display.get(row["Category"], "info")
                getattr(st, level)(row["Category"])
                st.markdown(f"<p style='color:#1a1a1a;'>Main: {row['Dominant Pollutant']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='color:#1a1a1a;'>{HEALTH_TIPS.get(row['Category'], '')}</p>", unsafe_allow_html=True)

        category_colors = {
            "Good": "#2e7d32",
            "Moderate": "#9e9d24",
            "Unhealthy for Sensitive Groups": "#f9a825",
            "Unhealthy": "#ef6c00",
            "Very Unhealthy": "#c62828",
            "Hazardous": "#6a1b9a",
        }
        bar_colors = [category_colors.get(cat, "#1b5e20") for cat in forecast_table["Category"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=forecast_table["Date"].astype(str),
            y=forecast_table["AQI"],
            marker_color=bar_colors,
            text=forecast_table["Category"],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"5-Day AQI Forecast for {city}",
            xaxis_title="Date", yaxis_title="AQI",
            showlegend=False,
        )
        st.plotly_chart(fig)

        st.markdown("## 📝 Summary")
        st.write(build_forecast_summary(city, forecast_table))

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
    st.markdown("<h2 style='font-size: 25px;'>💧 Water Quality Analysis</h2>", unsafe_allow_html=True)
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
    water_data["Quality"] = water_data.apply(lambda r: classify_water_quality(r["DO_avg"], r["BOD_avg"]), axis=1)

    states = water_data["State Name"].unique()
    selected_state = st.selectbox("Select State", states)
    state_data = water_data[water_data["State Name"] == selected_state]
    locations = state_data["Name of Monitoring Location"].unique()
    selected_location = st.selectbox("Select Monitoring Location", locations)
    loc_row = state_data[state_data["Name of Monitoring Location"] == selected_location].iloc[0]

    pH_avg = loc_row["pH_avg"]
    DO_avg = loc_row["DO_avg"]
    BOD_avg = loc_row["BOD_avg"]
    quality = loc_row["Quality"]
    concern = main_water_concern(pH_avg, DO_avg, BOD_avg)

    st.divider()
    st.markdown("### 💧 Results")
    st.write("State:", loc_row["State Name"])
    st.write("Average pH:", "N/A" if pd.isna(pH_avg) else round(pH_avg, 2))
    st.write("Average Dissolved Oxygen:", "N/A" if pd.isna(DO_avg) else round(DO_avg, 2))
    st.write("Average BOD:", "N/A" if pd.isna(BOD_avg) else round(BOD_avg, 2))
    st.markdown(f"<p style='color:#1a1a1a;'><b>Main concern:</b> {concern}</p>", unsafe_allow_html=True)

    if quality == "Good":
        st.success("GOOD Water Quality ")
    elif quality == "Moderate":
        st.warning("MODERATE Water Quality ")
    elif quality == "No data":
        st.info("No data available for this location to determine water quality.")
    else:
        st.error("POOR Water Quality ")

    st.markdown(
        f"<p style='color:#1a1a1a;'><b>Usage guidance:</b> {WATER_USAGE_GUIDANCE.get(quality, '')}</p>",
        unsafe_allow_html=True
    )

    st.markdown("## 📝 Summary")
    st.write(build_water_summary(selected_location, quality, pH_avg, DO_avg, BOD_avg, concern))

    st.divider()
    st.markdown(f"## 📊 {selected_state}: DO vs BOD Across Monitoring Locations")
    quality_colors = {
        "Good": "#2e7d32",
        "Moderate": "#f9a825",
        "Poor": "#c62828",
        "No data": "#9e9e9e",
    }
    scatter_fig = go.Figure()
    for q in ["Good", "Moderate", "Poor", "No data"]:
        subset = state_data[state_data["Quality"] == q]
        if not subset.empty:
            scatter_fig.add_trace(go.Scatter(
                x=subset["DO_avg"], y=subset["BOD_avg"],
                mode="markers", name=q,
                marker=dict(color=quality_colors[q], size=9),
                text=subset["Name of Monitoring Location"],
            ))
    scatter_fig.add_trace(go.Scatter(
        x=[DO_avg], y=[BOD_avg],
        mode="markers", name="Selected Location",
        marker=dict(color="#1b5e20", size=16, symbol="star", line=dict(width=2, color="black")),
        text=[selected_location],
    ))
    scatter_fig.update_layout(
        xaxis_title="Dissolved Oxygen (mg/L)",
        yaxis_title="BOD (mg/L)",
    )
    st.plotly_chart(scatter_fig)
    st.caption(
        "Green = Good, Yellow = Moderate, Red = Poor. The star marks your selected location "
        "compared to every other monitored location in this state."
    )

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

    @st.cache_resource
    def train_soil_model(_soil_data):
        X = _soil_data[["Nitrogen", "Phosphorous", "Potassium", "PH", "Temperature", "Humidity", "Rainfall"]]
        y = _soil_data["Crop"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=300, random_state=42)
        model.fit(X_train, y_train)
        test_prediction = model.predict(X_test)
        accuracy = accuracy_score(y_test, test_prediction)
        return model, accuracy

    model, accuracy = train_soil_model(soil_data)

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