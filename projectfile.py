import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

FOLDER = Path(__file__).resolve().parent

# ------------------------------------------------------------------
# AIR QUALITY
# AQI is calculated with the CPCB (India) formula from pollutant values
# ------------------------------------------------------------------

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

# city coordinates (lat, lon)
CITIES = {
    "Ahmedabad": (23.0225, 72.5714),
    "Aizawl": (23.7271, 92.7176),
    "Amaravati": (16.5731, 80.3575),
    "Amritsar": (31.6340, 74.8723),
    "Bengaluru": (12.9716, 77.5946),
    "Bhopal": (23.2599, 77.4126),
    "Brajrajnagar": (21.8167, 83.9167),
    "Chandigarh": (30.7333, 76.7794),
    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Delhi": (28.6139, 77.2090),
    "Ernakulam": (9.9816, 76.2999),
    "Gurugram": (28.4595, 77.0266),
    "Guwahati": (26.1445, 91.7362),
    "Hyderabad": (17.3850, 78.4867),
    "Jaipur": (26.9124, 75.7873),
    "Jorapokhar": (23.7000, 86.4167),
    "Kochi": (9.9312, 76.2673),
    "Kolkata": (22.5726, 88.3639),
    "Lucknow": (26.8467, 80.9462),
    "Mumbai": (19.0760, 72.8777),
    "Patna": (25.5941, 85.1376),
    "Shillong": (25.5788, 91.8933),
    "Talcher": (20.9500, 85.2167),
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Visakhapatnam": (17.6868, 83.2185),
}

# AQI value at each step (Good, Satisfactory, Moderate, Poor, Very Poor, Severe)
AQI_POINTS = [0, 50, 100, 200, 300, 400, 500]

# pollutant value at each of those steps (CPCB table)
# PM2.5, PM10, NO2, SO2, O3 are in ug/m3 and CO is in mg/m3
BREAKPOINTS = {
    "PM2.5": [0, 30, 60, 90, 120, 250, 380],
    "PM10": [0, 50, 100, 250, 350, 430, 510],
    "NO2": [0, 40, 80, 180, 280, 400, 520],
    "SO2": [0, 40, 80, 380, 800, 1600, 2400],
    "CO": [0, 1, 2, 10, 17, 34, 51],
    "O3": [0, 50, 100, 168, 208, 748, 1287],
}

# (highest AQI, category, colour, health tip)
CATEGORIES = [
    (50, "Good", "#2e7d32", "Air quality is good. Enjoy outdoor activities."),
    (100, "Satisfactory", "#7cb342", "Fine for most people. Very sensitive people may feel slight discomfort."),
    (200, "Moderately Polluted", "#f9a825", "People with asthma or heart problems, kids and older people should avoid long outdoor activity."),
    (300, "Poor", "#ef6c00", "Limit outdoor activity. Sensitive people should stay indoors."),
    (400, "Very Poor", "#c62828", "Avoid outdoor activity and wear a mask if you go out."),
    (500, "Severe", "#880e4f", "Stay indoors and keep windows closed."),
]


def get_subindex(pollutant, value):
    # converts one pollutant value into a number between 0 and 500
    if value is None or pd.isna(value) or value < 0:
        return None
    points = BREAKPOINTS[pollutant]
    for i in range(1, len(points)):
        if value <= points[i]:
            low, high = points[i - 1], points[i]
            aqi_low, aqi_high = AQI_POINTS[i - 1], AQI_POINTS[i]
            return aqi_low + (aqi_high - aqi_low) * (value - low) / (high - low)
    return 500


def calculate_aqi(values):
    # AQI = the highest sub-index, and that pollutant is the "main pollutant"
    subs = {}
    for p in POLLUTANTS:
        s = get_subindex(p, values.get(p))
        if s is not None:
            subs[p] = s
    # CPCB needs at least 3 pollutants and one of them must be PM2.5 or PM10
    if len(subs) < 3 or ("PM2.5" not in subs and "PM10" not in subs):
        return None
    main = max(subs, key=subs.get)
    aqi = int(min(subs[main], 500) + 0.5)
    return aqi, main


def get_category(aqi):
    for limit, name, colour, tip in CATEGORIES:
        if aqi <= limit:
            return name, colour, tip
    return CATEGORIES[-1][1:]


@st.cache_data(ttl=1800)
def get_air_data(lat, lon):
    # hourly data from Open-Meteo: yesterday + next 5 days
    # (yesterday is needed for the 24 hour averages)
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "past_days": 1,
        "forecast_days": 5,
        "timezone": "auto",
    }
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={
        "pm2_5": "PM2.5", "pm10": "PM10", "nitrogen_dioxide": "NO2",
        "sulphur_dioxide": "SO2", "carbon_monoxide": "CO", "ozone": "O3",
    })
    for p in POLLUTANTS:
        df[p] = pd.to_numeric(df[p], errors="coerce")
        df[p] = df[p].where(df[p] >= 0)

    # IMPORTANT: the API gives CO in ug/m3 but CPCB uses mg/m3, so divide by 1000
    # (this was the reason for the very high AQI in the old version)
    df["CO"] = df["CO"] / 1000

    # CPCB averages: 24 hours for PM2.5, PM10, NO2, SO2 and 8 hours for CO, O3
    for p in ["PM2.5", "PM10", "NO2", "SO2"]:
        df[p + " avg"] = df[p].rolling(24, min_periods=16).mean()
    for p in ["CO", "O3"]:
        df[p + " avg"] = df[p].rolling(8, min_periods=6).mean()

    return df, data.get("utc_offset_seconds", 19800)


def get_daily_aqi(df, today):
    days = []
    for date, group in df.groupby(df["time"].dt.date):
        if date < today or len(group) < 16:
            continue
        values = {}
        for p in ["PM2.5", "PM10", "NO2", "SO2"]:
            values[p] = group[p].mean() if group[p].count() >= 16 else np.nan
        values["CO"] = group["CO avg"].max()
        values["O3"] = group["O3 avg"].max()
        result = calculate_aqi(values)
        if result is not None:
            days.append({"date": date, "aqi": result[0], "main": result[1]})
    return days[:5]


def badge(text, colour):
    return ("<span style='background:" + colour + ";color:white;padding:3px 12px;display:inline-block;"
            "border-radius:12px;font-weight:600;'>" + text + "</span>")


def show_chart(fig):
    # white chart so it also looks fine when the user's phone is in dark mode
    fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(color="#1a1a1a"))
    st.plotly_chart(fig, theme=None)


def show_image(name, caption):
    try:
        st.image(Image.open(FOLDER / name), width=800, caption=caption)
    except FileNotFoundError:
        st.warning("Image not found: " + name)


def air_page():
    st.subheader("🏭 Air Quality")
    show_image("aqiimage.jpeg", "Air Quality Monitoring")
    st.write("Pick a city to see today's AQI and the forecast for the next 5 days. "
             "AQI is calculated from PM2.5, PM10, NO₂, SO₂, CO and O₃ with the official Indian (CPCB) formula.")
    st.divider()

    city = st.selectbox("Select City", list(CITIES))
    lat, lon = CITIES[city]

    try:
        df, offset = get_air_data(lat, lon)
    except (requests.exceptions.RequestException, KeyError, ValueError):
        st.error("Could not load air quality data right now. Please try again after some time.")
        return

    # current time in that city
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) + datetime.timedelta(seconds=offset)

    # ---- AQI right now ----
    st.markdown("## 🌐 Air quality right now")
    past = df[df["time"] <= now]
    result = None
    if len(past) > 0 and now - past.iloc[-1]["time"] < datetime.timedelta(hours=3):
        row = past.iloc[-1]
        result = calculate_aqi({p: row[p + " avg"] for p in POLLUTANTS})

    if result is None:
        st.warning("Not enough data to calculate the AQI right now.")
    else:
        aqi, main = result
        name, colour, tip = get_category(aqi)
        c1, c2, c3 = st.columns(3)
        c1.metric("AQI", aqi)
        c2.markdown("Category")
        c2.markdown(badge(name, colour), unsafe_allow_html=True)
        c3.metric("Main pollutant", main)
        st.write(tip)

        # current readings in ug/m3 (CO converted back from mg/m3)
        cols = st.columns(6)
        for col, p in zip(cols, POLLUTANTS):
            value = row[p] * 1000 if p == "CO" else row[p]
            col.metric(p + " (µg/m³)", "-" if pd.isna(value) else str(round(value)))
        st.caption("AQI uses the last 24 hours average (8 hours for CO and O₃), like the CPCB method.")

    # ---- next 5 days ----
    st.divider()
    st.markdown("## 📅 Next 5 days forecast")
    days = get_daily_aqi(df, now.date())
    if len(days) == 0:
        st.warning("Forecast data is not available.")
        return

    cols = st.columns(len(days))
    for col, d in zip(cols, days):
        name, colour, tip = get_category(d["aqi"])
        col.markdown("**" + d["date"].strftime("%a %d %b") + "**")
        col.metric("AQI", d["aqi"])
        col.markdown(badge(name, colour), unsafe_allow_html=True)
        col.write("Main: " + d["main"])

    fig = go.Figure(go.Bar(
        x=[d["date"].strftime("%a %d %b") for d in days],
        y=[d["aqi"] for d in days],
        marker_color=[get_category(d["aqi"])[1] for d in days],
        text=[get_category(d["aqi"])[0] for d in days],
        textposition="outside",
    ))
    fig.update_layout(title="5 day AQI forecast for " + city, xaxis_title="Date", yaxis_title="AQI")
    show_chart(fig)

    best = min(days, key=lambda d: d["aqi"])
    worst = max(days, key=lambda d: d["aqi"])
    st.write("**Summary:** the best day looks like " + best["date"].strftime("%a %d %b") + " (AQI " + str(best["aqi"]) +
             ") and the worst day is " + worst["date"].strftime("%a %d %b") + " (AQI " + str(worst["aqi"]) + ").")
    st.caption("These values come from an air quality model (Open-Meteo), so they can be a little different from station readings.")


# ------------------------------------------------------------------
# WATER QUALITY
# rating is based on CPCB water quality classes (pH, DO and BOD)
# ------------------------------------------------------------------

# class: (best use, lowest pH, highest pH, minimum DO, maximum BOD)
WATER_CLASSES = {
    "A": ("drinking water source (after disinfection)", 6.5, 8.5, 6, 2),
    "B": ("outdoor bathing", 6.5, 8.5, 5, 3),
    "C": ("drinking water source (after normal treatment)", 6.0, 9.0, 4, 3),
}
WATER_COLOURS = {"A": "#2e7d32", "B": "#7cb342", "C": "#f9a825", "Below C": "#c62828"}


def failed_checks(rule, ph_min, ph_max, do, bod):
    # returns the things that do not meet this class
    use, ph_low, ph_high, do_min, bod_max = rule
    failed = []
    if ph_min < ph_low or ph_max > ph_high:
        failed.append("pH is " + str(ph_min) + " to " + str(ph_max) + " (needs " + str(ph_low) + " to " + str(ph_high) + ")")
    if do < do_min:
        failed.append("dissolved oxygen is " + str(do) + " (needs at least " + str(do_min) + ")")
    if bod > bod_max:
        failed.append("BOD is " + str(bod) + " (needs at most " + str(bod_max) + ")")
    return failed


def get_water_class(ph_min, ph_max, do, bod):
    # we use the worst values (lowest DO, highest BOD) to be on the safe side
    for name in ["A", "B", "C"]:
        if len(failed_checks(WATER_CLASSES[name], ph_min, ph_max, do, bod)) == 0:
            return name
    return "Below C"


@st.cache_data
def load_water_data():
    raw = pd.read_csv(FOLDER / "cleaned_water_data.csv")
    water = pd.DataFrame()
    water["State"] = raw["State Name"].apply(lambda x: " ".join(str(x).split()).title())
    water["Location"] = raw["Name of Monitoring Location"].apply(lambda x: " ".join(str(x).split()))
    water["pH_min"] = pd.to_numeric(raw["Min pH"], errors="coerce")
    water["pH_max"] = pd.to_numeric(raw["Max pH"], errors="coerce")
    water["DO_min"] = pd.to_numeric(raw["Min Dissolved Oxygen"], errors="coerce")
    water["DO_max"] = pd.to_numeric(raw["Max Dissolved Oxygen"], errors="coerce")
    water["BOD_min"] = pd.to_numeric(raw["Min BOD"], errors="coerce")
    water["BOD_max"] = pd.to_numeric(raw["Max BOD"], errors="coerce")
    water = water.dropna(subset=["pH_min", "pH_max", "DO_min", "BOD_max"]).reset_index(drop=True)
    water["Class"] = [get_water_class(r.pH_min, r.pH_max, r.DO_min, r.BOD_max) for r in water.itertuples()]
    return water


def water_page():
    st.subheader("💧 Water Quality")
    show_image("water_image.jpeg", "Water Quality Monitoring")
    st.write("Water quality depends on pH (acidic or basic), dissolved oxygen (DO, higher is better) "
             "and BOD (pollution level, lower is better). Sites are compared with the CPCB classes A, B and C.")
    st.divider()

    water = load_water_data()
    state = st.selectbox("Select State", sorted(water["State"].unique()))
    state_data = water[water["State"] == state].sort_values("Location").reset_index(drop=True)
    number = st.selectbox("Select Monitoring Location", range(len(state_data)),
                          format_func=lambda i: state_data.loc[i, "Location"])
    site = state_data.loc[number]

    st.markdown("### 💧 Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("pH (min - max)", str(site["pH_min"]) + " - " + str(site["pH_max"]))
    c2.metric("Dissolved oxygen (mg/L)", str(site["DO_min"]) + " - " + str(site["DO_max"]))
    c3.metric("BOD (mg/L)", str(site["BOD_min"]) + " - " + str(site["BOD_max"]))

    cls = site["Class"]
    label = "Class " + cls if cls != "Below C" else "Below Class C"
    st.markdown("**Rating:** " + badge(label, WATER_COLOURS[cls]), unsafe_allow_html=True)

    if cls in WATER_CLASSES:
        st.write("**Best use:** " + WATER_CLASSES[cls][0])
    else:
        st.write("**Best use:** not suitable for drinking or bathing.")

    # what stops it from getting a better class
    if cls != "A":
        order = ["A", "B", "C"]
        better = order[order.index(cls) - 1] if cls in order else "C"
        problems = failed_checks(WATER_CLASSES[better], site["pH_min"], site["pH_max"], site["DO_min"], site["BOD_max"])
        st.write("**Why it is not Class " + better + ":**")
        for problem in problems:
            st.write("- " + problem)
    st.caption("Only pH, DO and BOD are checked, so this does not prove that the water is safe to drink.")

    st.divider()
    st.markdown("## 📊 " + state + ": DO vs BOD of all sites")
    fig = go.Figure()
    for name in ["A", "B", "C", "Below C"]:
        part = state_data[state_data["Class"] == name]
        if len(part) > 0:
            fig.add_trace(go.Scatter(
                x=part["DO_min"], y=part["BOD_max"], mode="markers",
                name="Class " + name if name != "Below C" else "Below Class C",
                marker=dict(color=WATER_COLOURS[name], size=9), text=part["Location"]))
    fig.add_trace(go.Scatter(
        x=[site["DO_min"]], y=[site["BOD_max"]], mode="markers", name="Selected site",
        marker=dict(color="#1b5e20", size=16, symbol="star", line=dict(width=2, color="black"))))
    fig.update_layout(xaxis_title="Lowest DO (mg/L)", yaxis_title="Highest BOD (mg/L, log scale)", yaxis_type="log")
    show_chart(fig)
    st.caption("Sites at the bottom right (more oxygen, less BOD) are cleaner. The star is the selected site.")


# ------------------------------------------------------------------
# SOIL AND CROP
# ------------------------------------------------------------------

FEATURES = ["Nitrogen", "Phosphorous", "Potassium", "PH", "Temperature", "Humidity", "Rainfall"]
FEATURE_NAMES = {
    "Nitrogen": "Nitrogen (N)", "Phosphorous": "Phosphorous (P)", "Potassium": "Potassium (K)",
    "PH": "Soil pH", "Temperature": "Temperature (°C)", "Humidity": "Humidity (%)", "Rainfall": "Rainfall (mm)",
}


@st.cache_data
def load_soil_data():
    return pd.read_csv(FOLDER / "crop_recommendation_dataset.csv")


@st.cache_resource
def train_soil_model():
    soil = load_soil_data()
    X = soil[FEATURES]
    y = soil["Crop"]
    # accuracy with 5-fold cross validation
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(RandomForestClassifier(n_estimators=300, random_state=42), X, y, cv=folds)
    # final model uses all the data
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X, y)
    return model, scores.mean()


def soil_page():
    st.subheader("🌱 Soil & Crop Recommendation")
    show_image("soil_image.jpeg", "Healthy soil, healthy crops")
    st.write("Enter the soil and weather values and the model will suggest the best crop.")
    st.divider()

    soil = load_soil_data()
    model, accuracy = train_soil_model()

    st.markdown("## Enter Soil Conditions")
    st.caption("The sliders only go up to the values present in the dataset, because the model cannot predict well outside them.")
    values = {}
    col1, col2 = st.columns(2)
    for i, f in enumerate(FEATURES):
        column = col1 if i % 2 == 0 else col2
        low, high, middle = soil[f].min(), soil[f].max(), soil[f].median()
        with column:
            if f in ["Nitrogen", "Phosphorous", "Potassium"]:
                values[f] = st.slider(FEATURE_NAMES[f], int(np.floor(low)), int(np.ceil(high)), int(round(middle)))
            else:
                values[f] = st.slider(FEATURE_NAMES[f], float(np.floor(low * 10) / 10),
                                      float(np.ceil(high * 10) / 10), float(round(middle, 1)), step=0.1)

    # predict_proba gives the chance of every crop
    chances = model.predict_proba(pd.DataFrame([values], columns=FEATURES))[0]
    top = np.argsort(chances)[::-1][:3]
    best_crop = model.classes_[top[0]]

    st.markdown("## 🌾 Recommended Crop")
    st.success("Best crop for this soil: " + best_crop.upper() + " (confidence " + str(round(chances[top[0]] * 100)) + "%)")
    others = [model.classes_[i].title() + " (" + str(round(chances[i] * 100)) + "%)" for i in top[1:] if chances[i] > 0]
    if len(others) > 0:
        st.write("**Other options:** " + ", ".join(others))
    st.metric("Model accuracy (5-fold cross validation)", str(round(accuracy * 100, 1)) + "%")

    # radar chart: your soil vs the average soil of the recommended crop
    st.markdown("## 📊 Your soil vs ideal soil for " + best_crop.title())
    crop_average = soil[soil["Crop"] == best_crop][FEATURES].mean()
    your_scaled, crop_scaled = [], []
    for f in FEATURES:
        low, high = soil[f].min(), soil[f].max()
        your_scaled.append(100 * (values[f] - low) / (high - low))
        crop_scaled.append(100 * (crop_average[f] - low) / (high - low))
    names = FEATURES + [FEATURES[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=crop_scaled + crop_scaled[:1], theta=names, fill="toself",
                                  name="Ideal for " + best_crop.title(), opacity=0.5))
    fig.add_trace(go.Scatterpolar(r=your_scaled + your_scaled[:1], theta=names, fill="toself",
                                  name="Your soil", opacity=0.5))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
    show_chart(fig)
    st.caption("Every axis is scaled from 0 to 100 using the dataset's minimum and maximum. The more the shapes overlap, the better the match.")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

STYLE = """
<style>
.stApp { background-color: #f5f7f6; }
.stApp p, .stApp li, .stApp label { color: #1a1a1a !important; }
h1, h2, h3 { color: #1b5e20 !important; }
section[data-testid="stSidebar"] { background-color: #2e7d32 !important; }
section[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSelectbox"] > div > div { background-color: white !important; color: #1a1a1a !important; }
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: #1a1a1a !important; background-color: transparent !important; }
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: #1a1a1a !important; }
[data-testid="stImageCaption"] { color: #555555 !important; }
</style>
"""


def intro_page():
    st.subheader("Analyzing Air, Water and Soil Conditions for a Sustainable Environment")
    show_image("environmental-protection-326923_1280.jpg", "Protecting our environment with data")
    st.write("GreenSphere has three parts:\n\n"
             "- 🏭 **Air** - AQI today and for the next 5 days\n"
             "- 💧 **Water** - how clean the water is at a monitoring site\n"
             "- 🌱 **Soil** - the best crop for your soil, predicted with machine learning")
    st.info("Use the sidebar to open each module.")


def main():
    st.set_page_config(page_title="GreenSphere", page_icon="🌍", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)

    with st.sidebar:
        st.title("🌿 GreenSphere")
        st.markdown("---")
        menu = st.radio("Go to", ["🏠 Introduction", "🏭 Air Quality", "💧 Water Quality", "🌱 Soil & Crop"])

    st.title("🌍 GreenSphere")
    st.header("Environmental Monitoring for India: Air, Water & Soil")

    if menu == "🏠 Introduction":
        intro_page()
    elif menu == "🏭 Air Quality":
        air_page()
    elif menu == "💧 Water Quality":
        water_page()
    else:
        soil_page()


if __name__ == "__main__":
    main()