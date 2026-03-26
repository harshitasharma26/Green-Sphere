
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.feature_selection import RFE
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
st.set_page_config(
    page_title="GreenSphere",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color:#f5f7f6 ;
}
[data-testid="stAppViewContainer"] * {
    color: #1a1a1a !important;
}

[data-testid="stSidebar"] {
    background-color: #2e7d32 !important;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="collapsedControl"] {
    display: none;
}

h1, h2, h3 {
    color: #1b5e20 !important ;
}

body {
    color: #1a1a1a;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🌿 GreenSphere")
    st.markdown("---")
    menu = st.radio("Go to", ["🏠 Introduction", "🏭 Air Quality", "💧 Water Quality", "🌱 Soil & Crop"])

st.title("🌍 GreenSphere")

if menu == "🏠 Introduction":
    st.markdown("<h6 style='font-size: 25px;'>Analyzing Air, Water and Soil Conditions for a Sustainable Environment</h6>", unsafe_allow_html=True)
    intro_image = Image.open("environmental-protection-326923_1280.jpg") 
    st.image(intro_image, width=800, caption="Protecting our environment with AI")
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
    air_image = Image.open("aqiimage.jpeg")
    st.image(air_image, caption="Air Quality Monitoring", width= 800)
    st.markdown("""
     Air quality tells us how clean or polluted the air is.
     We mainly look at pollutants like PM2.5, PM10, CO, NO₂, SO₂ and O₃.
     Here, we use past data and machine learning to predict AQI.
    """)
    st.divider()
    aqi_data = pd.read_csv("cleaned_aqi_data.csv")
    aqi_data = aqi_data[aqi_data['AQI'] <= 500]
    city = st.selectbox("Select City", aqi_data['City'].unique())
    city_data = aqi_data[aqi_data['City'] == city]
    features = ['Year','PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3'] 
    X = city_data[features]
    y = city_data['AQI']
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    year = st.number_input(
      "Enter Year",
      min_value=2015,
      max_value=2100,
      step=1
)
    
    recent_data = city_data[city_data['Year'] >= city_data['Year'].max() - 3]
    avg_values = recent_data[['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']].mean()
    user_input = [[
        year,
        avg_values['PM2.5'],
        avg_values['PM10'],
        avg_values['NO2'],
        avg_values['CO'],
        avg_values['SO2'],
        avg_values['O3'],
        
    ]]
    prediction = model.predict(user_input)[0]
    y_pred = model.predict(X_test)  
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
      st.subheader(f"Predicted AQI")
      st.write(f"{prediction:.0f}")

    with col2:
      st.subheader("Model RMSE")
      st.write(f"{rmse:.2f}")
    if prediction <= 50:
       st.success("Good  — Air quality is satisfactory")
    elif prediction <= 100:
         st.info("Moderate  — Acceptable quality")
    elif prediction <= 150:
        st.warning("Unhealthy for Sensitive Groups ")
    elif prediction <= 200:
        st.warning("Unhealthy  — Everyone may be affected")
    elif prediction <= 300:
         st.error("Very Unhealthy  — Health alert")
    else:
         st.error("Hazardous  — Emergency conditions")
    fig = px.scatter(
     city_data,
     x='Year',
     y='AQI',
     title=f"AQI Trend for {city}"
)

    fig.add_scatter(
    x=[year],
    y=[prediction],
    mode='markers',
    marker=dict(size=12),
    name='Prediction'
)

    st.plotly_chart(fig)

elif menu == "💧 Water Quality":
 st.markdown("<h2 style='font-size: 25px;'>💧 Water Quality Analysis</h2>",unsafe_allow_html=True)
 water_image = Image.open("water_image.jpeg")
 st.image(water_image, caption="Water Quality Monitoring", width=800)    
 st.markdown("""
 Water quality depends on a few important factors like pH, dissolved oxygen and BOD.
 - pH shows if water is acidic or basic  
 - DO tells how much oxygen is available  
 - BOD indicates pollution level  
 Based on these values, we classify water as good, moderate or poor.
 """)
 st.divider()
 water_data = pd.read_csv("cleaned_water_data.csv")
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

 if DO_avg >= 6 and BOD_avg <= 3:
        quality = "Good"
 elif DO_avg >= 5 and BOD_avg <= 3:
        quality = "Moderate"
 else:
        quality = "Poor"
 st.divider()
 st.markdown("### 💧 Results")
 st.write("State:", loc_row["State Name"])
 st.write("Average pH:", round(pH_avg, 2))
 st.write("Average Dissolved Oxygen:", round(DO_avg, 2))
 st.write("Average BOD:", round(BOD_avg, 2))
    
 if quality == "Good":
        st.success("GOOD Water Quality ")
 elif quality == "Moderate":
        st.warning("MODERATE Water Quality ")
 else:
        st.error("POOR Water Quality ")

elif menu == "🌱 Soil & Crop":
 st.markdown("<h2 style='font-size: 25px;'>🌱 Soil Fertility & Crop Recommendation</h2>", unsafe_allow_html=True)
 soil_image= Image.open("soil_image.jpeg")
 st.image(soil_image, caption="Healthy soil, healthy crops",width=800)
 st.markdown("""
 Soil fertility depends on nutrients like nitrogen, phosphorous and potassium.
 Other factors like pH, temperature, humidity and rainfall also matter.
 Using these values, the system suggests the best crop using machine learning.
 """)
 st.divider()
 soil_data = pd.read_csv("crop_recommendation_dataset.csv")
 X = soil_data[["Nitrogen", "Phosphorous", "Potassium", "PH","Temperature", "Humidity", "Rainfall"]]   
 y = soil_data["Crop"]  
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 model = LogisticRegression(max_iter=1000)
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
 nutrients = ["Nitrogen", "Phosphorous", "Potassium"]
 values = [Nitrogen, Phosphorous, Potassium]
 fig = go.Figure(data=[
    go.Bar(x=nutrients, y=values)
])
 fig.update_layout(
    title="Soil Nutrient Levels",
    xaxis_title="Nutrients",
    yaxis_title="Value"
)
 st.plotly_chart(fig)
