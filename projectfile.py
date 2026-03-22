import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
st.set_page_config(page_title="Environmental AI System", layout="wide")
st.title("🌍 GreenSphere")
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to",["Introduction","Air Quality", "Water Quality", "Soil & Crop"])
if menu == "Introduction":
    st.markdown(
    "<h6 style='font-size: 25px;'>🌿 ML-Powered Environmental Monitoring for a Sustainable Future</h6>",
    unsafe_allow_html=True
)
    intro_image = Image.open("environmental-protection-326923_1280.jpg") 
    st.image(intro_image, width=800, caption="Protecting our environment with AI")
    st.subheader("About GreenSphere")
    st.write(
        """
    This AI-based system monitors environmental conditions using machine learning
    to support sustainable decision-making.

    It analyzes:

    • 🌫 Air Quality  
    • 💧 Water Safety  
    • 🌱 Soil Fertility & Crop Recommendation  
    """
    )
    st.info("Use the sidebar to explore each module.")
elif menu == "Air Quality":
 st.markdown("<h6 style='font-size: 25px;'>🏭 Air Quality Monitoring</h6>",
         unsafe_allow_html=True)
 air_image = Image.open("aqiimage.jpeg")
 st.image(air_image, caption="Air Quality Monitoring", width= 800)
 st.markdown("""
        **Air Quality** refers to the condition of the air in terms of pollutants that can affect human health and the environment.  
        
        **Key Pollutants:** Particulate matter (PM2.5, PM10), CO, NO₂, SO₂, O₃.  
        
        **Air Quality Index (AQI):** A number from 0–500 indicating air cleanliness:  
        - 0–50: Good  
        - 51–100: Moderate  
        - 101–150: Unhealthy for sensitive groups  
        - 151–200: Unhealthy  
        - 201–300: Very Unhealthy  
        - 301–500: Hazardous  
        
        **Ways to Improve Air Quality:** Reduce vehicle use, plant trees, regulate industrial emissions.
        """)
 st.divider()
 aqi_data = pd.read_csv("cleaned_aqi_data.csv")
 aqi_data = aqi_data[aqi_data['AQI'] <= 500]
 city = st.selectbox("Select City", aqi_data['City'].unique())
 city_data = aqi_data[aqi_data['City'] == city]
 features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'Month'] 
 X = city_data[features]
 y = city_data['AQI']
 X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
 model = RandomForestRegressor(n_estimators=300, random_state=42)
 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)
 rmse = np.sqrt(mean_squared_error(y_test, y_pred))
 year = st.selectbox("Select Year", sorted(city_data['Year'].unique()))
 year_data = city_data[city_data['Year'] == year][features].mean()
 prediction = model.predict([year_data])[0]
 st.subheader(f"Predicted AQI for {city} in {year}: {prediction:.0f}")
 if prediction <= 50:
       st.success("Good 🟢 — Air quality is satisfactory")
 elif prediction <= 100:
         st.info("Moderate 🟡 — Acceptable quality")
 elif prediction <= 150:
        st.warning("Unhealthy for Sensitive Groups 🟠")
 elif prediction <= 200:
        st.warning("Unhealthy 🔴 — Everyone may be affected")
 elif prediction <= 300:
         st.error("Very Unhealthy 🟣 — Health alert")
 else:
         st.error("Hazardous ☠️ — Emergency conditions")
 st.metric(label="AQI Model RMSE", value=f"{rmse:.2f}")
 fig3 = px.scatter(
    city_data,
    x='Year',
    y='AQI',
    title=f"AQI Trend for {city}"
)

 fig3.add_scatter(
    x=[year],
    y=[prediction],
    mode='markers',
    marker=dict(size=12),
    name='Prediction'
)

 st.plotly_chart(fig3)

elif menu == "Water Quality":

    st.markdown("<h2>💧 Water Quality Analysis</h2>", unsafe_allow_html=True)

    water_image = Image.open("water_image.jpeg")
    st.image(water_image, caption="Water Quality Monitoring", width=800)

    st.markdown("""
        **Water Quality** is measured using these key parameters:
        - **pH** — acidity/alkalinity (safe range: 6.5 to 8.5)
        - **Dissolved Oxygen (DO)** — oxygen in water (higher = better)
        - **BOD** — Biological Oxygen Demand (lower = better)
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
    DO_avg  = loc_row["DO_avg"]
    BOD_avg = loc_row["BOD_avg"]

    if DO_avg >= 6 and BOD_avg <= 3:
        quality = "Good"
    elif DO_avg >= 5 and BOD_avg <= 3:
        quality = "Moderate"
    else:
        quality = "Poor"

    st.markdown("### 💧 Results")
    st.write("**State:**", loc_row["State Name"])
    st.write("**Average pH:**", round(pH_avg, 2))
    st.write("**Average Dissolved Oxygen:**", round(DO_avg, 2))
    st.write("**Average BOD:**", round(BOD_avg, 2))
    

    if quality == "Good":
        st.success("GOOD Water Quality ✅")
    elif quality == "Moderate":
        st.warning("MODERATE Water Quality ⚠️")
    else:
        st.error("POOR Water Quality ❌")

    sunburst_data = pd.DataFrame({
        "State": [loc_row["State Name"]],
        "Location": [loc_row["Name of Monitoring Location"]],
        "Quality": [quality]
    })
    fig = px.sunburst(sunburst_data, path=["State", "Location", "Quality"])
    st.plotly_chart(fig)

elif menu == "Soil & Crop":
 st.markdown(    "<h2 style='font-size: 25px;'>🌱 Soil Fertility & Crop Recommendation</h2>",
        unsafe_allow_html=True
    )
 soil_image= Image.open("soil_image.jpeg")
 st.image(soil_image, caption="Healthy soil, healthy crops",width=800)
 st.markdown("""
    **Soil Fertility** refers to the ability of soil to provide essential nutrients for healthy plant growth.  
    
    **Key Factors:** Nitrogen (N), Phosphorus (P), Potassium (K), soil pH, moisture, and organic matter.  
    
    **Importance:** Fertile soil supports higher crop yield, better plant health, and sustainable agriculture.  
    
    **Crop Recommendation:** Different crops require different soil conditions; analyzing soil properties helps select the most suitable crop.  
    
    **Improving Soil Fertility:** Use organic fertilizers, crop rotation, composting, and proper irrigation.
    """)
 st.divider()
 soil_data = pd.read_csv("crop_recommendation_dataset.csv")
 X = soil_data[["Nitrogen", "Phosphorous", "Potassium", "PH","Temperature", "Humidity", "Rainfall"]]   
 y = soil_data["Crop"]  
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 sc=StandardScaler()
 sc.fit(X_train)
 x_train=sc.fit_transform(X_train)
 x_test=sc.transform(X_test)
 model = LogisticRegression(max_iter=50000,random_state=42)               
 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)
 st.markdown("## 🎛️ Enter Soil Conditions")
 Nitrogen= st.slider("Nitrogen (N)", 0, 150, 50)
 Phosphorous= st.slider("Phosphorous (P)", 0, 150, 50)
 Potassium = st.slider("Potassium (K)", 0, 150, 50)
 PH = st.slider("Soil PH", 0.0, 14.0, 7.0)
 Temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
 Humidity    = st.slider("Humidity (%)", 0.0, 100.0, 50.0)
 Rainfall    = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)
 prediction = model.predict([[Nitrogen,Phosphorous, Potassium,PH,Temperature, Humidity, Rainfall]])
 st.markdown("## 🌾 Recommended Crop")
 st.success(f"Best crop for this soil: 🌿 {prediction[0].upper()}")
 accuracy = accuracy_score(y_test, y_pred)
 st.metric(label="Soil Model Accuracy", value=f"{accuracy * 100:.1f}%")
 st.markdown("## 🌱 Soil Fertility")
 if Nitrogen > 80 and Phosphorous > 80 and Potassium > 80:
        st.success("Soil fertility is HIGH 🌿")
 elif Nitrogen > 40 and Phosphorous > 40 and Potassium > 40:
        st.warning("Soil fertility is MODERATE ⚠️")
 else:
        st.error("Soil fertility is LOW ❌")
