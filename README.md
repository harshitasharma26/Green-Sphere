 Green-Sphere

GreenSphere is a web app made with Python and Streamlit that shows the condition of the environment in India.
It has three modules: Air, Water and Soil.

Live app: https://green-sphere.streamlit.app/

 🏭 Air Quality
- Select a city to see the **AQI right now** and the **AQI forecast for the next 5 days**.
- Pollutant data (PM2.5, PM10, NO₂, SO₂, CO, O₃) comes from the Open-Meteo Air Quality API (free, no API key needed).
- AQI is calculated with the official **CPCB (India) AQI formula**. The pollutant with the highest value decides the AQI and is shown as the main pollutant.
- It uses the CPCB averages: 24 hours for PM2.5, PM10, NO₂, SO₂ and 8 hours for CO and O₃.
- Each day shows the AQI, category (Good, Satisfactory, Moderately Polluted, Poor, Very Poor, Severe) and a health tip, with a colour-coded bar chart.

 💧 Water Quality
- Select a state and a monitoring location to see its pH, Dissolved Oxygen (DO) and BOD.
- The water is rated as **Class A, B, C or Below C** using the CPCB water quality criteria.
- It also shows why a site did not get a better class, and a chart comparing DO and BOD of all sites in the state.

 🌱 Soil & Crop
- Enter Nitrogen, Phosphorous, Potassium, pH, temperature, humidity and rainfall using sliders.
- A **Random Forest** model recommends the best crop with its confidence and two other options.
- Model accuracy is checked with 5-fold cross validation.
- A radar chart compares your soil with the ideal soil of the recommended crop.

 Tech Stack
- Python, Streamlit
- scikit-learn (RandomForestClassifier), pandas, numpy
- Plotly (charts)
- Open-Meteo Air Quality API

