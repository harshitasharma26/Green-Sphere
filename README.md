# Green-Sphere
 GreenSphere is a web-based environmental monitoring system built using 
Python and Streamlit. It uses Machine Learning to analyze and predict 
environmental conditions across India.
 https://green-sphere.streamlit.app/
The project has three modules:

 🏭 Air Quality
 
Select any city and get a 5-day AQI forecast, built from real forecasted pollutant data (via the Open-Meteo Air Quality API) fed into a trained ML model (RandomForest / GradientBoosting, whichever performs best per city).
Each day shows predicted AQI, health category, the dominant pollutant driving that day's reading, and a practical health tip.
Color-coded bar chart of the 5-day forecast.
Auto-generated plain-language summary (best/worst day, dominant pollutant, days requiring caution).
Current Conditions (Live) — real-time pollutant readings (PM2.5, PM10, NO2, CO, SO2, O3) for the selected city.

💧 Water Quality

Select a state and monitoring location to see average pH, Dissolved Oxygen (DO), and BOD levels.
Automatic quality classification (Good / Moderate / Poor) based on DO and BOD thresholds.
Main concern indicator — flags which parameter (pH, DO, or BOD) is furthest from a healthy range.
Usage guidance — practical recommendations based on the quality rating (drinking, irrigation, etc.).
Auto-generated summary text.
Scatter chart comparing DO vs BOD across every monitored location in the selected state, with the chosen location highlighted.

🌱 Soil & Crop

Enter soil nutrient levels (Nitrogen, Phosphorous, Potassium) and environmental conditions (pH, Temperature, Humidity, Rainfall) via sliders.
ML-based (RandomForestClassifier) crop recommendation.
Soil fertility rating (High / Moderate / Low).
Radar chart comparing your soil profile against the ideal profile for the recommended crop.

Tech Stack

Frontend / App framework: Streamlit
ML models: scikit-learn (RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor)
Data handling: pandas, numpy
Visualizations: Plotly
Live data: Open-Meteo Geocoding API and Open-Meteo Air Quality API (free, no API key required)
