"""
GreenSphere Air Quality API
----------------------------
FastAPI backend that serves AQI predictions for the Air Quality module.

Why this exists:
The original Streamlit app retrained a RandomForestRegressor on every single
user interaction (every slider move, every selectbox change), because
Streamlit reruns the whole script top-to-bottom on each interaction. That's
wasteful and slow. This API trains a model for a given city ONCE (the first
time it's requested) and caches it in memory, so subsequent predictions for
that city are instant.

Run with:
    uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

app = FastAPI(
    title="GreenSphere Air Quality API",
    description="Predicts AQI for a given city and year using a RandomForestRegressor.",
    version="1.0.0",
)

DATA_PATH = "cleaned_aqi_data.csv"
FEATURES = ["Year", "PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]

# Loaded once at startup
_aqi_data: pd.DataFrame | None = None

# Cache of trained models per city, e.g. {"Delhi": {"model": ..., "rmse": ..., "avg_values": ...}}
_model_cache: dict = {}

# Cache of city -> (latitude, longitude), so we don't re-geocode the same city every time
_coords_cache: dict = {}


def _geocode_city(city: str):
    """Convert a city name into (lat, lon) using Open-Meteo's free Geocoding API."""
    if city in _coords_cache:
        return _coords_cache[city]

    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if "results" not in data:
        raise HTTPException(status_code=404, detail=f"Could not geocode city '{city}'")

    result = data["results"][0]
    coords = (result["latitude"], result["longitude"])
    _coords_cache[city] = coords
    return coords


def _fetch_live_air_quality(lat: float, lon: float):
    """Fetch CURRENT (right-now) pollutant levels from Open-Meteo's Air Quality API."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()["current"]


@app.on_event("startup")
def load_data():
    """Load the dataset once when the API starts, instead of on every request."""
    global _aqi_data
    df = pd.read_csv(DATA_PATH)
    df = df[df["AQI"] <= 500]
    _aqi_data = df


def _get_or_train_model(city: str):
    """
    Return a cached model for the city, training it the first time it's requested.

    Improvements over the original single RandomForest:
    - Tunes RandomForest hyperparameters (limits tree depth / leaf size to
      reduce overfitting, which a totally unconstrained forest is prone to
      on a small per-city dataset).
    - Also trains a GradientBoostingRegressor and keeps whichever model
      scores a lower RMSE via 5-fold cross-validation (a more honest,
      stable estimate than a single train/test split).
    """
    if city in _model_cache:
        return _model_cache[city]

    if _aqi_data is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")

    city_data = _aqi_data[_aqi_data["City"] == city]
    if city_data.empty:
        raise HTTPException(status_code=404, detail=f"No data found for city '{city}'")

    # Drop rows with missing values in the features/target used for training.
    # NaN values here would otherwise cause model fitting to silently fail
    # inside cross-validation (returning NaN scores instead of a real error),
    # leaving no model selected at all.
    required_cols = FEATURES + ["AQI"]
    city_data = city_data.dropna(subset=required_cols)
    if len(city_data) < 10:
        raise HTTPException(
            status_code=422,
            detail=f"Not enough clean (non-missing) data for city '{city}' to train a model "
                   f"(only {len(city_data)} usable rows after removing missing values).",
        )

    X = city_data[FEATURES]
    y = city_data["AQI"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    n_rows = len(city_data)
    # Guard against tiny per-city datasets where 5-fold CV isn't meaningful
    cv_folds = min(5, max(2, n_rows // 10)) if n_rows >= 10 else None

    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        ),
    }

    best_name, best_model, best_cv_rmse = None, None, float("inf")
    for name, candidate in candidates.items():
        if cv_folds:
            scores = cross_val_score(
                candidate, X, y, cv=cv_folds, scoring="neg_mean_squared_error"
            )
            cv_rmse = float(np.sqrt(-scores.mean()))
        else:
            # Too little data for cross-validation -- fall back to the single split
            candidate.fit(X_train, y_train)
            cv_rmse = float(np.sqrt(mean_squared_error(y_test, candidate.predict(X_test))))

        if cv_rmse < best_cv_rmse:
            best_name, best_model, best_cv_rmse = name, candidate, cv_rmse

    if best_model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model training failed for city '{city}' -- all candidate models "
                   f"produced invalid (NaN) scores. This usually means the data still "
                   f"contains unexpected values after cleaning.",
        )

    # Fit the winning model type on the actual train split for the final model,
    # and report RMSE on the held-out test set (consistent with how it was measured before)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    recent_data = city_data[city_data["Year"] >= city_data["Year"].max() - 3]
    avg_values = recent_data[["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]].mean()

    entry = {
        "model": best_model,
        "model_name": best_name,
        "rmse": rmse,
        "cv_rmse": round(best_cv_rmse, 2),
        "avg_values": avg_values,
        "trend": city_data[["Year", "AQI"]].to_dict(orient="records"),
    }
    _model_cache[city] = entry
    return entry


class PredictRequest(BaseModel):
    city: str
    year: int = Field(..., ge=2015, le=2100)


class PredictResponse(BaseModel):
    city: str
    year: int
    predicted_aqi: float
    rmse: float
    cv_rmse: float
    model_name: str
    category: str
    used_live_data: bool


def _categorize(aqi: float) -> str:
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


@app.get("/cities")
def get_cities():
    """Return the list of cities available in the dataset, for the Streamlit selectbox."""
    if _aqi_data is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    return {"cities": sorted(_aqi_data["City"].unique().tolist())}


@app.get("/trend/{city}")
def get_trend(city: str):
    """Return historical Year/AQI pairs for a city, used to draw the trend chart."""
    entry = _get_or_train_model(city)
    return {"city": city, "trend": entry["trend"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predicts AQI for a given city and YEAR using HISTORICAL average pollutant
    levels for that city (not live data). This is intentional: Year only
    meaningfully influences the prediction when pollutant inputs also vary
    across years, the way they do historically. Live, current-moment pollutant
    data is shown separately in /current, where it belongs.
    """
    entry = _get_or_train_model(req.city)
    model = entry["model"]
    avg_values = entry["avg_values"]

    user_input = [[
        req.year,
        avg_values["PM2.5"],
        avg_values["PM10"],
        avg_values["NO2"],
        avg_values["CO"],
        avg_values["SO2"],
        avg_values["O3"],
    ]]
    prediction = float(model.predict(user_input)[0])

    return PredictResponse(
        city=req.city,
        year=req.year,
        predicted_aqi=round(prediction, 1),
        rmse=round(entry["rmse"], 2),
        cv_rmse=entry["cv_rmse"],
        model_name=entry["model_name"],
        category=_categorize(prediction),
        used_live_data=False,
    )


@app.get("/forecast/{city}")
def forecast(city: str, start_year: int, end_year: int):
    """
    Predicts AQI for EVERY year from start_year to end_year, using this
    city's HISTORICAL average pollutant levels (same inputs as /predict).
    Used to draw a multi-year forecast line, continuing from the historical trend.
    """
    if end_year < start_year:
        raise HTTPException(status_code=400, detail="end_year must be >= start_year")
    if end_year - start_year > 30:
        raise HTTPException(status_code=400, detail="Range too large (max 30 years)")

    entry = _get_or_train_model(city)
    model = entry["model"]

    avg_values = entry["avg_values"]
    pm25, pm10 = avg_values["PM2.5"], avg_values["PM10"]
    no2, co, so2, o3 = avg_values["NO2"], avg_values["CO"], avg_values["SO2"], avg_values["O3"]

    years = list(range(start_year, end_year + 1))
    inputs = [[yr, pm25, pm10, no2, co, so2, o3] for yr in years]
    predictions = model.predict(inputs)

    return {
        "city": city,
        "used_live_data": False,
        "forecast": [
            {"Year": yr, "AQI": round(float(p), 1)} for yr, p in zip(years, predictions)
        ],
    }


@app.get("/feature_importance/{city}")
def feature_importance(city: str):
    """
    Shows how much each input feature actually influences the model's
    predictions for this city. Useful for diagnosing why changing one
    input (like Year) might barely move the prediction.
    """
    entry = _get_or_train_model(city)
    model = entry["model"]
    importances = model.feature_importances_
    return {
        "city": city,
        "importances": {
            feat: round(float(imp), 4)
            for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1])
        },
    }


@app.get("/current/{city}")
def get_current_conditions(city: str):
    """
    Returns REAL, RIGHT-NOW air quality data for a city, pulled live from the
    internet (Open-Meteo). This is separate from /predict, which forecasts a
    FUTURE year using the trained ML model. This endpoint does no prediction
    at all -- it's a live measurement, not a model output.
    """
    lat, lon = _geocode_city(city)
    live_data = _fetch_live_air_quality(lat, lon)
    return {
        "city": city,
        "latitude": lat,
        "longitude": lon,
        "pm2_5": live_data["pm2_5"],
        "pm10": live_data["pm10"],
        "no2": live_data["nitrogen_dioxide"],
        "co": live_data["carbon_monoxide"],
        "so2": live_data["sulphur_dioxide"],
        "o3": live_data["ozone"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "cached_cities": list(_model_cache.keys())}