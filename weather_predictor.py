"""
weather_predictor.py

Predicts next-day maximum temperature using real historical weather data
pulled live from the free Open-Meteo archive API — no account, no API key,
no downloaded dataset required.

Usage:
    pip install requests pandas scikit-learn matplotlib
    python weather_predictor.py
"""

import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# ---- 1. Settings: change these for any city on Earth ----
LATITUDE = 12.9716      # Bangalore
LONGITUDE = 77.5946
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"


def fetch_weather(lat, lon, start, end):
    """Pull daily historical weather from Open-Meteo (free, keyless, no signup)."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    daily = response.json()["daily"]
    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={
        "time": "date",
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "precipitation_sum": "rain",
    })
    df = df.sort_values("date").reset_index(drop=True)

    # Save a plain CSV snapshot -- this is your "dataset file" to
    # point to, cite, or hand in alongside the code,
    # dataset is sourced from a live API.
    df.to_csv("weather_dataset.csv", index=False)
    print(f"Saved {len(df)} rows to weather_dataset.csv")
    return df


def build_features(df):
    """Turn raw daily weather into a supervised-learning table: use today's
    weather to predict TOMORROW's max temperature."""
    df = df.copy()
    df["day_of_year"] = df["date"].dt.dayofyear
    df["target_next_max"] = df["temp_max"].shift(-1)     # tomorrow's max temp
    df["temp_max_lag1"] = df["temp_max"]
    df["temp_min_lag1"] = df["temp_min"]
    df["rain_lag1"] = df["rain"]
    df["temp_max_lag2"] = df["temp_max"].shift(1)          # day before that
    df = df.dropna().reset_index(drop=True)
    return df


def train_and_evaluate(df):
    features = ["temp_max_lag1", "temp_min_lag1", "rain_lag1", "temp_max_lag2", "day_of_year"]

    # Chronological split -- never shuffle time-series data, or the model
    # ends up "predicting the past" using information from the future.
    split = int(len(df) * 0.85)
    train, test = df.iloc[:split], df.iloc[split:]

    X_train, y_train = train[features], train["target_next_max"]
    X_test, y_test = test[features], test["target_next_max"]

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Naive baseline: "tomorrow will be the same as today."
    baseline_preds = X_test["temp_max_lag1"]

    model_mae = mean_absolute_error(y_test, preds)
    baseline_mae = mean_absolute_error(y_test, baseline_preds)

    print(f"Model MAE:    {model_mae:.2f} degC")
    print(f"Baseline MAE: {baseline_mae:.2f} degC  (predicting 'same as today')")
    print(f"Improvement over baseline: {baseline_mae - model_mae:.2f} degC")

    plt.figure(figsize=(11, 5))
    plt.plot(test["date"], y_test.values, label="Actual max temp", linewidth=1.5)
    plt.plot(test["date"], preds, label="Predicted max temp", linewidth=1.5)
    plt.title("Next-day max temperature: actual vs predicted")
    plt.xlabel("Date")
    plt.ylabel("Temperature (degC)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("prediction_results.png", dpi=150)
    print("Saved chart to prediction_results.png")

    return model, features


if __name__ == "__main__":
    print("Fetching historical weather data from Open-Meteo...")
    raw = fetch_weather(LATITUDE, LONGITUDE, START_DATE, END_DATE)
    print(f"Got {len(raw)} days of data.")

    data = build_features(raw)
    model, feature_cols = train_and_evaluate(data)

    # Predict tomorrow's max temp from the most recent day on record
    latest = data.iloc[[-1]][feature_cols]
    tomorrow_pred = model.predict(latest)[0]
    print(f"\nPredicted next-day max temperature: {tomorrow_pred:.1f} degC")
