from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "seoul_bike_hgb_model.pkl"
SCHEMA_PATH = ROOT / "models" / "feature_columns.pkl"

SEASONS = {"Spring", "Summer", "Autumn", "Winter"}
HOLIDAYS = {"Holiday", "No Holiday"}
DAYS_OF_WEEK = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}

REQUIRED_INPUTS = {
    "Month",
    "DayOfWeek",
    "Hour",
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)",
    "Seasons",
    "Holiday",
}


def validate_inputs(values):
    errors = []

    for name in REQUIRED_INPUTS:
        if values.get(name) in (None, ""):
            errors.append(f"{name} is required")

    if values.get("Month") not in (None, ""):
        try:
            month = int(values["Month"])
            if not 1 <= month <= 12:
                errors.append("Month must be between 1 and 12")
        except (TypeError, ValueError):
            errors.append("Month must be an integer")

    if values.get("DayOfWeek") not in DAYS_OF_WEEK:
        errors.append("Invalid day of week")

    ranges = {
        "Hour": (0, 23),
        "Temperature(°C)": (-30, 45),
        "Humidity(%)": (0, 100),
        "Wind speed (m/s)": (0, 50),
        "Visibility (10m)": (0, 3000),
        "Dew point temperature(°C)": (-40, 40),
        "Solar Radiation (MJ/m2)": (0, 10),
        "Rainfall(mm)": (0, 1000),
        "Snowfall (cm)": (0, 100),
    }

    for name, (low, high) in ranges.items():
        value = values.get(name)
        if value in (None, ""):
            continue
        try:
            number = float(value)
            if not np.isfinite(number) or not low <= number <= high:
                errors.append(f"{name} must be between {low} and {high}")
        except (TypeError, ValueError):
            errors.append(f"{name} must be numeric")

    if values.get("Seasons") not in SEASONS:
        errors.append("Invalid season")
    if values.get("Holiday") not in HOLIDAYS:
        errors.append("Invalid holiday value")

    return errors


def _engineer_features(values):
    row = {
        "Month": int(values["Month"]),
        "DayOfWeek": DAYS_OF_WEEK[values["DayOfWeek"]],
        "Hour": int(values["Hour"]),
        "Temperature(°C)": float(values["Temperature(°C)"]),
        "Humidity(%)": float(values["Humidity(%)"]),
        "Wind speed (m/s)": float(values["Wind speed (m/s)"]),
        "Visibility (10m)": float(values["Visibility (10m)"]),
        "Dew point temperature(°C)": float(values["Dew point temperature(°C)"]),
        "Solar Radiation (MJ/m2)": float(values["Solar Radiation (MJ/m2)"]),
        "Rainfall(mm)": float(values["Rainfall(mm)"]),
        "Snowfall (cm)": float(values["Snowfall (cm)"]),
        "Seasons": values["Seasons"],
        "Holiday": values["Holiday"],
    }

    frame = pd.DataFrame([row])
    frame["IsWeekend"] = frame["DayOfWeek"].isin([5, 6]).astype(int)
    frame["TimeOfDay"] = pd.cut(
        frame["Hour"],
        bins=[-1, 5, 10, 15, 20, 23],
        labels=["Night", "Morning", "Midday", "Evening", "LateNight"],
    )
    is_weekday = frame["IsWeekend"] == 0
    morning_rush = frame["Hour"].between(7, 9)
    evening_rush = frame["Hour"].between(17, 19)
    frame["IsRushHour"] = (
        is_weekday & (morning_rush | evening_rush)
    ).astype(int)

    return pd.get_dummies(
        frame,
        columns=["Seasons", "Holiday", "TimeOfDay"],
        drop_first=True,
        dtype=int,
    )


def predict_demand(values):
    errors = validate_inputs(values)
    if errors:
        raise ValueError("; ".join(errors))

    if not MODEL_PATH.exists() or not SCHEMA_PATH.exists():
        raise FileNotFoundError("Model artifacts are missing from models/.")

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(SCHEMA_PATH)
    features = _engineer_features(values).reindex(
        columns=feature_columns,
        fill_value=0,
    )
    prediction = model.predict(features)[0]
    return int(max(0, np.rint(prediction)))
