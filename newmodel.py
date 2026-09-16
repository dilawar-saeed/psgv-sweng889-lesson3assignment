# import io
# import zipfile
# from pathlib import Path

# import joblib
# import matplotlib.pyplot as plt
# import pandas as pd
# import requests
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# # Save generated files in the models folder beside this script.
# MODELS_DIR = Path(__file__).resolve().parent / "models"
# FEATURE_COLUMNS_FILE = MODELS_DIR / "feature_columns.pkl"
# MODEL_FILE = MODELS_DIR / "seoul_bike_hgb_model.pkl"


# # Download and load the dataset into a DataFrame.
# def load_data():
#     url = "https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip"
#     response = requests.get(url)
#     response.raise_for_status()

#     with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
#         with archive.open("SeoulBikeData.csv") as csv_file:
#             return pd.read_csv(csv_file, encoding="cp949")


# # Remove duplicate rows so they do not bias analysis or model training.
# def remove_duplicates(df):
#     return df.drop_duplicates().copy()


# # Create date and time features that can explain seasonal and daily demand patterns.
# def engineer_date_and_time_features(df):
#     df = df.copy()
#     date = pd.to_datetime(df["Date"], dayfirst=True)

#     df["Year"] = date.dt.year
#     df["Month"] = date.dt.month
#     df["DayOfWeek"] = date.dt.dayofweek
#     df["IsWeekend"] = (date.dt.dayofweek >= 5).astype(int)
#     df["TimeOfDay"] = pd.cut(
#         df["Hour"],
#         bins=[-1, 5, 11, 16, 20, 23],
#         labels=["Night", "Morning", "Afternoon", "Evening", "LateNight"],
#     )
#     return df.drop(columns="Date")


# # Fill numeric gaps with medians and categorical gaps with the most common value.
# def fill_missing_values(df):
#     df = df.copy()
#     numeric_columns = df.select_dtypes(include="number").columns
#     categorical_columns = df.select_dtypes(exclude="number").columns

#     df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
#     for column in categorical_columns:
#         if df[column].isna().any():
#             df[column] = df[column].fillna(df[column].mode()[0])
#     return df


# # One-hot encode categorical columns so the model receives numeric inputs.
# def encode_categorical_columns(df):
#     return pd.get_dummies(df, drop_first=True, dtype=int)


# # Clean the data, create features, and separate predictors from the target.
# def prepare_features_and_target(df):
#     target = "Rented Bike Count"
#     df = remove_duplicates(df)
#     df = engineer_date_and_time_features(df)
#     df = fill_missing_values(df)

#     y = df.pop(target)
#     X = encode_categorical_columns(df)
#     return X, y


# # Split chronologically so the test set represents future demand.
# def split_data(X, y):
#     split_index = int(len(X) * 0.8)
#     return (
#         X.iloc[:split_index], X.iloc[split_index:],
#         y.iloc[:split_index], y.iloc[split_index:],
#     )


# # Train a simple model that can learn nonlinear weather and time relationships.
# def train_model(X_train, y_train):
#     model = RandomForestRegressor(
#         n_estimators=100,
#         random_state=42,
#         n_jobs=-1,
#     )
#     model.fit(X_train, y_train)
#     return model


# # Save the model and feature columns for later predictions in the Streamlit app.
# def save_model_files(model, feature_columns):
#     MODELS_DIR.mkdir(exist_ok=True)
#     joblib.dump(feature_columns, FEATURE_COLUMNS_FILE)
#     joblib.dump(model, MODEL_FILE)
#     print(f"Saved feature columns to: {FEATURE_COLUMNS_FILE}")
#     print(f"Saved model to: {MODEL_FILE}")


# # Calculate errors and goodness of fit on unseen test data.
# def evaluate_model(model, X_test, y_test):
#     predictions = model.predict(X_test)
#     mae = mean_absolute_error(y_test, predictions)
#     rmse = mean_squared_error(y_test, predictions) ** 0.5
#     r_squared = r2_score(y_test, predictions)

#     print(f"MAE: {mae:.2f} bikes")
#     print(f"RMSE: {rmse:.2f} bikes")
#     print(f"R-squared: {r_squared:.3f}")
#     return predictions


# # Compare actual demand with predictions to reveal overall model fit.
# def plot_predictions(y_test, predictions):
#     plt.figure(figsize=(8, 5))
#     plt.scatter(y_test, predictions, alpha=0.35)
#     line_min = min(y_test.min(), predictions.min())
#     line_max = max(y_test.max(), predictions.max())
#     plt.plot([line_min, line_max], [line_min, line_max], "r--", label="Perfect predictions")
#     plt.xlabel("Actual rented bike count")
#     plt.ylabel("Predicted rented bike count")
#     plt.title("Actual vs. Predicted Bike Demand")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     df = load_data()
#     X, y = prepare_features_and_target(df)
#     X_train, X_test, y_train, y_test = split_data(X, y)
#     model = train_model(X_train, y_train)

#     # Save exactly two files in the sibling models folder.
#     save_model_files(model, X_train.columns.tolist())

#     predictions = evaluate_model(model, X_test, y_test)
#     plot_predictions(y_test, predictions)

import io
import zipfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODELS_DIR = Path(__file__).resolve().parent / "models"
FEATURE_COLUMNS_FILE = MODELS_DIR / "feature_columns.pkl"
MODEL_FILE = MODELS_DIR / "seoul_bike_hgb_model.pkl"

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/560/"
    "seoul+bike+sharing+demand.zip"
)


def load_data():
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        with archive.open("SeoulBikeData.csv") as csv_file:
            return pd.read_csv(csv_file, encoding="cp949")


def remove_duplicates(df):
    return df.drop_duplicates().copy()


def engineer_date_and_time_features(df):
    result = remove_duplicates(df)
    parsed_date = pd.to_datetime(result.pop("Date"), dayfirst=True, errors="raise")

    # Year is intentionally excluded so predictions work for current/future years.
    result["Month"] = parsed_date.dt.month
    result["DayOfWeek"] = parsed_date.dt.dayofweek
    result["IsWeekend"] = (result["DayOfWeek"] >= 5).astype(int)
    result["TimeOfDay"] = pd.cut(
        result["Hour"],
        bins=[-1, 5, 10, 15, 20, 23],
        labels=["Night", "Morning", "Midday", "Evening", "LateNight"],
    )
    return result


def fill_missing_values(df):
    result = df.copy()
    numeric = result.select_dtypes(include="number").columns
    categorical = result.select_dtypes(exclude="number").columns
    result[numeric] = result[numeric].fillna(result[numeric].median())
    for column in categorical:
        if result[column].isna().any():
            result[column] = result[column].fillna(result[column].mode()[0])
    return result


def encode_categorical_columns(df):
    return pd.get_dummies(df, drop_first=True, dtype=int)


def prepare_features_and_target(df):
    result = engineer_date_and_time_features(df)
    result = fill_missing_values(result)
    target = "Rented Bike Count"
    y = result.pop(target).astype(float)
    return encode_categorical_columns(result), y


def split_data(X, y):
    split_index = int(len(X) * 0.8)
    return X.iloc[:split_index], X.iloc[split_index:], y.iloc[:split_index], y.iloc[split_index:]


def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def save_model_files(model, feature_columns):
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(feature_columns, FEATURE_COLUMNS_FILE)
    joblib.dump(model, MODEL_FILE)
    print(f"Saved feature columns to: {FEATURE_COLUMNS_FILE}")
    print(f"Saved model to: {MODEL_FILE}")


def evaluate_model(model, X_test, y_test):
    predictions = np.clip(model.predict(X_test), 0, None)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r_squared = r2_score(y_test, predictions)
    print(f"MAE: {mae:.2f} bikes")
    print(f"RMSE: {rmse:.2f} bikes")
    print(f"R-squared: {r_squared:.3f}")
    return predictions


def plot_predictions(y_test, predictions):
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, predictions, alpha=0.35)
    line_min = min(y_test.min(), predictions.min())
    line_max = max(y_test.max(), predictions.max())
    plt.plot([line_min, line_max], [line_min, line_max], "r--", label="Perfect predictions")
    plt.xlabel("Actual rented bike count")
    plt.ylabel("Predicted rented bike count")
    plt.title("Actual vs. Predicted Bike Demand")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = load_data()
    X, y = prepare_features_and_target(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    predictions = evaluate_model(model, X_test, y_test)

    # Refit on all historical rows before saving the production artifact.
    final_model = train_model(X, y)
    save_model_files(final_model, X.columns.tolist())
    plot_predictions(y_test, predictions)
