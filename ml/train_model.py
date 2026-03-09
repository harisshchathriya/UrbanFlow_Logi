import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
ORDERS_CSV = BASE_DIR / "chennai_orders.csv"
VEHICLES_CSV = BASE_DIR / "chennai_vehicles.csv"
MODEL_PATH = BASE_DIR / "trip_cost_model.pkl"

PRIORITY_MAP = {"low": 1, "medium": 2, "high": 3}
COMMODITY_PROFILES = {
    "groceries": {"weight_per_unit": 2.2, "volume_per_unit": 0.035},
    "electronics": {"weight_per_unit": 1.4, "volume_per_unit": 0.02},
    "medicine": {"weight_per_unit": 0.5, "volume_per_unit": 0.01},
    "general": {"weight_per_unit": 1.8, "volume_per_unit": 0.028},
}
NUMERIC_FEATURES = [
    "quantity",
    "weight",
    "volume",
    "distance_km",
    "priority_num",
    "time_remaining_hr",
    "pickup_latitude",
    "pickup_longitude",
    "delivery_latitude",
    "delivery_longitude",
]
CATEGORICAL_FEATURES = [
    "commodity",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def haversine(lat1, lon1, lat2, lon2):
    earth_radius_km = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return earth_radius_km * c


def load_data():
    if not ORDERS_CSV.exists():
        raise FileNotFoundError(f"Orders dataset not found: {ORDERS_CSV}")
    if not VEHICLES_CSV.exists():
        raise FileNotFoundError(f"Vehicles dataset not found: {VEHICLES_CSV}")

    orders = pd.read_csv(ORDERS_CSV)
    vehicles = pd.read_csv(VEHICLES_CSV)
    if orders.empty:
        raise ValueError("Orders dataset is empty.")
    if vehicles.empty:
        raise ValueError("Vehicles dataset is empty.")
    return orders, vehicles


def get_commodity_profile(name):
    key = str(name or "general").strip().lower()
    return COMMODITY_PROFILES.get(key, COMMODITY_PROFILES["general"])


def enrich_orders_frame(orders):
    frame = orders.copy()

    frame["commodity"] = frame.get("commodity", "General").fillna("General").astype(str)
    frame["quantity"] = pd.to_numeric(frame.get("quantity", 1), errors="coerce").fillna(1.0).clip(lower=1.0)
    frame["priority_num"] = (
        frame.get("priority", "medium")
        .fillna("medium")
        .astype(str)
        .str.lower()
        .map(PRIORITY_MAP)
        .fillna(2.0)
        .astype(float)
    )

    frame["pickup_latitude"] = pd.to_numeric(frame.get("pickup_latitude", frame.get("pickup_lat")), errors="coerce")
    frame["pickup_longitude"] = pd.to_numeric(frame.get("pickup_longitude", frame.get("pickup_lng")), errors="coerce")
    frame["delivery_latitude"] = pd.to_numeric(frame.get("delivery_latitude", frame.get("drop_lat")), errors="coerce")
    frame["delivery_longitude"] = pd.to_numeric(frame.get("delivery_longitude", frame.get("drop_lng")), errors="coerce")

    distance_series = frame.get("distance_km", frame.get("distance"))
    frame["distance_km"] = pd.to_numeric(distance_series, errors="coerce")

    missing_distance = frame["distance_km"].isna()
    if missing_distance.any():
      frame.loc[missing_distance, "distance_km"] = frame.loc[missing_distance].apply(
          lambda row: haversine(
              row["pickup_latitude"],
              row["pickup_longitude"],
              row["delivery_latitude"],
              row["delivery_longitude"],
          ),
          axis=1,
      )

    created_at = pd.to_datetime(frame.get("created_at"), errors="coerce")
    deadline = pd.to_datetime(frame.get("delivery_deadline", frame.get("deadline")), errors="coerce")
    frame["time_remaining_hr"] = ((deadline - created_at).dt.total_seconds() / 3600.0).fillna(6.0).clip(lower=0.25)

    frame["weight"] = np.nan
    frame["volume"] = np.nan
    for idx, commodity in frame["commodity"].items():
        profile = get_commodity_profile(commodity)
        quantity = float(frame.at[idx, "quantity"])
        frame.at[idx, "weight"] = quantity * profile["weight_per_unit"]
        frame.at[idx, "volume"] = quantity * profile["volume_per_unit"]

    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(0.0)
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    frame["earnings"] = pd.to_numeric(frame.get("earnings"), errors="coerce")

    return frame


def train_and_save(orders, vehicles):
    enriched = enrich_orders_frame(orders)
    trainable = enriched.dropna(subset=["earnings"]).copy()
    if trainable.empty:
        raise ValueError("Orders dataset does not contain trainable earnings values.")

    feature_frame = trainable[FEATURE_COLUMNS].copy()
    target = trainable["earnings"].astype(float)

    x_train, x_test, y_train, y_test = train_test_split(
        feature_frame,
        target,
        test_size=0.2,
        random_state=42,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=14,
                    min_samples_leaf=2,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)
    test_predictions = model.predict(x_test)
    train_predictions = model.predict(x_train)

    summary = {
        "status": "ok",
        "orders_loaded": int(len(orders)),
        "vehicles_loaded": int(len(vehicles)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "feature_columns": FEATURE_COLUMNS,
        "mae": float(round(mean_absolute_error(y_test, test_predictions), 4)),
        "r2": float(round(r2_score(y_test, test_predictions), 4)),
        "train_r2": float(round(r2_score(y_train, train_predictions), 4)),
    }

    model_bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "priority_map": PRIORITY_MAP,
        "commodity_profiles": COMMODITY_PROFILES,
        "default_mileage": 12.0,
        "eta_speed_kmph": 28.0,
        "summary": summary,
    }
    joblib.dump(model_bundle, MODEL_PATH)
    return summary


def main():
    orders, vehicles = load_data()
    summary = train_and_save(orders, vehicles)
    summary["model_path"] = str(MODEL_PATH)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
