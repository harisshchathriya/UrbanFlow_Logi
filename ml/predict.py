import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "trip_cost_model.pkl"
VEHICLES_CSV = BASE_DIR / "chennai_vehicles.csv"
DEFAULT_PRIORITY_MAP = {"low": 1, "medium": 2, "high": 3}
DEFAULT_MILEAGE = 12.0
DEFAULT_FEATURE_COLUMNS = [
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
    "commodity",
]
DEFAULT_COMMODITY_PROFILES = {
    "groceries": {"weight_per_unit": 2.2, "volume_per_unit": 0.035},
    "electronics": {"weight_per_unit": 1.4, "volume_per_unit": 0.02},
    "medicine": {"weight_per_unit": 0.5, "volume_per_unit": 0.01},
    "general": {"weight_per_unit": 1.8, "volume_per_unit": 0.028},
}
DEFAULT_VEHICLE_ROWS = [
    {
        "vehicle_id": "DEFAULT_VEHICLE",
        "max_weight_capacity": 1000.0,
        "max_volume_capacity": 10.0,
        "mileage": DEFAULT_MILEAGE,
        "current_lat": 13.0827,
        "current_long": 80.2707,
    }
]
REQUIRED_VEHICLE_COLUMNS = [
    "vehicle_id",
    "max_weight_capacity",
    "max_volume_capacity",
    "mileage",
    "current_lat",
    "current_long",
]


def haversine(lat1, lon1, lat2, lon2):
    earth_radius_km = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(earth_radius_km * c)


def parse_json_text(raw_text, source_name):
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {source_name}: {exc.msg} (line {exc.lineno}, column {exc.colno})."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(f"JSON from {source_name} must be an object.")
    return payload


def parse_payload():
    args = sys.argv[1:]
    force_stdin = False
    file_path = None
    arg_parts = []
    index = 0

    while index < len(args):
        token = args[index]
        if token == "--stdin":
            force_stdin = True
        elif token == "--file":
            if index + 1 >= len(args):
                raise ValueError("Missing file path after --file.")
            file_path = args[index + 1]
            index += 1
        else:
            arg_parts.append(token)
        index += 1

    if file_path:
        input_path = Path(file_path)
        if not input_path.is_absolute():
            input_path = (Path.cwd() / input_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"JSON input file not found: {input_path}")
        return parse_json_text(input_path.read_text(encoding="utf-8"), f"file {input_path}")

    if arg_parts:
        raw_arg = " ".join(arg_parts).strip()
        try:
            return parse_json_text(raw_arg, "command-line argument")
        except ValueError as arg_error:
            if not sys.stdin.isatty():
                raw_stdin = sys.stdin.read().strip()
                if raw_stdin:
                    return parse_json_text(raw_stdin, "stdin")
            raise ValueError(
                f"{arg_error} Use --stdin with piped JSON or --file <path>."
            ) from arg_error

    if force_stdin:
        if sys.stdin.isatty():
            raise ValueError("No JSON received on stdin.")
        raw_stdin = sys.stdin.read().strip()
        if not raw_stdin:
            raise ValueError("No JSON received on stdin.")
        return parse_json_text(raw_stdin, "stdin")

    if not sys.stdin.isatty():
        raw_stdin = sys.stdin.read().strip()
        if raw_stdin:
            return parse_json_text(raw_stdin, "stdin")

    raise ValueError("Missing JSON input. Pass JSON arg, use --stdin with a pipe, or use --file <path>.")


def load_model_bundle():
    loaded = joblib.load(MODEL_PATH)
    if isinstance(loaded, dict) and "model" in loaded:
        return loaded
    if hasattr(loaded, "predict"):
        return {
            "model": loaded,
            "feature_columns": DEFAULT_FEATURE_COLUMNS,
            "priority_map": DEFAULT_PRIORITY_MAP,
            "commodity_profiles": DEFAULT_COMMODITY_PROFILES,
            "default_mileage": DEFAULT_MILEAGE,
            "eta_speed_kmph": 28.0,
        }
    raise ValueError("Unsupported model format in trip_cost_model.pkl")


def get_commodity_profile(name, commodity_profiles):
    key = str(name or "general").strip().lower()
    return commodity_profiles.get(key, commodity_profiles["general"])


def get_series(frame, primary, fallback=None, default_value=None):
    if primary in frame.columns:
        return frame[primary]
    if fallback and fallback in frame.columns:
        return frame[fallback]
    return pd.Series([default_value] * len(frame), index=frame.index)


def _parse_orders(payload):
    if "orders" in payload:
        orders_raw = payload["orders"]
        if not isinstance(orders_raw, list) or not orders_raw:
            raise ValueError("'orders' must be a non-empty array.")
        for item in orders_raw:
            if not isinstance(item, dict):
                raise ValueError("Every entry in 'orders' must be a JSON object.")
        return orders_raw, True
    return [payload], False


def _load_vehicle_frame(payload, default_mileage):
    vehicles_raw = payload.get("vehicles")

    if vehicles_raw is None:
        if VEHICLES_CSV.exists():
            vehicles = pd.read_csv(VEHICLES_CSV)
        else:
            vehicles = pd.DataFrame(DEFAULT_VEHICLE_ROWS)
    else:
        if not isinstance(vehicles_raw, list) or not vehicles_raw:
            raise ValueError("'vehicles' must be a non-empty array when provided.")
        for item in vehicles_raw:
            if not isinstance(item, dict):
                raise ValueError("Every entry in 'vehicles' must be a JSON object.")
        vehicles = pd.DataFrame(vehicles_raw)

    for column in REQUIRED_VEHICLE_COLUMNS:
        if column not in vehicles.columns:
            if column == "vehicle_id":
                vehicles[column] = ""
            elif column == "mileage":
                vehicles[column] = float(default_mileage)
            else:
                vehicles[column] = np.nan

    vehicles = vehicles[REQUIRED_VEHICLE_COLUMNS].copy()
    vehicles["vehicle_id"] = vehicles["vehicle_id"].fillna("").astype(str).str.strip()
    missing_ids = vehicles["vehicle_id"] == ""
    if missing_ids.any():
        vehicles.loc[missing_ids, "vehicle_id"] = [
            f"VEHICLE_{idx + 1}" for idx in range(int(missing_ids.sum()))
        ]

    for column in ["max_weight_capacity", "max_volume_capacity", "mileage", "current_lat", "current_long"]:
        vehicles[column] = pd.to_numeric(vehicles[column], errors="coerce")
    vehicles["mileage"] = vehicles["mileage"].replace(0, np.nan).fillna(float(default_mileage)).clip(lower=0.1)

    if vehicles.empty:
        vehicles = pd.DataFrame(DEFAULT_VEHICLE_ROWS)
    return vehicles.reset_index(drop=True)


def _build_orders_frame(orders_raw, priority_map, commodity_profiles):
    orders = pd.DataFrame(orders_raw).copy()
    orders["order_index"] = range(len(orders))

    if "order_id" in orders.columns:
        orders["order_ref"] = orders["order_id"].fillna("").astype(str)
    else:
        orders["order_ref"] = [f"order_{idx + 1}" for idx in range(len(orders))]

    orders["commodity"] = get_series(orders, "commodity", default_value="General").fillna("General").astype(str)
    orders["quantity"] = pd.to_numeric(get_series(orders, "quantity", default_value=1), errors="coerce").fillna(1.0).clip(lower=1.0)

    for column, fallback_column in [
        ("pickup_latitude", "pickup_lat"),
        ("pickup_longitude", "pickup_lng"),
        ("delivery_latitude", "drop_lat"),
        ("delivery_longitude", "drop_lng"),
    ]:
        source = get_series(orders, column, fallback_column)
        orders[column] = pd.to_numeric(source, errors="coerce")

    distance_source = get_series(orders, "distance_km", "distance")
    orders["distance_km"] = pd.to_numeric(distance_source, errors="coerce")
    missing_distance = orders["distance_km"].isna()
    if missing_distance.any():
        missing_coords = orders.loc[missing_distance, ["pickup_latitude", "pickup_longitude", "delivery_latitude", "delivery_longitude"]].isna().any(axis=1)
        if missing_coords.any():
            raise ValueError(
                "Each order requires 'distance_km' or all pickup/delivery latitude and longitude values."
            )
        orders.loc[missing_distance, "distance_km"] = orders.loc[missing_distance].apply(
            lambda row: haversine(
                row["pickup_latitude"],
                row["pickup_longitude"],
                row["delivery_latitude"],
                row["delivery_longitude"],
            ),
            axis=1,
        )

    if "priority_num" in orders.columns:
        orders["priority_num"] = pd.to_numeric(orders["priority_num"], errors="coerce")
    else:
        orders["priority_num"] = np.nan
    priority_series = get_series(orders, "priority", default_value="medium").fillna("medium").astype(str).str.lower()
    orders["priority_num"] = orders["priority_num"].fillna(priority_series.map(priority_map).fillna(2)).astype(float)

    created_at = pd.to_datetime(get_series(orders, "created_at"), errors="coerce")
    delivery_deadline = pd.to_datetime(get_series(orders, "delivery_deadline", "deadline"), errors="coerce")
    inferred_time = (delivery_deadline - created_at).dt.total_seconds() / 3600.0
    explicit_time = pd.to_numeric(get_series(orders, "time_remaining_hr"), errors="coerce")
    orders["time_remaining_hr"] = explicit_time.fillna(inferred_time).fillna(6.0).clip(lower=0.25)

    explicit_weight = pd.to_numeric(get_series(orders, "weight"), errors="coerce")
    explicit_volume = pd.to_numeric(get_series(orders, "volume"), errors="coerce")
    orders["weight"] = explicit_weight
    orders["volume"] = explicit_volume

    for idx, commodity in orders["commodity"].items():
        profile = get_commodity_profile(commodity, commodity_profiles)
        quantity = float(orders.at[idx, "quantity"])
        if not np.isfinite(orders.at[idx, "weight"]):
            orders.at[idx, "weight"] = quantity * profile["weight_per_unit"]
        if not np.isfinite(orders.at[idx, "volume"]):
            orders.at[idx, "volume"] = quantity * profile["volume_per_unit"]

    orders["weight"] = pd.to_numeric(orders["weight"], errors="coerce").fillna(0.0)
    orders["volume"] = pd.to_numeric(orders["volume"], errors="coerce").fillna(0.0)
    return orders


def _choose_cluster_count(orders, vehicles):
    avg_weight_capacity = max(float(vehicles["max_weight_capacity"].replace(0, np.nan).mean()), 1.0)
    avg_volume_capacity = max(float(vehicles["max_volume_capacity"].replace(0, np.nan).mean()), 0.1)
    total_weight = float(orders["weight"].sum())
    total_volume = float(orders["volume"].sum())
    by_weight = int(np.ceil(total_weight / avg_weight_capacity))
    by_volume = int(np.ceil(total_volume / avg_volume_capacity))
    by_stops = int(np.ceil(len(orders) / 4.0))
    return max(1, min(len(orders), len(vehicles), max(by_weight, by_volume, by_stops, 1)))


def _pick_best_vehicle(cluster_orders, vehicles):
    total_weight = float(cluster_orders["weight"].sum())
    total_volume = float(cluster_orders["volume"].sum())
    centroid_lat = float(cluster_orders["pickup_latitude"].mean())
    centroid_lng = float(cluster_orders["pickup_longitude"].mean())

    feasible = vehicles[
        (vehicles["max_weight_capacity"] >= total_weight)
        & (vehicles["max_volume_capacity"] >= total_volume)
    ].copy()
    candidate_pool = feasible if not feasible.empty else vehicles.copy()

    candidate_pool["distance_to_cluster"] = candidate_pool.apply(
        lambda row: haversine(
            row["current_lat"] if np.isfinite(row["current_lat"]) else centroid_lat,
            row["current_long"] if np.isfinite(row["current_long"]) else centroid_lng,
            centroid_lat,
            centroid_lng,
        ),
        axis=1,
    )
    candidate_pool = candidate_pool.sort_values(
        by=["distance_to_cluster", "max_weight_capacity", "max_volume_capacity"],
        ascending=[True, True, True],
    )
    selected = candidate_pool.iloc[0]
    capacity_exceeded = feasible.empty
    return selected, capacity_exceeded


def _apply_load_consolidation(orders, vehicles, default_mileage):
    orders = orders.copy()
    vehicles = vehicles.copy()
    if len(orders) > 1 and not orders[["pickup_latitude", "pickup_longitude", "delivery_latitude", "delivery_longitude"]].isna().any(axis=1).any():
        cluster_count = _choose_cluster_count(orders, vehicles)
        features = np.column_stack([
            orders["pickup_latitude"].to_numpy(),
            orders["pickup_longitude"].to_numpy(),
            orders["delivery_latitude"].to_numpy(),
            orders["delivery_longitude"].to_numpy(),
            (orders["priority_num"].to_numpy() / 3.0),
            np.clip(orders["time_remaining_hr"].to_numpy(), 0.25, 24.0) / 24.0,
        ])
        kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        orders["cluster_id"] = kmeans.fit_predict(features)
    else:
        orders["cluster_id"] = 0

    orders["assigned_vehicle"] = ""
    orders["capacity_exceeded"] = False
    orders["cluster_total_weight"] = 0.0
    orders["cluster_total_volume"] = 0.0
    orders["cluster_size"] = 0
    orders["vehicle_mileage"] = float(default_mileage)

    for cluster_id in sorted(orders["cluster_id"].unique()):
        cluster_mask = orders["cluster_id"] == cluster_id
        cluster_orders = orders.loc[cluster_mask]
        vehicle, exceeds_capacity = _pick_best_vehicle(cluster_orders, vehicles)
        total_weight = float(cluster_orders["weight"].sum())
        total_volume = float(cluster_orders["volume"].sum())
        orders.loc[cluster_mask, "assigned_vehicle"] = str(vehicle["vehicle_id"])
        orders.loc[cluster_mask, "capacity_exceeded"] = bool(exceeds_capacity)
        orders.loc[cluster_mask, "cluster_total_weight"] = total_weight
        orders.loc[cluster_mask, "cluster_total_volume"] = total_volume
        orders.loc[cluster_mask, "cluster_size"] = int(cluster_orders.shape[0])
        orders.loc[cluster_mask, "vehicle_mileage"] = float(vehicle["mileage"])

    orders["mileage_for_calc"] = orders["vehicle_mileage"].fillna(float(default_mileage)).clip(lower=0.1)
    orders["fuel_used"] = orders["distance_km"] / orders["mileage_for_calc"]
    orders["fuel_used"] = orders["fuel_used"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    dispatch_order = orders.sort_values(
        by=["cluster_id", "priority_num", "time_remaining_hr"],
        ascending=[True, False, True],
    ).copy()
    dispatch_order["dispatch_rank"] = dispatch_order.groupby("cluster_id").cumcount() + 1
    orders = orders.merge(dispatch_order[["order_index", "dispatch_rank"]], on="order_index", how="left")
    orders["dispatch_rank"] = orders["dispatch_rank"].fillna(1).astype(int)
    return orders


def _predict_with_consolidation(payload, model_bundle):
    orders_raw, is_batch = _parse_orders(payload)
    priority_map = model_bundle.get("priority_map", DEFAULT_PRIORITY_MAP)
    commodity_profiles = model_bundle.get("commodity_profiles", DEFAULT_COMMODITY_PROFILES)
    default_mileage = float(model_bundle.get("default_mileage", DEFAULT_MILEAGE))
    eta_speed_kmph = float(model_bundle.get("eta_speed_kmph", 28.0))
    eta_speed_kmph = eta_speed_kmph if eta_speed_kmph > 0 else 28.0

    vehicles = _load_vehicle_frame(payload, default_mileage)
    orders = _build_orders_frame(orders_raw, priority_map, commodity_profiles)
    orders = _apply_load_consolidation(orders, vehicles, default_mileage)

    feature_columns = model_bundle.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
    model = model_bundle["model"]
    predicted_costs = model.predict(orders[feature_columns].copy())
    orders["predicted_cost"] = np.round(predicted_costs.astype(float), 2)

    orders["eta_hr"] = orders["distance_km"] / eta_speed_kmph
    orders["deadline_risk"] = orders["eta_hr"] > orders["time_remaining_hr"]

    consolidation_discount = np.where(orders["cluster_size"] > 1, np.minimum((orders["cluster_size"] - 1) * 0.04, 0.18), 0.0)
    deadline_penalty = np.where(orders["deadline_risk"], 0.12, 0.0)
    capacity_penalty = np.where(orders["capacity_exceeded"], 0.15, 0.0)
    adjusted_multiplier = 1.0 - consolidation_discount + deadline_penalty + capacity_penalty
    orders["predicted_cost"] = np.round(orders["predicted_cost"] * adjusted_multiplier, 2)
    orders["predicted_cost"] = orders["predicted_cost"].clip(lower=40.0)

    orders = orders.sort_values("order_index").reset_index(drop=True)

    prediction_rows = []
    for _, row in orders.iterrows():
        prediction_rows.append(
            {
                "order_index": int(row["order_index"]),
                "order_ref": str(row["order_ref"]),
                "predicted_cost": float(row["predicted_cost"]),
                "consolidation": {
                    "cluster_id": int(row["cluster_id"]),
                    "cluster_size": int(row["cluster_size"]),
                    "cluster_total_weight": round(float(row["cluster_total_weight"]), 4),
                    "cluster_total_volume": round(float(row["cluster_total_volume"]), 4),
                    "assigned_vehicle": str(row["assigned_vehicle"]),
                    "capacity_exceeded": bool(row["capacity_exceeded"]),
                    "dispatch_rank": int(row["dispatch_rank"]),
                    "deadline_risk": bool(row["deadline_risk"]),
                },
            }
        )

    summary = {
        "orders_count": int(len(prediction_rows)),
        "clusters_count": int(orders["cluster_id"].nunique()),
        "capacity_exceeded_clusters": int(orders.groupby("cluster_id")["capacity_exceeded"].any().sum()),
        "deadline_risk_orders": int(orders["deadline_risk"].sum()),
        "predicted_cost_total": round(float(orders["predicted_cost"].sum()), 2),
    }

    if not is_batch and len(prediction_rows) == 1:
        first = prediction_rows[0]
        return {
            "predicted_cost": first["predicted_cost"],
            "consolidation": first["consolidation"],
            "summary": summary,
        }
    return {"predictions": prediction_rows, "summary": summary}


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run train_model.py first.")
    model_bundle = load_model_bundle()
    payload = parse_payload()
    result = _predict_with_consolidation(payload, model_bundle)
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)
