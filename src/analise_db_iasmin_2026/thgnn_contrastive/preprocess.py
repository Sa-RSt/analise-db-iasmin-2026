import os
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from analise_db_iasmin_2026.thgnn_contrastive.split import get_engine_split

# All 21 sensors
ALL_SENSORS = list(range(1, 22))
# Operating settings
OPERATING_SETTINGS = ["os1", "os2", "os3"]

# Sensor type mapping (0: Temp, 1: Press, 2: Speed, 3: Engine Params, 4: Coolant, 5, 6, 7: OpSettings)
_SENSOR_TYPES_RAW = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 1,
    6: 1,
    7: 1,
    10: 1,
    11: 1,
    8: 2,
    9: 2,
    13: 2,
    14: 2,
    18: 2,
    19: 2,
    12: 3,
    15: 3,
    17: 3,
    20: 3,
    21: 3,
    16: 4,
    "os1": 5,
    "os2": 6,
    "os3": 7,
}

# Feature list: Operating Settings + Sensors
# Order: os1, os2, os3, s1, s2, ..., s21
ACTIVE_FEATURES = OPERATING_SETTINGS + [f"s{i}" for i in ALL_SENSORS]

# Build SENSOR_TYPES with string keys for sensors
SENSOR_TYPES = {}
for k, v in _SENSOR_TYPES_RAW.items():
    if isinstance(k, int):
        SENSOR_TYPES[f"s{k}"] = v
    else:
        SENSOR_TYPES[k] = v

# Type IDs for the features
ACTIVE_SENSOR_TYPES = [SENSOR_TYPES[f] for f in ACTIVE_FEATURES]


def load_data(data_path, dataset_id):
    """
    Load CMAPSS data.
    """
    train_file = os.path.join(data_path, f"train_{dataset_id}.txt")
    test_file = os.path.join(data_path, f"test_{dataset_id}.txt")
    rul_file = os.path.join(data_path, f"RUL_{dataset_id}.txt")

    col_names = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]

    train_df = pd.read_csv(train_file, sep="\\s+", header=None, names=col_names)
    test_df = pd.read_csv(test_file, sep="\\s+", header=None, names=col_names)
    rul_df = pd.read_csv(rul_file, sep="\\s+", header=None, names=["rul"])

    return train_df, test_df, rul_df


def process_targets(df, r_max=125):
    """
    Piecewise linear RUL.
    """
    results = []
    for unit in df["unit"].unique():
        unit_df = df[df["unit"] == unit]
        max_cycle = unit_df["cycle"].max()
        rul = max_cycle - unit_df["cycle"]
        rul = np.minimum(rul, r_max)
        results.append(rul)
    return pd.concat(results)


def normalize_data(train_df, test_df, val_df=None, kmeans=None):
    """
    Standard (Z-score) normalization per operating condition for sensors.
    Operating settings are normalized globally.
    Sensors and operating settings scalers are fit strictly on train_df.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    if val_df is not None:
        val_df = val_df.copy()

    sensor_features = [f"s{i}" for i in ALL_SENSORS]
    for feature in sensor_features:
        train_df[feature] = np.float64(train_df[feature])
        test_df[feature] = np.float64(test_df[feature])
        if val_df is not None:
            val_df[feature] = np.float64(val_df[feature])

    # Check variance to identify single-regime datasets (e.g. FD001, FD003)
    if train_df["os1"].var() < 1.0:
        train_regimes = np.zeros(len(train_df), dtype=int)
        test_regimes = np.zeros(len(test_df), dtype=int)
        val_regimes = np.zeros(len(val_df), dtype=int) if val_df is not None else None
    else:
        # Multi-regime datasets (e.g. FD002, FD004)
        assert kmeans is not None
        train_regimes = kmeans.predict(train_df[OPERATING_SETTINGS].to_numpy())
        test_regimes = kmeans.predict(test_df[OPERATING_SETTINGS].to_numpy())
        val_regimes = (
            kmeans.predict(val_df[OPERATING_SETTINGS].to_numpy())
            if val_df is not None
            else None
        )

    # Scale sensors per regime
    for regime in np.unique(train_regimes):
        train_mask = train_regimes == regime
        test_mask = test_regimes == regime
        val_mask = val_regimes == regime if val_regimes is not None else None

        if train_mask.any():
            scaler = StandardScaler()
            # Fit and transform training data for this regime
            train_df.loc[train_mask, sensor_features] = scaler.fit_transform(
                train_df.loc[train_mask, sensor_features]
            )
            # Transform test data for this regime
            if test_mask.any():
                test_df.loc[test_mask, sensor_features] = scaler.transform(
                    test_df.loc[test_mask, sensor_features]
                )
            # Transform val data for this regime
            if val_mask is not None and np.any(val_mask):
                val_df.loc[val_mask, sensor_features] = scaler.transform(
                    val_df.loc[val_mask, sensor_features]
                )

    # Global scaling for operating settings
    scaler_os = StandardScaler()
    train_df[OPERATING_SETTINGS] = scaler_os.fit_transform(train_df[OPERATING_SETTINGS])
    test_df[OPERATING_SETTINGS] = scaler_os.transform(test_df[OPERATING_SETTINGS])
    if val_df is not None:
        val_df[OPERATING_SETTINGS] = scaler_os.transform(val_df[OPERATING_SETTINGS])

    # Fill NaNs for constant features (std=0 in a regime)
    train_df[ACTIVE_FEATURES] = train_df[ACTIVE_FEATURES].fillna(0)
    test_df[ACTIVE_FEATURES] = test_df[ACTIVE_FEATURES].fillna(0)
    if val_df is not None:
        val_df[ACTIVE_FEATURES] = val_df[ACTIVE_FEATURES].fillna(0)
        return train_df, val_df, test_df

    return train_df, test_df


def create_windows(df, window_size, step=1, last_only=False):
    """
    Create sliding windows.
    """
    X = []
    y = []

    for unit in df["unit"].unique():
        unit_df = df[df["unit"] == unit]
        if len(unit_df) < window_size:
            if last_only:
                data = unit_df[ACTIVE_FEATURES].values
                padding = np.zeros((window_size - len(data), len(ACTIVE_FEATURES)))
                X.append(np.vstack([padding, data]))
                y.append(unit_df["rul"].values[-1])
            continue

        data = unit_df[ACTIVE_FEATURES].values
        targets = unit_df["rul"].values

        if last_only:
            X.append(data[-window_size:])
            y.append(targets[-1])
        else:
            for i in range(0, len(data) - window_size + 1, step):
                X.append(data[i : i + window_size])
                y.append(targets[i + window_size - 1])

    return np.array(X), np.array(y)


def get_preprocessed_data(
    data_path,
    dataset_id,
    window_size=50,
    val_size=0.2,
    fold=None,
    n_splits=5,
    random_state=42,
    mode="kfold",
    loaded_data=None,
):
    """
    Full preprocessing pipeline with engine-level splitting and K-fold/LOEO support.
    """

    if loaded_data is not None:
        train_df_all, test_df, rul_df = loaded_data
        train_df_all = train_df_all.copy()
        test_df = test_df.copy()
        rul_df = rul_df.copy()
    else:
        train_df_all, test_df, rul_df = load_data(data_path, dataset_id)

    # Training targets
    train_df_all["rul"] = process_targets(train_df_all)

    # Test targets
    test_df["rul"] = 0
    for i, unit in enumerate(test_df["unit"].unique()):
        unit_df = test_df[test_df["unit"] == unit]
        max_cycle = unit_df["cycle"].max()
        actual_rul = rul_df.iloc[i].item()
        test_df.loc[test_df["unit"] == unit, "rul"] = np.minimum(
            actual_rul + (max_cycle - unit_df["cycle"]), 125
        )

    # Get unique units for engine-level splitting
    units = train_df_all["unit"].unique()

    train_units, val_units = get_engine_split(
        units,
        val_size=val_size,
        fold=fold,
        n_splits=n_splits,
        random_state=random_state,
        mode=mode,
    )

    train_df = train_df_all[train_df_all["unit"].isin(train_units)].copy()
    val_df = train_df_all[train_df_all["unit"].isin(val_units)].copy()

    # Load pre-computed K-Means if available for multi-regime datasets
    kmeans = None
    if fold is not None and train_df["os1"].var() >= 1.0:
        kmeans_path = os.path.join(data_path, f"kmeans_{dataset_id}_fold_{fold}.pkl")
        with open(kmeans_path, "rb") as f:
            kmeans = pickle.load(f)

    train_df, val_df, test_df = normalize_data(
        train_df, test_df, val_df=val_df, kmeans=kmeans
    )

    X_train, y_train = create_windows(train_df, window_size=window_size)
    X_val, y_val = create_windows(val_df, window_size=window_size)
    X_test, y_test = create_windows(test_df, window_size=window_size, last_only=True)

    return X_train, y_train, X_val, y_val, X_test, y_test
