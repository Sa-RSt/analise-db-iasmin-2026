import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from analise_db_iasmin_2026.thgnn_contrastive.split import get_engine_split

# Dropped constant/uninformative sensors: 1, 5, 6, 10, 16, 18, 19
# Active 14 sensors
ACTIVE_SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
ALL_SENSORS = ACTIVE_SENSORS
# Operating settings
OPERATING_SETTINGS = ["os1", "os2", "os3"]

# Sensor type mapping:
# Type 0: Operating Settings (os1, os2, os3)
# Type 1: Temperature (s2, s3, s4)
# Type 2: Pressure (s7, s11)
# Type 3: Speed (s8, s9, s13, s14)
# Type 4: Engine Params / Flow / Bleed (s12, s15, s17, s20, s21)
_SENSOR_TYPES_RAW = {
    "os1": 0,
    "os2": 5,
    "os3": 6,
    2: 1,
    3: 1,
    4: 1,
    7: 2,
    11: 2,
    8: 3,
    9: 3,
    13: 3,
    14: 3,
    12: 4,
    15: 4,
    17: 4,
    20: 4,
    21: 4,
}

# Feature list: Operating Settings + Sensors
# Order: os1, os2, os3, s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21
ACTIVE_FEATURES = OPERATING_SETTINGS + [f"s{i}" for i in ACTIVE_SENSORS]

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
    MinMax normalization per operating condition (grouped by os1 rounded to 1 decimal)
    for sensors, and global StandardScaler for operating settings.
    Scalers are fit strictly on train_df.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    if val_df is not None:
        val_df = val_df.copy()

    sensor_features = [f"s{i}" for i in ACTIVE_SENSORS]
    for feature in sensor_features:
        train_df[feature] = np.float64(train_df[feature])
        test_df[feature] = np.float64(test_df[feature])
        if val_df is not None:
            val_df[feature] = np.float64(val_df[feature])

    train_regimes = train_df["os1"].round(1)
    test_regimes = test_df["os1"].round(1)
    val_regimes = val_df["os1"].round(1) if val_df is not None else None

    # Scale sensors per regime
    for regime in np.unique(train_regimes):
        train_mask = train_regimes == regime
        test_mask = test_regimes == regime
        val_mask = val_regimes == regime if val_regimes is not None else None

        if train_mask.any():
            scaler = MinMaxScaler()
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

    # Fill NaNs for constant features if any
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

    train_df, val_df, test_df = normalize_data(train_df, test_df, val_df=val_df)

    X_train, y_train = create_windows(train_df, window_size=window_size)
    X_val, y_val = create_windows(val_df, window_size=window_size)
    X_test, y_test = create_windows(test_df, window_size=window_size, last_only=True)

    return X_train, y_train, X_val, y_val, X_test, y_test
