import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from sksurv.metrics import (
    cumulative_dynamic_auc,
    concordance_index_censored,
    integrated_brier_score,
)


def calculate_score(y_true, y_pred):
    """
    Standard CMAPSS Score function (vectorized).
    """
    y_true = np.asanyarray(y_true).flatten()
    y_pred = np.asanyarray(y_pred).flatten()
    y_pred = np.clip(y_pred, 0, 125)
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return np.sum(score)


def calculate_rmse(y_true, y_pred):
    """
    Root Mean Square Error.
    """
    y_true = np.asanyarray(y_true).flatten()
    y_pred = np.asanyarray(y_pred).flatten()
    y_pred = np.clip(y_pred, 0, 125)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true, y_pred):
    """
    Mean Absolute Error.
    """
    y_true = np.asanyarray(y_true).flatten()
    y_pred = np.asanyarray(y_pred).flatten()
    y_pred = np.clip(y_pred, 0, 125)
    return mean_absolute_error(y_true, y_pred)


def calculate_map(y_true, y_pred):
    """
    Mean Absolute Percentage Error (Safe for zero).
    """
    y_true = np.asanyarray(y_true).flatten()
    y_pred = np.asanyarray(y_pred).flatten()
    y_pred = np.clip(y_pred, 0, 125)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9)))


def calculate_r2(y_true, y_pred):
    """
    R-squared (Coefficient of Determination).
    """
    y_true = np.asanyarray(y_true).flatten()
    y_pred = np.asanyarray(y_pred).flatten()
    y_pred = np.clip(y_pred, 0, 125)
    return r2_score(y_true, y_pred)


def calculate_concordance_index(y_test, y_pred):
    """
    Concordance Index.
    """
    y_test = np.asanyarray(y_test).flatten()
    y_pred = np.asanyarray(y_pred).flatten()
    y_pred = np.clip(y_pred, 0, 125)
    surv_test = np.array(
        [(True, y) for y in y_test], dtype=[("event", "bool"), ("time", "float")]
    )
    risk_scores = -y_pred
    result = concordance_index_censored(
        surv_test["event"], surv_test["time"], risk_scores
    )
    return result[0]


def calculate_brier_score(y_train, y_test, y_pred):
    """
    Integrated Brier Score (Vectorized).
    """
    y_test = np.asanyarray(y_test).flatten()
    y_pred = np.asanyarray(y_pred).flatten()
    y_pred = np.clip(y_pred, 0, 125)
    surv_train = np.array(
        [(True, y) for y in y_train], dtype=[("event", "bool"), ("time", "float")]
    )
    surv_test = np.array(
        [(True, y) for y in y_test], dtype=[("event", "bool"), ("time", "float")]
    )

    # Define evaluation time points
    min_time = np.min(y_test)
    max_time = np.max(y_test)
    if min_time == max_time:
        return np.nan

    times = np.percentile(y_test, np.linspace(5, 95, 15))
    times = np.unique(times)
    times = times[(times > min_time) & (times < max_time)]

    if len(times) == 0:
        return np.nan

    # Point predictions as survival probabilities (StepFunction at y_pred)
    # S(t) = 1 if t < y_pred else 0
    # Vectorized: (N, 1) > (1, T) -> (N, T)
    preds = (y_pred[:, np.newaxis] > times).astype(float)

    try:
        score = integrated_brier_score(surv_train, surv_test, preds, times)
        return score
    except Exception as e:
        print(f"Warning: Could not calculate Brier Score: {e}")
        return np.nan


def calculate_cd_auc(y_train, y_test, y_pred):
    """
    Cumulative Dynamic AUC.
    Treats the RUL prediction problem as a survival analysis task where:
    - Event is always True (run to failure).
    - Time is the RUL.
    - Risk is the negative predicted RUL (lower RUL = higher risk).
    """
    y_test = np.asanyarray(y_test).flatten()
    y_pred = np.asanyarray(y_pred).flatten()
    y_pred = np.clip(y_pred, 0, 125)
    surv_train = np.array(
        [(True, y) for y in y_train], dtype=[("event", "bool"), ("time", "float")]
    )
    surv_test = np.array(
        [(True, y) for y in y_test], dtype=[("event", "bool"), ("time", "float")]
    )

    # Evaluate at 25th, 50th, and 75th percentiles of the test times
    # Ensure times are strictly within the observed range to avoid errors
    min_time = np.min(y_test)
    max_time = np.max(y_test)

    if min_time == max_time:
        return np.nan

    p25 = np.percentile(y_test, 25)
    p50 = np.percentile(y_test, 50)
    p75 = np.percentile(y_test, 75)

    # Fallback if percentiles are too close to boundaries due to clipping (e.g. lots of 125s)
    times = np.unique([p25, p50, p75])
    times = times[(times > min_time) & (times < max_time)]

    if len(times) == 0:
        # If no valid percentiles, just pick the midpoint
        times = np.array([(min_time + max_time) / 2.0])

    risk_scores = -y_pred

    try:
        _, mean_auc = cumulative_dynamic_auc(surv_train, surv_test, risk_scores, times)
        return mean_auc
    except Exception as e:
        print(f"Warning: Could not calculate CD-AUC: {e}")
        return np.nan
