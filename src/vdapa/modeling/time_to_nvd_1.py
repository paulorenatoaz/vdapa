"""
Train and evaluate models to predict time_to_nvd (days) using both linear and tree-based regressors.

We apply a log1p transform to the target to mitigate heavy-tail outliers, then invert predictions for final metrics.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from vdapa.config import BASE_DIR, config
from vdapa.utils import setup_logging

# initialize logger
logger = setup_logging("modeling", "time_to_nvd")

# paths
FEATURES_TRAIN_PATH = BASE_DIR / config["paths"]["processed_data"] / "features_train.parquet"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute evaluation metrics: MAE, RMSE, R^2.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        dict with keys 'mae', 'rmse', 'r2'.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def run() -> None:
    """Runs training, validation, and reports metrics for Ridge and RandomForest models."""
    logger.info(f"Loading training features from {FEATURES_TRAIN_PATH}")
    df = pd.read_parquet(FEATURES_TRAIN_PATH)
    logger.info(f"Loaded {len(df)} training rows")

    # split features and target
    X = df.drop(columns=["time_to_nvd"])
    y = df["time_to_nvd"].astype(float)

    # log-transform to tame outliers
    y_log = np.log1p(y)

    # train/validation split
    X_train, X_val, y_train_log, y_val_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    # invert for true scale
    y_val = np.expm1(y_val_log)
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # ----- Ridge Regression -----
    logger.info("Training Ridge regression...")
    model_lin = Ridge(alpha=1.0)
    model_lin.fit(X_train, y_train_log)
    y_pred_val_log_lin = model_lin.predict(X_val)
    y_pred_val_lin = np.expm1(y_pred_val_log_lin)
    metrics_lin = compute_metrics(y_val, y_pred_val_lin)
    logger.info(
        f"Ridge Val MAE={metrics_lin['mae']:.3f}, RMSE={metrics_lin['rmse']:.3f}, R2={metrics_lin['r2']:.3f}"
    )

    # ----- Random Forest Regressor -----
    logger.info("Training RandomForestRegressor...")
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train_log)
    y_pred_val_log_rf = model_rf.predict(X_val)
    y_pred_val_rf = np.expm1(y_pred_val_log_rf)
    metrics_rf = compute_metrics(y_val, y_pred_val_rf)
    logger.info(
        f"RF Val MAE={metrics_rf['mae']:.3f}, RMSE={metrics_rf['rmse']:.3f}, R2={metrics_rf['r2']:.3f}"
    )

    logger.info("Modeling complete.")


if __name__ == "__main__":
    run()
