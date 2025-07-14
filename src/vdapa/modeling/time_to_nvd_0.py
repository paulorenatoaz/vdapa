import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from vdapa.config import BASE_DIR, config
from vdapa.utils import setup_logging

logger = setup_logging("modeling", "time_to_nvd")


def load_data(path):
    """
    Loads feature dataset from a Parquet file.

    Args:
        path (str or Path): Path to the feature Parquet.

    Returns:
        DataFrame: Features and target.
    """
    df = pd.read_parquet(path)
    return df


def compute_metrics(y_true, y_pred):
    """
    Computes MAE, RMSE and R2 between true and predicted values.

    Args:
        y_true (array-like): Ground-truth target.
        y_pred (array-like): Predicted target.

    Returns:
        dict: {'mae': ..., 'rmse': ..., 'r2': ...}
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def run():
    """
    Executes the modeling pipeline:
      - Load features
      - Split train/validation
      - Fit and evaluate a linear (Ridge) and tree (RandomForest)
      - Report metrics
    """
    # Paths
    train_fp = BASE_DIR / config['paths']['processed_data'] / 'features_train.parquet'

    # Load
    logger.info(f"Loading training features from {train_fp}")
    df = load_data(train_fp)
    logger.info(f"Loaded {len(df)} training rows")

    # Split into X, y
    X = df.drop('time_to_nvd', axis=1)
    y = df['time_to_nvd']

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # Linear model
    lin = Ridge(random_state=42)
    logger.info("Training Ridge regression...")
    lin.fit(X_train, y_train)
    y_pred_val_lin = lin.predict(X_val)
    metrics_lin = compute_metrics(y_val, y_pred_val_lin)
    logger.info(f"Ridge Val MAE={metrics_lin['mae']:.3f}, RMSE={metrics_lin['rmse']:.3f}, R2={metrics_lin['r2']:.3f}")

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    logger.info("Training RandomForestRegressor...")
    rf.fit(X_train, y_train)
    y_pred_val_rf = rf.predict(X_val)
    metrics_rf = compute_metrics(y_val, y_pred_val_rf)
    logger.info(f"RF Val MAE={metrics_rf['mae']:.3f}, RMSE={metrics_rf['rmse']:.3f}, R2={metrics_rf['r2']:.3f}")

    # Summary
    logger.info("Modeling complete.")


if __name__ == '__main__':
    run()
