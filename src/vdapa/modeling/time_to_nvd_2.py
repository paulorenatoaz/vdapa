"""
Train and evaluate models to predict time_to_nvd (days) using both linear and tree-based regressors.

We apply a 99th percentile outlier removal, a log1p transform on the target to mitigate heavy-tail effects,
then invert predictions for final metrics.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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

    # ensure target exists and convert to float
    df = df[df["time_to_nvd"].notna()].copy()
    df["time_to_nvd"] = df["time_to_nvd"].astype(float)

    # remove extreme outliers beyond 99th percentile
    threshold = df["time_to_nvd"].quantile(0.95)
    n_outliers = (df["time_to_nvd"] > threshold).sum()
    logger.info(f"Removing {n_outliers} rows with time_to_nvd > {threshold:.1f}")
    df = df[df["time_to_nvd"] <= threshold]

    # split features and target
    X = df.drop(columns=["time_to_nvd"])
    y = df["time_to_nvd"]

    # log-transform to tame remaining heavy-tail
    y_log = np.log1p(y)

    # train/validation split
    X_train, X_val, y_train_log, y_val_log = train_test_split(
        X, y_log, test_size=0.2
    )
    y_val = np.expm1(y_val_log)
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # ----- Ridge Regression -----
    logger.info("Training Ridge regression...")
    model_lin = Ridge(alpha=1.0)
    model_lin.fit(X_train, y_train_log)
    y_pred_val_lin = np.expm1(model_lin.predict(X_val))
    metrics_lin = compute_metrics(y_val, y_pred_val_lin)
    logger.info(
        f"Ridge Val MAE={metrics_lin['mae']:.3f}, RMSE={metrics_lin['rmse']:.3f}, R2={metrics_lin['r2']:.3f}"
    )

    bins = np.linspace(min(y_val.min(), y_pred_val_lin.min()),
                       max(y_val.max(), y_pred_val_lin.max()), 50)

    plt.figure()
    plt.hist(y_val, bins=bins, density=True, histtype='step', label='Ground Truth')
    plt.hist(y_pred_val_lin, bins=bins, density=True, histtype='step', label='Prediction')
    plt.xlabel('time_to_nvd (dias)')
    plt.ylabel('Densidade')
    plt.title('PDF: Verdadeiro vs Estimado')
    plt.legend()
    plt.show()

    df = pd.DataFrame({
	    'X_n': y_val,
	    'f(X_n)': y_pred_val_lin
    }).sort_values(by=['X_n', 'f(X_n)'])
    # imprime todas as linhas, sem índice extra
    print(df.to_string(index=False, header=[ 'Xₙ', 'f(Xₙ)']))

    # ----- Random Forest Regressor -----
    logger.info("Training RandomForestRegressor...")
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train_log)
    y_pred_val_rf = np.expm1(model_rf.predict(X_val))
    metrics_rf = compute_metrics(y_val, y_pred_val_rf)
    logger.info(
        f"RF Val MAE={metrics_rf['mae']:.3f}, RMSE={metrics_rf['rmse']:.3f}, R2={metrics_rf['r2']:.3f}"
    )

    logger.info("Modeling complete.")


    bins = np.linspace(min(y_val.min(), y_pred_val_rf.min()),
                       max(y_val.max(), y_pred_val_rf.max()), 50)

    plt.figure()
    plt.hist(y_val, bins=bins, density=True, histtype='step', label='Ground Truth')
    plt.hist(y_pred_val_rf, bins=bins, density=True, histtype='step', label='Prediction')
    plt.xlabel('time_to_nvd (dias)')
    plt.ylabel('Densidade')
    plt.title('PDF: Verdadeiro vs Estimado')
    plt.legend()
    plt.show()

    df = pd.DataFrame({
	    'X_n': y_val,
	    'f(X_n)': y_pred_val_rf
    }).sort_values(by= ['X_n', 'f(X_n)'])
    # imprime todas as linhas, sem índice extra
    print(df.to_string(index=False, header=[ 'Xₙ', 'f(Xₙ)']))


if __name__ == "__main__":
    run()
