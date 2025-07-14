#!/usr/bin/env python
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

from vdapa.config import BASE_DIR, config
from vdapa.utils import setup_logging

logger = setup_logging("modeling", "time_to_review_estimator")


def clip_target(y, lower_q=0.0, upper_q=0.5):
    """Winsorize the target to reduce the impact of extreme outliers."""
    lower = y.quantile(lower_q)
    upper = y.quantile(upper_q)
    logger.info(f"Clipping target to [{lower:.2f}, {upper:.2f}] (quantiles=({lower_q}, {upper_q}))")
    return y.clip(lower, upper), upper


def signed_log1p(y):
    """Apply a signed log1p transform."""
    return np.sign(y) * np.log1p(np.abs(y))


def inv_signed_log1p(z):
    """Inverse of the signed log1p."""
    return np.sign(z) * (np.expm1(np.abs(z)))


def load_data():
    path = BASE_DIR / config["paths"]["processed_data"] / "features_time_to_review_train.parquet"
    logger.info(f"Loading data from {path}")
    df = pd.read_parquet(path)
    # Exclude non-positive review times
    df = df[df['time_to_review'] > 0].copy()
    logger.info(f"Filtered to {len(df)} positive time_to_review rows")
    X = df.drop(columns=["time_to_review"])
    y = df["time_to_review"].astype(float)
    return X, y


def build_models():
    """Define base learners (no stacking)."""
    enet = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            alphas=np.logspace(-3, 1, 10),
            cv=5,
            max_iter=5000,
            n_jobs=-1
        ))
    ])
    lgbm = Pipeline([
        ("model", LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            force_col_wise=True
        ))
    ])
    return {"ElasticNetCV": enet, "LightGBM": lgbm}


def main():
    # 1) Load & filter
    X, y_orig = load_data()

    # 2) Winsorize target
    y_clipped, capcut = clip_target(y_orig,
                                   lower_q=config.get('model',{}).get('cap_lower', 0.0),
                                   upper_q=config.get('model',{}).get('cap_upper', 0.5))

    # 3) Transform target
    logger.info("Applying signed log1p transform to target")
    y_trans = signed_log1p(y_clipped)

    # 4) Define models
    models = build_models()

    # 5) Cross-validation evaluation
    cv_folds = config.get('model', {}).get('cv_folds', 5)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, model in models.items():
        logger.info(f"Cross-validating {name} ({cv_folds}-fold)")
        # Out-of-fold predictions on transformed scale
        preds_trans = cross_val_predict(model, X, y_trans, cv=cv, n_jobs=-1)
        # Inverse transform
        preds = inv_signed_log1p(preds_trans)
        # Cap predictions at upper quantile
        preds_capped = np.minimum(preds, capcut)

        # Metrics on clipped target
        mae = mean_absolute_error(y_clipped, preds_capped)
        rmse = np.sqrt(mean_squared_error(y_clipped, preds_capped))
        r2 = r2_score(y_clipped, preds_capped)
        logger.info(f"{name} CV -> MAE={mae:.3f} days, RMSE={rmse:.3f} days, R2={r2:.3f}")

        # 6) Refit on full data and save final model
        model.fit(X, y_trans)
        out_dir = Path(BASE_DIR) / config['paths']['processed_data'] / 'estimator_outputs'
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / f"{name}_final.joblib"
        joblib.dump(model, file_path)
        logger.info(f"Saved final {name} model to {file_path}")

    logger.info("Time-to-review estimator completed.")

if __name__ == '__main__':
    main()
