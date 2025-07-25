#!/usr/bin/env python3
"""
time_to_review_estimator_bench.py

Train and compare multiple regression models to predict the time it takes for
security advisories to be reviewed. Performs target winsorization, optional
signed log1p transform, k-fold cross-validation for hyperparameter tuning,
and reports MAE, RMSE, and R². Saves best models and a JSON summary of results.
"""

import json
import time
from pathlib import Path
from tabnanny import verbose

import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_regression

from sklearn.model_selection import GridSearchCV, cross_val_predict, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
	LinearRegression, RidgeCV, LassoCV, ElasticNetCV
)

from sklearn.ensemble import (
	RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.metrics import (
	mean_absolute_error, mean_squared_error, r2_score, make_scorer
)

from vdapa.config import BASE_DIR, config
from vdapa.utils import setup_logging

logger = setup_logging("modeling", "time_to_review_estimator_bench")


def load_data(quantiles):
	"""
    Load features and target from parquet, filter out non-positive targets.
    Returns X (DataFrame), y_clipped (Series), and capcutceil (float).
    """
	path = BASE_DIR / config['paths']['processed_data'] / 'features_time_to_review_train.parquet'
	logger.info(f"Loading data from {path}")
	df = pd.read_parquet(path)
	initial_count = len(df)
	df = df[df['time_to_review'] > 0].copy()
	filtered_count = len(df)
	logger.info(f"Filtered to {filtered_count} positive time_to_review rows (dropped {initial_count - filtered_count})")

	y = df['time_to_review'].astype(float)
	lower, upper = y.quantile(quantiles[0]), y.quantile(quantiles[1])
	logger.info(f"Clipping target to [{lower:.2f}, {upper:.2f}]")
	y_clipped = y.clip(lower, upper)
	capcutceil = upper

	X = df.drop(columns=['time_to_review'])
	return X, y_clipped, capcutceil


def signed_log1p(y):
	"""
    Signed log1p transform: sign(y) * log1p(abs(y)).
    """
	return np.sign(y) * np.log1p(np.abs(y))


def inv_signed_log1p(z):
	"""
    Inverse of signed_log1p: sign(z) * (exp(|z|) - 1).
    """
	return np.sign(z) * (np.expm1(np.abs(z)))


def define_models_and_grids():
	"""
    Define regression models and their hyperparameter grids.
    Returns models dict and grids dict.
    """
	models = {
		'LinearRegression': Pipeline([
			# 1) keep top 20 features by univariate F-test
			# ('uni', SelectKBest(score_func=f_regression)),
			# 2) fit a fast forest to pick the most important of those
			# ('mb', SelectFromModel(RandomForestRegressor(n_estimators=200, random_state=0), threshold='median')),
			# 3) compress remaining into 95% variance
			# 4) your penalized linear model
			('scale', StandardScaler()),
			# ('pca', PCA(n_components=0.98)),

			('model', LinearRegression())
		]),
		'Ridge': Pipeline([
            # 1) keep top 20 features by univariate F-test
            # ('uni', SelectKBest(score_func=f_regression, k=20)),
            # # 2) fit a fast forest to pick the most important of those
            # ('mb', SelectFromModel(RandomForestRegressor(n_estimators=200, random_state=0),
            #                        max_features=10, threshold='median')),
			('scale', StandardScaler()),
			# ('pca', PCA(n_components=0.95)),
			('model', RidgeCV(alphas=np.logspace(-5, 1, 15), cv=5))
		]),
		'Lasso': Pipeline([

            # ('uni', SelectKBest(score_func=f_regression, k=20)),

            # ('mb', SelectFromModel(RandomForestRegressor(n_estimators=200, random_state=0),
            #                        max_features=10, threshold='median')),

			('scale', StandardScaler()),
			# ('pca', PCA(n_components=0.95)),
			('model', LassoCV(alphas=np.logspace(-5, 1, 15), cv=5, max_iter=5000))
		]),
		'ElasticNet': Pipeline([
            # ('uni', SelectKBest(score_func=f_regression, k=20)),
            # ('mb', SelectFromModel(RandomForestRegressor(n_estimators=200, random_state=0),
            #                        max_features=30, threshold='median')),
			('scale', StandardScaler()),
			# ('pca', PCA(n_components=0.95)),
			('model', ElasticNetCV(
				l1_ratio=[0.1, 0.5, 0.9],
				alphas=np.logspace(-5, 1, 15),
				cv=5,
				max_iter=5000
			))
		]),
		'RandomForest': RandomForestRegressor(n_jobs=-1, random_state=42),
		'ExtraTrees': ExtraTreesRegressor(n_jobs=-1, random_state=42),
		'GradientBoosting': GradientBoostingRegressor(random_state=42),
		'LightGBM': LGBMRegressor(
			n_estimators=500,
			learning_rate=0.05,
			reg_alpha=1.0,
			reg_lambda=1.0,
			random_state=42,
			force_col_wise=True,
		    verbose= -1,
		    verbosity= -1
		),
		'XGBoost': XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='rmse', verbosity= 0),
		'MLP': Pipeline([
			('uni', SelectKBest(score_func=f_regression)),
			('scale', StandardScaler()),
			# ('pca', PCA(n_components=0.95)),
			('model', MLPRegressor(
				early_stopping=True,
				max_iter=500
			))
		])
	}

	grids = {
		'LinearRegression': {

			# 'mb__max_features': [ 24, 38],

		},
		'Ridge': {
			# 'uni__k': [15, 20, 30],
			# 'mb__max_features': [7, 8, 12, 15],

		},
		'Lasso': {
			# 'uni__k': [15],
			# 'mb__max_features': [8],
		},
		'ElasticNet': {
			# 'uni__k': [15, 20],
			# 'mb__max_features': [8, 12]
		},
		'RandomForest': {
			'n_estimators': [400],
			'max_depth': [14],
			'min_samples_leaf': [3],
			'min_samples_split': [8]
		},
		'ExtraTrees': {
			'n_estimators': [500],
			'max_depth': [None],
			'max_features': [0.8],
			'min_samples_leaf': [1],
			'min_samples_split': [2]
		},
		'GradientBoosting': {
			'learning_rate': [0.1],
			'n_estimators': [600],
			'max_depth': [5]
		},
		'LightGBM': {   'learning_rate': [0.1],
                        'n_estimators': [600],
                        'max_depth': [5]},
		'XGBoost': {
			'learning_rate': [0.1],
			'n_estimators': [600],
			'max_depth': [5]
		},
		'MLP': {
			'uni__k': [23],
			'model__hidden_layer_sizes': [(50, 50)],
			'model__alpha': [1e-5],
			'model__learning_rate_init': [1e-2],
		}
	}
	return models, grids


def evaluate_cv(X, y_trans, y_orig, capcutceil, models, grids, quantiles, cv_folds=5):
	"""
    Cross-validate each model, compute out-of-fold predictions,
    clamp predictions to [0, capcutceil], and report MAE, RMSE, R2.
    Save each trained model and a JSON summary.
    """

	def real_days_neg_mae(estimator, X_test, y_test_trans):
		# invert true values
		y_true = inv_signed_log1p(y_test_trans)
		# predict & invert
		y_pred_trans = estimator.predict(X_test)
		y_pred = inv_signed_log1p(y_pred_trans)
		# clamp
		y_pred = np.clip(y_pred, 0.0, capcutceil)
		# return negative MAE so GridSearchCV maximizes it
		return -mean_absolute_error(y_true, y_pred)

	out_dir = BASE_DIR / config['paths']['processed_data'] / 'time_to_review_model'
	out_dir.mkdir(parents=True, exist_ok=True)
	timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
	results = {}
	for name, model in models.items():
		logger.info(f"Tuning and CV for {name}")
		grid = grids.get(name, {})

		cv = KFold(n_splits=5, shuffle=True, random_state=42)
		search = GridSearchCV(
			estimator=model,
			param_grid=grid,
			cv=cv,
			scoring=real_days_neg_mae,
			n_jobs=-1,
            verbose=1
		)
		search.fit(X, y_trans)
		best = search.best_estimator_
		logger.info(f"Best params for {name}: {search.best_params_}")
		best_idx = search.best_index_
		mean_fit = search.cv_results_['mean_fit_time'][best_idx]
		mean_score = search.cv_results_['mean_score_time'][best_idx]
		std_fit = search.cv_results_['std_fit_time'][best_idx]
		std_score = search.cv_results_['std_score_time'][best_idx]

		t0 = time.perf_counter()

		preds_trans = cross_val_predict(
			best, X, y_trans, cv=cv, n_jobs=-1
		)
		fit_timer = time.perf_counter() - t0
		preds = inv_signed_log1p(preds_trans)
		preds = np.clip(preds, 0.0, capcutceil)

		mae = mean_absolute_error(y_orig, preds)
		rmse = np.sqrt(mean_squared_error(y_orig, preds))
		r2 = r2_score(y_orig, preds)

		# joblib.dump(best, out_dir / f"{name}.joblib")
		# logger.info(f"Saved {name} model to {out_dir}/{name}.joblib")

		results[name] = {
			'model': name,
			'MAE': float(mae),
			'RMSE': float(rmse),
			'R2': float(r2),
			'best_params': search.best_params_,
			'mean_fit_time': float(mean_fit),
			'std_fit_time': float(std_fit),
			'mean_score_time': float(mean_score),
			'std_score_time': float(std_score),
			'fit_time': float(fit_timer),
			'cv_folds': cv_folds,
		}
		logger.info(f"{name} -> MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

	best_model = min(results, key=lambda m: results[m]['MAE'])
	results['timestamp'] = timestamp
	results['bestmodel'] = best_model
	results['capcutceil'] = int(capcutceil)
	results['quantiles'] = quantiles



	#load results summary base to add new results
	summary_path = out_dir / 'train_summary.json'
	train_summary = []
	if summary_path.exists():
		with open(summary_path, 'r') as f:
			train_summary = json.load(f)

	train_summary.append(results)
	#order by timestamp
	train_summary = sorted(train_summary, key=lambda x: x['timestamp'], reverse=True)


	with open(summary_path, 'w') as f:
		json.dump(train_summary, f, indent=2)
	logger.info(f"Saved summary to {summary_path}")

	return results

import joblib
from pathlib import Path

def train_final_model(X, y_trans, models, results):
    """
    Retrain the best model on the entire dataset using its tuned hyperparameters,
    then save it for production use.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix used in cross-val.
    y_trans : np.ndarray or pd.Series
        Transformed target used in cross-val (e.g. signed_log1p(y_clipped)).
    models : dict[str, sklearn estimator]
        The dict you got from define_models_and_grids().
    results : dict
        The dict you wrote out to JSON, must contain:
          - results['bestmodel'] : name of the winning model
          - results[model_name]['best_params'] : dict of its best params
    output_subdir : str
        Sub-folder (under BASE_DIR/processed_data) to write the final model.

    Returns
    -------
    Path
        Full path to the saved joblib file.
    """
    winner = results.get("bestmodel")
    if winner is None or winner not in models:
        logger.error(f"Could not find winning model in results: {winner!r}")
        return

    # grab pipeline, inject tuned params, refit on full data
    pipe = models[winner]
    best_params = results[winner]["best_params"]
    logger.info(f"Retraining {winner} on full data with params: {best_params}")
    pipe.set_params(**best_params)
    pipe.fit(X, y_trans)

    # save out
    out_dir = Path(BASE_DIR) / config["paths"]["processed_data"] / 'time_to_review_model'
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"train_{results['timestamp']}_{winner}.joblib"
    joblib.dump(pipe, dest)
    logger.info(f"Saved final {winner} model to {dest}")
    return dest


def run():
	quantiles = (0.0, 0.5)  # lower and upper quantiles for clipping
	X, y_clipped, capcutceil = load_data(quantiles)
	logger.info("Applying signed log1p to target")
	y_trans = signed_log1p(y_clipped)

	models, grids = define_models_and_grids()
	results = evaluate_cv(X, y_trans, y_clipped, capcutceil, models, grids, quantiles, cv_folds=5)
	logger.info("Cross-validation and model tuning completed.")

	# retrain best model on full data
	train_final_model(X, y_trans, models, results)
	logger.info("Final model retrained on full data.")


	logger.info("Time-to-review estimator completed.")



if __name__ == '__main__':
	run()
