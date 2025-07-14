#!/usr/bin/env python3
"""
time_to_review_estimator.py

Train and compare multiple regression models to predict the time it takes for
security advisories to be reviewed. Performs target winsorization, optional
signed log1p transform, k-fold cross-validation for hyperparameter tuning,
and reports MAE, RMSE, and R². Saves best models and a JSON summary of results.
"""

import json
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


def load_data():
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
	lower, upper = y.quantile(0.05), y.quantile(0.50)
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
		# 'LinearRegression': Pipeline([
		# 	# 1) keep top 20 features by univariate F-test
		# 	# ('uni', SelectKBest(score_func=f_regression)),
		# 	# 2) fit a fast forest to pick the most important of those
		# 	# ('mb', SelectFromModel(RandomForestRegressor(n_estimators=200, random_state=0), threshold='median')),
		# 	# 3) compress remaining into 95% variance
		# 	# 4) your penalized linear model
		# 	('scale', StandardScaler()),
		# 	# ('pca', PCA(n_components=0.98)),
		#
		# 	('model', LinearRegression())
		# ]),
		# 'Ridge': Pipeline([
        #     # 1) keep top 20 features by univariate F-test
        #     # ('uni', SelectKBest(score_func=f_regression, k=20)),
        #     # # 2) fit a fast forest to pick the most important of those
        #     # ('mb', SelectFromModel(RandomForestRegressor(n_estimators=200, random_state=0),
        #     #                        max_features=10, threshold='median')),
		# 	('scale', StandardScaler()),
		# 	# ('pca', PCA(n_components=0.95)),
		# 	('model', RidgeCV(alphas=np.logspace(-5, 1, 15), cv=5))
		# ]),
		# 'Lasso': Pipeline([
		#
        #     # ('uni', SelectKBest(score_func=f_regression, k=20)),
		#
        #     # ('mb', SelectFromModel(RandomForestRegressor(n_estimators=200, random_state=0),
        #     #                        max_features=10, threshold='median')),
		#
		# 	('scale', StandardScaler()),
		# 	('pca', PCA(n_components=0.95)),
		# 	('model', LassoCV(alphas=np.logspace(-5, 1, 15), cv=5, max_iter=5000))
		# ]),
		# 'ElasticNet': Pipeline([
        #     # ('uni', SelectKBest(score_func=f_regression, k=20)),
        #     # ('mb', SelectFromModel(RandomForestRegressor(n_estimators=200, random_state=0),
        #     #                        max_features=30, threshold='median')),
		# 	('scale', StandardScaler()),
		# 	# ('pca', PCA(n_components=0.95)),
		# 	('model', ElasticNetCV(
		# 		l1_ratio=[0.1, 0.5, 0.9],
		# 		alphas=np.logspace(-5, 1, 15),
		# 		cv=5,
		# 		max_iter=5000
		# 	))
		# ]),
		# 'RandomForest': RandomForestRegressor(n_jobs=-1, random_state=42),
		'ExtraTrees': ExtraTreesRegressor(n_jobs=-1, random_state=42),
		# 'GradientBoosting': GradientBoostingRegressor(random_state=42),
		# 'LightGBM': LGBMRegressor(
		# 	n_estimators=500,
		# 	learning_rate=0.05,
		# 	reg_alpha=1.0,
		# 	reg_lambda=1.0,
		# 	random_state=42,
		# 	force_col_wise=True,
		#     verbose= -1,
		#     verbosity= -1
		# ),
		# 'XGBoost': XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='rmse', verbosity= 0),
		# 'MLP': Pipeline([
		# 	('uni', SelectKBest(score_func=f_regression)),
		# 	('scale', StandardScaler()),
		# 	# ('pca', PCA(n_components=0.95)),
		# 	('model', MLPRegressor(
		# 		early_stopping=True,
		# 		max_iter=500
		# 	))
		# ])
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
			'n_estimators': [300],
			'max_depth': [None],
			'max_features': [ 0.9],
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


def evaluate_cv(X, y_trans, y_orig, capcutceil, models, grids, cv_folds=5):
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

	out_dir = BASE_DIR / config['paths']['processed_data'] / 'time_to_review_models'
	out_dir.mkdir(parents=True, exist_ok=True)

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

		preds_trans = cross_val_predict(
			best, X, y_trans, cv=cv, n_jobs=-1
		)
		preds = inv_signed_log1p(preds_trans)
		preds = np.clip(preds, 0.0, capcutceil)

		mae = mean_absolute_error(y_orig, preds)
		rmse = np.sqrt(mean_squared_error(y_orig, preds))
		r2 = r2_score(y_orig, preds)

		joblib.dump(best, out_dir / f"{name}.joblib")
		logger.info(f"Saved {name} model to {out_dir}/{name}.joblib")

		results[name] = {
			'MAE': float(mae),
			'RMSE': float(rmse),
			'R2': float(r2),
			'best_params': search.best_params_
		}
		logger.info(f"{name} -> MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

	best_model = min(results, key=lambda m: results[m]['MAE'])
	results['BestModel'] = best_model

	summary_path = out_dir / 'results_summary.json'
	with open(summary_path, 'w') as f:
		json.dump(results, f, indent=2)
	logger.info(f"Saved summary to {summary_path}")

	return results


def main():
	X, y_clipped, capcutceil = load_data()
	logger.info("Applying signed log1p to target")
	y_trans = signed_log1p(y_clipped)

	models, grids = define_models_and_grids()
	evaluate_cv(X, y_trans, y_clipped, capcutceil, models, grids, cv_folds=5)
	logger.info("Time-to-review estimator completed.")


if __name__ == '__main__':
	main()
