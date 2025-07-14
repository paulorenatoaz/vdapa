#!/usr/bin/env python
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from vdapa.config import BASE_DIR, config
from vdapa.utils import setup_logging

logger = setup_logging('modeling', 'time_to_review_survival')


def load_features():
    path = BASE_DIR / config['paths']['processed_data'] / 'features_time_to_review_train.parquet'
    logger.info(f"Loading features from {path}")
    df = pd.read_parquet(path)
    df = df[df['time_to_review'] > 0].copy()
    df['duration_orig'] = df['time_to_review']
    return df


def prepare_survival_data(df):
    # 1) winsorize
    cap_q = config.get('model', {}).get('cap_quantiles', (0.0, 0.60))
    lower, upper = df['duration_orig'].quantile(cap_q).values
    logger.info(f"Capping durations to [{lower:.0f}, {upper:.0f}] (quantiles={cap_q})")

    # 2) censura + clip
    orig = df['duration_orig'].values
    events = orig <= upper
    clipped = np.clip(orig, lower, upper)

    # 3) log‐transform opcional
    if config.get('model', {}).get('log_duration', True):
        durations = np.log1p(clipped)
        capcut = np.log1p(upper)
        logger.info("Applied log1p transform to durations")
    else:
        durations = clipped
        capcut = upper

    # 4) montar y
    y = np.array(
        [(e, d) for e, d in zip(events, durations)],
        dtype=[('event', '?'), ('duration', '<f8')]
    )

    # 5) features
    X = df.drop(columns=['time_to_review', 'duration_orig'])
    return X, y, capcut


def define_models_and_grids():
    coxridge = Pipeline([
        ('scale', StandardScaler()),
        ('model', CoxnetSurvivalAnalysis(
            l1_ratio=1e-4,
            fit_baseline_model=True
        ))
    ])
    rsf = RandomSurvivalForest(
        n_estimators=50,
        min_samples_split=10,
        min_samples_leaf=15,
        max_depth=5,
        bootstrap=True,
        max_samples=0.7,
        n_jobs=1,
        # random_state=42
    )
    # gbsurv = Pipeline([
    #     ('scale', StandardScaler()),
    #     ('model', GradientBoostingSurvivalAnalysis(random_state=42))
    # ])

    models = {'CoxRidge': coxridge, 'RSF': rsf}#, 'GBSurv': gbsurv}
    grids = {
        'CoxRidge': {'model__alphas': [np.logspace(-4, 2, 10)]},
        'RSF': {
            'n_estimators': [500],
            'min_samples_leaf': [5],
            'min_samples_split': [5],
            'max_depth': [10]
        },
        'GBSurv': {
            'model__learning_rate': [0.01],
            'model__n_estimators': [50],
            'model__max_depth': [5]
        }
    }
    return models, grids


def extract_median_time(surv_funcs, capcut):
    """Dado um array de survival_functions, retorna vetor de tempos medianos, truncado."""
    preds = []
    for sf in surv_funcs:
        # acha primeiro índice onde prob ≤ 0.5
        idx = np.where(sf.y <= 0.5)[0]
        if idx.size:
            t0 = sf.x[idx[0]]
        else:
            t0 = sf.x[-1]
        # truncar
        preds.append(min(t0, capcut))
    return np.array(preds)


def cindex_scorer(est, X_test, y_test):
    """C-index censurado com predição truncada."""
    # pegar capcut que gravamos como atributo
    capcut = cindex_scorer.capcut

    if hasattr(est, 'predict_survival_function'):
        sf_list = est.predict_survival_function(X_test)
        pred_t = extract_median_time(sf_list, capcut)
    else:
        pred_raw = -est.predict(X_test)
        pred_t = np.minimum(pred_raw, capcut)

    # invertendo sinal: menor t → maior risco
    return concordance_index_censored(y_test['event'], y_test['duration'], -pred_t)[0]


def evaluate_cv(models, grids, X, y, capcut):
    out_dir = Path(BASE_DIR / config['paths']['processed_data'] / 'survival_outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # pendura capcut no scorer
    cindex_scorer.capcut = capcut
    n_folds = config.get('model', {}).get('cv_folds', 5)

    for name, model in models.items():
        logger.info(f"Tuning and CV for {name}")
        grid = grids.get(name, {})
        search = GridSearchCV(
            estimator=model,
            param_grid=grid,
            cv=n_folds,
            scoring=cindex_scorer,
            n_jobs=1,
            verbose=1,
            error_score='raise'
        )
        search.fit(X, y)
        best = search.best_estimator_
        cidx = search.best_score_

        logger.info(f"{name} CV best C-index: {cidx:.3f}, params: {search.best_params_}")
        results[name] = cidx

        # salva para produção
        fp = out_dir / f"{name}_cv_best.joblib"
        joblib.dump(best, fp)
        logger.info(f"Saved {name} model to {fp}")

        # MAE censurado (opcional)
        if hasattr(best, 'predict_survival_function'):
            sf_list = best.predict_survival_function(X)
            preds = extract_median_time(sf_list, capcut)
        else:
            preds = np.minimum(-best.predict(X), capcut)

        preds_days = np.expm1(preds)
        true_days = np.expm1(y['duration'])
        mae = np.mean(np.abs(preds_days - true_days))
        logger.info(f"{name} MAE censurado: {mae:.3f}")

    # grava CSV resumo
    pd.DataFrame.from_dict(results, orient='index', columns=['C-index']) \
      .to_csv(out_dir / 'survival_cv_cindex.csv')
    logger.info(f"Saved CV C-index summary to {out_dir / 'survival_cv_cindex.csv'}")


def main():
    df = load_features()
    X, y, capcut = prepare_survival_data(df)
    models, grids = define_models_and_grids()
    evaluate_cv(models, grids, X, y, capcut)
    logger.info("Survival CV analysis completed.")


if __name__ == '__main__':
    main()
