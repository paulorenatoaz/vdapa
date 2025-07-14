"""
Analyze feature relevance for predicting time to NVD confirmation.

This script loads the normalized advisory dataset, prepares input features and target,
and prints lists of feature importance based on correlation with the target and a
Random Forest Regressor model.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from vdapa.config import BASE_DIR, config
from vdapa.utils import setup_logging
from cvss import CVSS3


logger = setup_logging("modeling", "train_and_predict")

def extract_cvss_score(severity):
    """Extracts the first CVSS base score from severity field.

    Args:
        severity (list): List of severity dicts.

    Returns:
        float or None: CVSS numeric score if parseable.
    """
    if isinstance(severity, list):
        for item in severity:
            if isinstance(item, dict) and "score" in item:
                try:
                    return CVSS3(item["score"]).score()
                except Exception:
                    return None
    return None

def load_data():
    """Loads the normalized advisories dataset.

    Returns:
        DataFrame: Loaded and parsed dataset.
    """
    path = BASE_DIR / config["paths"]["processed_data"] / "advisories_normalized.json"
    df = pd.read_json(path, lines=True, convert_dates=["published_at", "github_reviewed_at", "nvd_published_at"])
    return df

def prepare_features(df):
    """Extracts and encodes features for model relevance analysis.

    Args:
        df (DataFrame): Input dataset.

    Returns:
        X (DataFrame): Feature matrix.
        y (Series): Target variable (time_to_nvd).
    """
    df = df.copy()
    df = df[df["time_to_nvd"].notna()].copy()

    df["year"] = df["published_at"].dt.year
    df["month"] = df["published_at"].dt.month
    df["has_cve"] = df["cve_ids"].apply(lambda x: isinstance(x, str) and len(x) > 0 or isinstance(x, list) and len(x) > 0)
    df["was_withdrawn"] = df["withdrawn_at"].notna()
    df["is_reviewed"] = df["github_reviewed"]
    df["time_to_review"] = df["time_to_review"].fillna(-1)
    df["has_aliases"] = df.get("aliases", pd.Series([[]]*len(df))).apply(lambda x: isinstance(x, list) and len(x) > 0)
    df["num_references"] = df.get("references", pd.Series([[]]*len(df))).apply(lambda x: len(x) if isinstance(x, list) else 0)
    df["num_cwes"] = df.get("cwe_ids", pd.Series([[]]*len(df))).apply(lambda x: len(x) if isinstance(x, list) else 0)

    df["cvss_score"] = df.get("severity").apply(extract_cvss_score)

    categorical = ["ecosystem", "severity_label"]
    df_cat = pd.get_dummies(df[categorical].astype(str), dummy_na=True)

    numeric_cols = [        "year", "month", "has_cve", "was_withdrawn", "is_reviewed",
        "time_to_review", "has_aliases", "num_references", "num_cwes", "cvss_score"
    ]

    X = pd.concat([
        df[numeric_cols].astype(float),
        df_cat
    ], axis=1)

    y = df["time_to_nvd"]
    return X, y

def analyze_feature_relevance(X, y):
    """Analyzes and prints the importance of features.

    Args:
        X (DataFrame): Feature matrix.
        y (Series): Target variable.
    """
    print("\n--- Pearson correlation with target ---")
    correlations = X.corrwith(y).sort_values(key=abs, ascending=False)
    for i, (feature, value) in enumerate(correlations.items(), 1):
        print(f"{i}. {feature} - {value:.5f}")

    print("\n--- RandomForestRegressor feature importance ---")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    for i, (feature, value) in enumerate(importances.items(), 1):
        print(f"{i}. {feature} - {value:.5f}")

def run():
    """Runs the feature relevance analysis pipeline."""
    logger.info("Starting feature relevance analysis...")
    df = load_data()
    X, y = prepare_features(df)
    analyze_feature_relevance(X, y)
    logger.info("Feature relevance analysis completed.")


if __name__ == "__main__":
    run()
