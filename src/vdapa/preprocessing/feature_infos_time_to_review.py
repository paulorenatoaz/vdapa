#!/usr/bin/env python
import pandas as pd
import numpy as np

from vdapa.config import config, BASE_DIR
from vdapa.utils import setup_logging

from sklearn.ensemble import RandomForestRegressor

logger = setup_logging("analysis", "features_info_time_to_review")

def main():
    # Mostra todas as colunas do DataFrame
    pd.set_option('display.max_columns', None)

    # (Opcional) aumenta a largura total do display
    pd.set_option('display.width', None)

    # 1) Carrega parquet de treino
    train_path = BASE_DIR / config["paths"]["processed_data"] / "features_time_to_review_train.parquet"
    print(f"Loading training features from {train_path}")
    df = pd.read_parquet(train_path)
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns")

    # 2) Null percentages
    print("Null % per column:")
    null_pct = df.isna().mean() * 100
    print(null_pct.sort_values(ascending=False), "\n")

    # 3) Estatísticas descritivas numéricas
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # remove o target da lista se estiver
    num_cols = [c for c in num_cols if c != "time_to_review"]
    print("Summary statistics (numéricas):")
    print(df[num_cols + ["time_to_review"]].describe().T, "\n")

    # 4) Correlação absoluta com target
    corr = df[num_cols + ["time_to_review"]].corr()["time_to_review"].abs().sort_values(ascending=False)
    print("Correlação absoluta com time_to_review:")
    print(corr, "\n")

    # 5) Importâncias via RandomForest
    print("Treinando RandomForest para obter feature_importances_")
    X = df[num_cols]
    y = df["time_to_review"]
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=num_cols).sort_values(ascending=False)
    print("Feature importances (RandomForest):")
    print(importances, "\n")

if __name__ == "__main__":
    main()
