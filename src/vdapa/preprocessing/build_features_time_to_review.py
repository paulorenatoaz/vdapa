#!/usr/bin/env python
import re
from pathlib import Path

import numpy as np
import pandas as pd
import holidays

from vdapa.config import config, BASE_DIR
from vdapa.utils import setup_logging

logger = setup_logging("preprocessing", "build_features_time_to_review")


def count_urls(refs):
    """Conta quantas URLs únicas existem na lista de referências."""
    if not isinstance(refs, list):
        return 0
    return len({r.get("url") for r in refs if r.get("url")})


def is_holiday_series(dates: pd.Series) -> pd.Series:
    """
    Marca feriados internacionais (proxy: G7) na série de datas.
    Usa python-holidays para US, CA, GB, FR, DE, IT, JP.
    """
    years = sorted(dates.dt.year.dropna().unique().astype(int).tolist())
    # montar mapa de feriados por país
    oecd_countries = ["US", "CA", "GB", "FR", "DE", "IT", "JP"]
    hols = []
    for c in oecd_countries:
        try:
            hols.append(holidays.country_holidays(c, years=years))
        except Exception:
            continue
    def _is_hol(d):
        if pd.isna(d):
            return False
        for h in hols:
            if d.date() in h:
                return True
        return False
    return dates.map(_is_hol).astype(int)


def make_features(df_sub: pd.DataFrame, df_ref: pd.DataFrame) -> pd.DataFrame:
    """
    Monta o DataFrame de features a partir do sub-DataFrame `df_sub`.
    Usa `df_ref` somente para medianas (imputação de CVSS).
    """
    X = pd.DataFrame(index=df_sub.index)

    # --- temporais (recomputar aqui) ---
    dates = pd.to_datetime(df_sub["published_at"], errors="coerce")
    # ano
    year_median = int(dates.dt.year.median())
    X["year"] = dates.dt.year.fillna(year_median).astype(int)
    # mês → sen/cos
    month = dates.dt.month.fillna(1).astype(int)
    angle_m = 2 * np.pi * (month - 1) / 12
    X["month_sin"] = np.sin(angle_m)
    X["month_cos"] = np.cos(angle_m)
    # dia da semana → sen/cos
    dow = dates.dt.weekday.fillna(0).astype(int)  # Monday=0
    angle_w = 2 * np.pi * dow / 7
    X["dow_sin"] = np.sin(angle_w)
    X["dow_cos"] = np.cos(angle_w)
    # dia do mês → sen/cos (ciclo de 31 dias)
    dom = dates.dt.day.fillna(1).astype(int)
    angle_d = 2 * np.pi * (dom - 1) / 31
    X["dom_sin"] = np.sin(angle_d)
    X["dom_cos"] = np.cos(angle_d)

    doy = dates.dt.dayofyear.fillna(1).astype(int)
    X['doy_sin'] = np.sin(2 * np.pi * doy / 365)
    X['doy_cos'] = np.cos(2 * np.pi * doy / 365)

    # feriado
    X["is_holiday"] = is_holiday_series(dates)

    # --- numéricas ---
    X["num_refs"] = df_sub["references"].map(len)
    X["num_cwes"] = df_sub["cwe_ids"].map(len)
    # cvss
    med_cvss = df_ref["cvss_number"].median()
    X["cvss_number"] = df_sub["cvss_number"].fillna(med_cvss)
    X["cvss_number_isna"] = df_sub["cvss_number"].isna().astype(int)

    # --- categóricas ---
    # severity_label one-hot
    for lvl in ["CRITICAL", "HIGH", "LOW", "MODERATE"]:
        X[f"severity_label_{lvl}"] = (df_sub["severity_label"] == lvl).astype(int)

    # top ecosystems + “other”
    top_ecos = df_ref["ecosystem"].value_counts().nlargest(6).index
    for eco in top_ecos:
        X[f"ecosystem_{eco}"] = (df_sub["ecosystem"] == eco).astype(int)
    X["ecosystem_other"] = (~df_sub["ecosystem"].isin(top_ecos)).astype(int)

    # --- CWEs ---
    top_cwes = (
        df_ref
        .explode("cwe_ids")["cwe_ids"]
        .value_counts()
        .nlargest(10)
        .index
    )
    for cwe in top_cwes:
        X[f"has_{cwe}"] = df_sub["cwe_ids"].map(lambda lst: int(cwe in lst))
    X["has_other_cwe"] = df_sub["cwe_ids"].map(
        lambda lst: int(any(c not in top_cwes for c in lst))
    )

    # --- texto ---
    X["summary_char_count"] = df_sub["summary"].str.len().fillna(0).astype(int)
    X["details_char_count"] = df_sub["details"].str.len().fillna(0).astype(int)


    # --- target (só no train) ---
    if "time_to_review" in df_sub.columns:
        X["time_to_review"] = df_sub["time_to_review"].astype(float)

    return X


def run():
    logger.info("Loading normalized advisories")
    input_file = BASE_DIR / config["paths"]["processed_data"] / "advisories_normalized.json"
    df = pd.read_json(input_file, lines=True)
    logger.info(f"Loaded {len(df)} records")

    # split train/infer
    df_train = df[df["time_to_review"].notna()].copy()
    df_infer = df[df["time_to_review"].isna()].copy()
    logger.info(f"Train set: {len(df_train)} rows, Infer set: {len(df_infer)} rows")

    output_folder = BASE_DIR / config["paths"]["processed_data"]
    output_folder.mkdir(parents=True, exist_ok=True)

    # train
    logger.info("Assembling train features")
    X_train = make_features(df_train, df_train)
    train_file = output_folder / "features_time_to_review_train.parquet"
    X_train.to_parquet(train_file)
    logger.info(f"Saved train features to {train_file}")

    # infer
    logger.info("Assembling infer features")
    X_infer = make_features(df_infer, df_train)
    infer_file = output_folder / "features_time_to_review_infer.parquet"
    X_infer.to_parquet(infer_file)
    logger.info(f"Saved infer features to {infer_file}")

    logger.info("Feature build for time_to_review completed.")


if __name__ == "__main__":
    run()
