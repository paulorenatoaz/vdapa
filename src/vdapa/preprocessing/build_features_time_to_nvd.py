"""
Build features datasets for model training and inference.

This script reads the full normalized advisories JSON, engineers a fixed set of features,
filters train (with target) vs. infer (no target), and writes two Parquet files ready for modeling:

- features_train.parquet  -> records with `time_to_nvd` + target
- features_infer.parquet  -> records without `time_to_nvd` (for real-time inference)
"""
from pathlib import Path
import pandas as pd
from vdapa.config import BASE_DIR, config
from vdapa.utils import setup_logging

# initialize logger
logger = setup_logging("preprocessing", "build_features")


def build_features():
    """
    Reads normalized advisories, engineers numeric/categorical/CWE features,
    splits into train/infer sets, and writes Parquet files.
    """
    # Paths
    processed_dir = Path(BASE_DIR) / config["paths"]["processed_data"]
    input_file = processed_dir / "advisories_normalized.json"
    output_train = processed_dir / "features_train.parquet"
    output_infer = processed_dir / "features_infer.parquet"

    logger.info(f"Loading data from {input_file}")
    # Load full dataset
    df = pd.read_json(
        input_file,
        lines=True,
        convert_dates=["published_at", "modified_at", "github_reviewed_at", "nvd_published_at"],
    )
    logger.info(f"Loaded {len(df)} advisories")

    # Impute cvss_number and flag
    if "cvss_number" in df.columns:
        median_cvss = df["cvss_number"].median()
        logger.info(f"Imputing cvss_number missing values with median: {median_cvss:.2f}")
        df["cvss_number_isna"] = df["cvss_number"].isna().astype(int)
        df["cvss_number"] = df["cvss_number"].fillna(median_cvss)

    # Select top ecosystems
    top_ecos = df["ecosystem"].value_counts().nlargest(6).index.tolist()
    logger.info(f"Top ecosystems: {top_ecos}")
    df["ecosystem_mod"] = df["ecosystem"].where(
        df["ecosystem"].isin(top_ecos), other="other_ecosystem"
    )

    # Identify top-10 CWEs from training set
    df_train_all = df[df["time_to_nvd"].notna()]
    all_cwes = df_train_all["cwe_ids"].explode()
    top_cwes = all_cwes.value_counts().nlargest(10).index.tolist()
    logger.info(f"Top-10 CWE IDs: {top_cwes}")

    # Split train vs infer
    df_train = df_train_all.copy()
    df_infer = df[df["time_to_nvd"].isna()].copy()
    logger.info(f"Train set: {len(df_train)} records, Infer set: {len(df_infer)} records")

    # Common feature columns
    num_cols = ["num_refs", "num_cwes", "year", "month_sin", "month_cos"]
    cat_cols = ["severity_label", "ecosystem_mod"]
    impute_cols = []
    if "cvss_number" in df.columns:
        num_cols.append("cvss_number")
        impute_cols.append("cvss_number_isna")

    # build CWE flags
    def make_cwe_flags(df_):
        flags = pd.DataFrame(index=df_.index)
        for c in top_cwes:
            flags[f"has_{c}"] = df_["cwe_ids"].apply(lambda lst: c in lst).astype(int)
        flags["has_other_cwe"] = df_["cwe_ids"].apply(
            lambda lst: any(c not in top_cwes for c in lst)
        ).astype(int)
        return flags

    # One-hot encode categoricals
    def encode_cats(df_, cols):
        return pd.get_dummies(df_[cols].astype(str), dummy_na=False)

    # Assemble features for a given split
    def assemble(df_, split_name):
        logger.info(f"Assembling features for {split_name}, {len(df_)} rows")
        parts = [df_[num_cols]]
        if impute_cols:
            parts.append(df_[impute_cols])
        parts.append(encode_cats(df_, cat_cols))
        parts.append(make_cwe_flags(df_))
        features = pd.concat(parts, axis=1)
        logger.info(
            f"  -> numeric: {num_cols}, impute flags: {impute_cols}, "
            f"categorical: {cat_cols}, CWE flags: {len(top_cwes)+1} cols"
        )
        return features

    # Prepare train set
    X_train = assemble(df_train, "train")
    y_train = df_train["time_to_nvd"].astype(float)
    train_df = X_train.copy()
    train_df["time_to_nvd"] = y_train
    logger.info(f"Writing train features to {output_train}")
    train_df.to_parquet(output_train, index=False)

    # Prepare inference set
    X_infer = assemble(df_infer, "infer")
    logger.info(f"Writing infer features to {output_infer}")
    X_infer.to_parquet(output_infer, index=False)

    logger.info("Feature build complete.")


if __name__ == "__main__":
    build_features()
