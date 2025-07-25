#!/usr/bin/env python3
"""
time_to_review_infer.py

Load the latest tuned model, run it on your inference feature set,
and write out a JSON with all predictions plus timestamps.
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from vdapa.config import BASE_DIR, config
from vdapa.utils import setup_logging

logger = setup_logging("modeling", "time_to_review_infer")


def inv_signed_log1p(z):
    """Inverse of the signed log1p."""
    return np.sign(z) * (np.expm1(np.abs(z)))

def run():

    time_to_review_model_folder = Path(BASE_DIR) / config["paths"]["processed_data"] / "time_to_review_model"

    # recoer winner
    summary_base_path = time_to_review_model_folder / "train_summary.json"
    summary_base = json.loads(summary_base_path.read_text())
    winner = summary_base[0]["bestmodel"]
    model_timestamp = summary_base[0]["timestamp"]
    capcutceil = summary_base[0]["capcutceil"]


    joblib_filename = f'train_{model_timestamp}_{winner}.joblib'
    logger.info(f'Last Elected is {winner} timestamp {model_timestamp}')



    # load the model
    model_path = time_to_review_model_folder / joblib_filename
    pipe = joblib.load(model_path)
    logger.info(f"Loaded model {model_path}")



    # load inference features
    infer_path = Path(BASE_DIR) / config["paths"]["processed_data"] / "features_time_to_review_infer.parquet"
    logger.info(f"Loading inference data from {infer_path}")
    df = pd.read_parquet(infer_path)
    # ensure we have an ID column

    ids = df["ghsa_id"].astype(str).tolist()

    X_inf = df.drop(columns=["ghsa_id", "time_to_review"], errors="ignore")


    # 5) predict
    preds_trans = pipe.predict(X_inf)
    preds = inv_signed_log1p(preds_trans)
    preds = np.clip(preds, 0.0, capcutceil)

    # 6) summary
    ser = pd.Series(preds, name="time_to_review")
    logger.info(f"Inference: count={len(ser)}, mean={ser.mean():.3f}, "
                f"min={ser.min():.3f}, max={ser.max():.3f}")

    # 7) assemble JSON
    now = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp": now,
        "model_timestamp": model_timestamp,
        "prediction": [
            {"ghsa_id": gid, "time_to_review": float(t)}
            for gid, t in zip(ids, preds)
        ]
    }

    # recover json and add new entry
    infer_json_path = time_to_review_model_folder / f"prediction_{model_timestamp}.json"

    existing = []
    if infer_json_path.exists():
        logger.info(f"Loading existing predictions from {infer_json_path}")
        with infer_json_path.open("r") as f:
            existing = json.load(f)

    existing.append(out)
    existing = sorted(existing, key=lambda x: x["timestamp"], reverse=True)

    with open(infer_json_path, "w") as f:
        json.dump(existing, f, indent=2)

    logger.info(f"Wrote predictions to {infer_json_path}")


if __name__ == "__main__":
    main()
