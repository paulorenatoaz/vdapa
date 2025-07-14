import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from vdapa.utils import setup_logging
from vdapa.config import BASE_DIR, config

logger = setup_logging('modeling', 'time_to_nvd_classification')

# Load features
path = BASE_DIR / config['paths']['processed_data'] / 'features_train.parquet'
logger.info(f"Reading training data from {path}")
df = pd.read_parquet(path)

# Define target and features
y = (df['time_to_nvd'] > 1).astype(int)
X = df.drop(columns=['time_to_nvd'])

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")

# Handle imbalance with SMOTE
logger.info("Applying SMOTE to balance classes")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
logger.info(f"After SMOTE: {pd.Series(y_res).value_counts().to_dict()}")

def evaluate(name, model, X_tr, y_tr, X_va, y_va):
    logger.info(f"Training {name}...")
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)
    probs = model.predict_proba(X_va)[:,1]
    acc = accuracy_score(y_va, preds)
    prec = precision_score(y_va, preds)
    rec = recall_score(y_va, preds)
    f1 = f1_score(y_va, preds)
    auc = roc_auc_score(y_va, probs)
    logger.info(f"--- {name} ---")
    logger.info(f"Accuracy : {acc:.3f}")
    logger.info(f"Precision: {prec:.3f}")
    logger.info(f"Recall   : {rec:.3f}")
    logger.info(f"F1       : {f1:.3f}")
    logger.info(f"ROC AUC  : {auc:.3f}")
    logger.info("Confusion Matrix:")
    logger.info(f"{confusion_matrix(y_va, preds)}")
    logger.info("Classification Report:")
    logger.info(f"{classification_report(y_va, preds)}")

# Models to test
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(random_state=42)
}

# Evaluate each
for name, mdl in models.items():
    evaluate(name, mdl, X_res, y_res, X_val, y_val)

logger.info("Classification modeling complete.")
