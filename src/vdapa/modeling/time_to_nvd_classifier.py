import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from vdapa.config import BASE_DIR, config
from vdapa.utils import setup_logging

logger = setup_logging('modeling', 'time_to_nvd_classifier')


def run():
    # 1. Load features
    df = pd.read_parquet(BASE_DIR / config['paths']['processed_data'] / 'features_train.parquet')
    logger.info(f"Number of records: {len(df)}")
    # 2. Create binary target
    df = df.dropna(subset=["time_to_nvd"])
    df['y'] = (df.time_to_nvd > 1).astype(int)
    X = df.drop(columns=["time_to_nvd", "y"])
    y = df["y"]

    # 3. Split (estratificado)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y
    )
    logger.info(f"Train: {y_train.shape}, Val: {y_val.shape}")

    # 4a. Logistic Regression
    logger.info(f"Training LogisticRegression model: {LogisticRegression()}")
    clf_lr = LogisticRegression(
        solver="liblinear", class_weight="balanced", random_state=42
    )
    clf_lr.fit(X_train, y_train)
    y_pred_lr = clf_lr.predict(X_val)
    y_proba_lr = clf_lr.predict_proba(X_val)[:,1]

    # 4b. Random Forest
    logger.info(f"Validation Accuracy: {accuracy_score(y_val, y_pred_lr)}")
    clf_rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
    )
    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_val)
    y_proba_rf = clf_rf.predict_proba(X_val)[:,1]

    # 5. Metrics
    def report(name, y_true, y_pred, y_proba):

        logger.info(f"--- {name} ---")
        logger.info(f"Accuracy : {accuracy_score(y_true, y_pred):.3f}")
        logger.info(f"Precision: {precision_score(y_true, y_pred):.3f}")
        logger.info(f"Recall   : {recall_score(y_true, y_pred):.3f}")
        logger.info(f"F1       : {f1_score(y_true, y_pred):.3f}")
        logger.info(f"ROC AUC  : {roc_auc_score(y_true, y_proba):.3f}")
        logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_true, y_pred)))
        logger.info("Classification Report:\n" + classification_report(y_true, y_pred))

    report("LogisticRegression", y_val, y_pred_lr, y_proba_lr)
    report("RandomForest",      y_val, y_pred_rf, y_proba_rf)

if __name__ == "__main__":
    run()
