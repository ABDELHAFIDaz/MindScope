import json
import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from mindscope.config import (
    BEST_PARAMS,
    LABELS,
    METRICS_PATH,
    MODEL_NAME,
    MODEL_PATH,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    PROCESSED_DIR,
    RAW_DATA_PATH,
)
from mindscope.preprocessing import TARGET, build_pipeline, build_preprocessor, load_data, split_data

log = logging.getLogger(__name__)


def _check_file(path, hint):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}. {hint}")


def _load_split(name):
    x_path = PROCESSED_DIR / f"X_{name}.csv"
    y_path = PROCESSED_DIR / f"y_{name}.csv"
    hint = "Run the prepare_data task first."
    _check_file(x_path, hint)
    _check_file(y_path, hint)
    return pd.read_csv(x_path), pd.read_csv(y_path)[TARGET]


def prepare_data():
    _check_file(RAW_DATA_PATH, "Put the dataset in data/raw/.")
    if TARGET not in pd.read_csv(RAW_DATA_PATH, nrows=0).columns:
        raise ValueError(f"Target column '{TARGET}' not found in {RAW_DATA_PATH.name}.")

    df = load_data(RAW_DATA_PATH)
    log.info("Clean data: %s rows, %s columns", *df.shape)

    X_train, X_test, y_train, y_test = split_data(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    log.info("Split saved: %s train rows, %s test rows", len(X_train), len(X_test))


def preprocess():
    X_train, _ = _load_split("train")

    preprocessor = build_preprocessor(X_train)
    X_prepared = preprocessor.fit_transform(X_train)
    if np.isnan(X_prepared).any():
        raise ValueError("Missing values remain after preprocessing.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    log.info("Preprocessing OK: %s rows, %s features", *X_prepared.shape)


def train_model():
    X_train, y_train = _load_split("train")

    model = build_pipeline(LogisticRegression(**BEST_PARAMS), X_train)
    model.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    log.info("%s trained on %s students and saved to %s", MODEL_NAME, len(X_train), MODEL_PATH.name)


def evaluate_model():
    _, y_train = _load_split("train")
    X_test, y_test = _load_split("test")
    _check_file(MODEL_PATH, "Run the train_model task first.")

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    report = classification_report(y_test, y_pred, labels=LABELS, output_dict=True, zero_division=0)

    feature_names = [name.split("__", 1)[1] for name in model.named_steps["preprocessor"].get_feature_names_out()]
    coefficients = np.abs(model.named_steps["model"].coef_).mean(axis=0)
    importance = pd.Series(coefficients, index=feature_names).sort_values(ascending=False).head(10)

    results = {
        "model_name": MODEL_NAME,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "labels": LABELS,
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="macro")),
            "recall": float(recall_score(y_test, y_pred, average="macro")),
            "f1_score": float(f1_score(y_test, y_pred, average="macro")),
            "roc_auc": float(roc_auc_score(y_test, y_proba, multi_class="ovr", labels=model.classes_)),
        },
        "per_class": {
            label: {key: float(report[label][key]) for key in ("precision", "recall", "f1-score", "support")}
            for label in LABELS
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=LABELS).tolist(),
        "feature_importance": {name: round(float(value), 4) for name, value in importance.items()},
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2))
    log.info(
        "Test results: accuracy %.3f | F1-score %.3f | ROC-AUC %.3f",
        results["metrics"]["accuracy"],
        results["metrics"]["f1_score"],
        results["metrics"]["roc_auc"],
    )