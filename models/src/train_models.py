"""
Training pipeline for CashYar AI.

Trains three models:
1. Monthly spending prediction (regression)
2. Overspending prediction (binary classification)
3. Monthly savings prediction (regression)

Artifacts saved:
- models/spending_model.pkl
- models/overspend_model.pkl
- models/savings_model.pkl
- models/training_artifacts.pkl
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split

from feature_engineering import (
    build_training_dataset,
    get_feature_columns,
    load_raw_data,
    make_model_matrix,
)
from utils import (
    DEFAULT_RANDOM_STATE,
    OVERSPEND_MODEL_FILE,
    SAVINGS_MODEL_FILE,
    SPENDING_MODEL_FILE,
    TRAINING_ARTIFACTS_FILE,
    ensure_directories,
    logger,
    save_joblib,
)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root mean squared error helper.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_regression_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_all: pd.DataFrame,
    y_all: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate a regression model using holdout and light cross-validation metrics.
    """
    preds = model.predict(X_test)

    # 3-fold CV for speed
    cv = KFold(n_splits=3, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    cv_r2 = cross_val_score(model, X_all, y_all, cv=cv, scoring="r2", n_jobs=1)
    cv_mae = -cross_val_score(model, X_all, y_all, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)
    cv_rmse = np.sqrt(
        -cross_val_score(model, X_all, y_all, cv=cv, scoring="neg_mean_squared_error", n_jobs=1)
    )

    return {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(rmse(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "cv_r2_mean": float(np.mean(cv_r2)),
        "cv_r2_std": float(np.std(cv_r2)),
        "cv_mae_mean": float(np.mean(cv_mae)),
        "cv_rmse_mean": float(np.mean(cv_rmse)),
    }


def evaluate_classification_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_all: pd.DataFrame,
    y_all: pd.Series,
) -> Dict[str, object]:
    """
    Evaluate a classification model using holdout and light cross-validation metrics.
    """
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    cv_auc = cross_val_score(model, X_all, y_all, cv=cv, scoring="roc_auc", n_jobs=1)
    cv_acc = cross_val_score(model, X_all, y_all, cv=cv, scoring="accuracy", n_jobs=1)
    cm = confusion_matrix(y_test, preds)

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 0.5,
        "confusion_matrix": cm.tolist(),
        "cv_accuracy_mean": float(np.mean(cv_acc)),
        "cv_roc_auc_mean": float(np.mean(cv_auc)),
        "cv_roc_auc_std": float(np.std(cv_auc)),
    }


def train_all_models() -> Dict[str, object]:
    """
    Full training pipeline (fast baseline version).

    - Builds modeling dataset
    - Trains three Random Forest models with fixed hyperparameters
    - Evaluates each model
    - Saves models and training artifacts
    """
    ensure_directories()

    # Build dataset
    bundle = load_raw_data()
    dataset = build_training_dataset(bundle)
    feature_columns = get_feature_columns(dataset)
    X, model_feature_names = make_model_matrix(dataset, feature_columns=feature_columns)

    # Targets
    y_spending = dataset["target_monthly_spending"]
    y_savings = dataset["target_monthly_savings"]
    y_overspend = dataset["target_overspend"]

    # Train/test splits
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y_spending, test_size=0.2, random_state=DEFAULT_RANDOM_STATE
    )

    X_train_sv, X_test_sv, y_train_sv, y_test_sv = train_test_split(
        X, y_savings, test_size=0.2, random_state=DEFAULT_RANDOM_STATE
    )

    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
        X, y_overspend, test_size=0.2, random_state=DEFAULT_RANDOM_STATE, stratify=y_overspend
    )

    # Baseline models (no tuning, n_jobs=1 to avoid nested parallelism)
    spending_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=DEFAULT_RANDOM_STATE,
        n_jobs=1,
    )
    spending_model.fit(X_train_s, y_train_s)

    savings_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=DEFAULT_RANDOM_STATE,
        n_jobs=1,
    )
    savings_model.fit(X_train_sv, y_train_sv)

    overspend_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=DEFAULT_RANDOM_STATE,
        n_jobs=1,
    )
    overspend_model.fit(X_train_o, y_train_o)

    # Evaluate models
    spending_metrics = evaluate_regression_model(spending_model, X_test_s, y_test_s, X, y_spending)
    savings_metrics = evaluate_regression_model(savings_model, X_test_sv, y_test_sv, X, y_savings)
    overspend_metrics = evaluate_classification_model(overspend_model, X_test_o, y_test_o, X, y_overspend)

    # Save models
    save_joblib(spending_model, SPENDING_MODEL_FILE)
    save_joblib(savings_model, SAVINGS_MODEL_FILE)
    save_joblib(overspend_model, OVERSPEND_MODEL_FILE)

    # Save artifacts
    training_artifacts = {
        "model_feature_names": model_feature_names,
        "base_feature_columns": feature_columns,
        "training_metrics": {
            "spending_model": spending_metrics,
            "savings_model": savings_metrics,
            "overspend_model": overspend_metrics,
        },
    }
    save_joblib(training_artifacts, TRAINING_ARTIFACTS_FILE)

    logger.info("Training complete.")
    logger.info("Spending metrics: %s", spending_metrics)
    logger.info("Savings metrics: %s", savings_metrics)
    logger.info("Overspend metrics: %s", overspend_metrics)

    return training_artifacts


if __name__ == "__main__":
    train_all_models()