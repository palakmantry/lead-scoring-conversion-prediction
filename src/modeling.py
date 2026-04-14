from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE


def build_preprocessor(X: pd.DataFrame):
    """Create sklearn preprocessing for numeric + categorical columns."""
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols


def build_baseline_pipeline(X: pd.DataFrame) -> Pipeline:
    """Baseline logistic regression model."""
    preprocessor, _, _ = build_preprocessor(X)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return pipeline


def build_random_forest_pipeline(X: pd.DataFrame, params: dict | None = None) -> Pipeline:
    """Improved tree-based model."""
    preprocessor, _, _ = build_preprocessor(X)

    default_params = {
        "n_estimators": 400,
        "max_depth": None,
        "min_samples_leaf": 3,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    if params:
        default_params.update(params)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(**default_params)),
        ]
    )
    return pipeline


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    """
    Tune the improved model using chronological CV.
    """
    pipeline = build_random_forest_pipeline(X_train)

    param_distributions = {
        "model__n_estimators": [200, 300, 400, 500],
        "model__max_depth": [None, 8, 12, 16],
        "model__min_samples_leaf": [1, 3, 5, 10],
        "model__max_features": ["sqrt", 0.5, None],
    }

    cv = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=12,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    search.fit(X_train, y_train)
    return search


def _top_share_precision(y_true: np.ndarray, y_score: np.ndarray, share: float = 0.10) -> tuple[float, float]:
    """Precision and lift among the top X% highest-scoring rows."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    n = max(1, int(np.ceil(len(y_score) * share)))
    order = np.argsort(-y_score)
    selected = order[:n]

    precision = float(y_true[selected].mean()) if n > 0 else 0.0
    baseline = float(y_true.mean()) if len(y_true) > 0 else 0.0
    lift = float(precision / baseline) if baseline > 0 else 0.0
    return precision, lift


def classification_metrics(y_true, y_score) -> dict:
    """Core model metrics for ranking + probability quality."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    precision_at_10pct, lift_at_10pct = _top_share_precision(y_true, y_score, share=0.10)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "log_loss": float(log_loss(y_true, y_score, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "precision_at_10pct": float(precision_at_10pct),
        "lift_at_10pct": float(lift_at_10pct),
        "base_conversion_rate": float(np.mean(y_true)),
    }
    return metrics


def _to_python(obj):
    """Convert numpy / pandas types into JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_python(v) for v in obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return obj


def save_json(payload: dict, path: Path) -> None:
    """Save a dict to JSON with nice formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(_to_python(payload), file, indent=2)