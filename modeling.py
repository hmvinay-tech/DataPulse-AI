"""Reusable preprocessing and model-training helpers for DataPulse AI."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


def infer_task_type(target: pd.Series) -> str:
    """Infer classification for categorical/boolean or low-cardinality targets."""
    if not pd.api.types.is_numeric_dtype(target) or pd.api.types.is_bool_dtype(target):
        return "Classification"
    return "Classification" if target.nunique(dropna=True) <= 15 else "Regression"


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing that is fitted only on the training data."""
    numeric = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]), categorical))
    if not transformers:
        raise ValueError("Select at least one usable feature column.")
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)


def model_candidates(task_type: str) -> dict[str, Any]:
    if task_type == "Classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=2),
        }
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost Regressor": XGBRegressor(random_state=42, n_jobs=2),
    }


def metric_values(task_type: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if task_type == "Classification":
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
    return {
        "R2 Score": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def fit_competition(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    task_type: str,
    ranking_metric: str,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Fit each candidate pipeline, returning successful results and models."""
    preprocessor = build_preprocessor(X_train)
    rows: list[dict[str, Any]] = []
    models: dict[str, Pipeline] = {}
    for name, estimator in model_candidates(task_type).items():
        model = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        model.fit(X_train, y_train)
        rows.append({"Model": name, **metric_values(task_type, y_test, model.predict(X_test))})
        models[name] = model
    if not rows:
        raise ValueError("No model could be trained on this dataset.")
    ascending = ranking_metric in {"RMSE", "MAE"}
    results = pd.DataFrame(rows).sort_values(ranking_metric, ascending=ascending).reset_index(drop=True)
    return results, models


def encode_target(target: pd.Series, task_type: str) -> tuple[np.ndarray, LabelEncoder | None]:
    if task_type == "Classification":
        encoder = LabelEncoder()
        return encoder.fit_transform(target.astype(str)), encoder
    return target.to_numpy(), None
