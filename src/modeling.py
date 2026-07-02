"""Model training and evaluation utilities for DriftLens."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ProblemType = Literal["classification", "regression"]


@dataclass
class ModelBundle:
    pipeline: Pipeline
    feature_columns: list[str]
    target_column: str
    problem_type: ProblemType
    validation_X: pd.DataFrame
    validation_y: pd.Series


def split_features_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    # Drop obvious identifier columns from modeling while keeping them available in the raw data.
    id_like = [col for col in X.columns if col.lower().endswith("id") or col.lower() in {"id", "customer_id"}]
    X = X.drop(columns=id_like, errors="ignore")
    return X, y


def infer_problem_type(y: pd.Series) -> tuple[str, str]:
    """Infer whether a target is classification or regression.

    Heuristic:
    - object/category/bool targets are classification
    - numeric targets with <= 20 unique values are classification
    - numeric targets with > 20 unique values are regression
    """
    y_clean = y.dropna()
    unique_count = int(y_clean.nunique())

    if unique_count < 2:
        return "invalid", "The selected target has fewer than 2 unique values."

    if pd.api.types.is_bool_dtype(y_clean) or pd.api.types.is_object_dtype(y_clean) or pd.api.types.is_categorical_dtype(y_clean):
        return "classification", f"Classification target detected with {unique_count} class(es)."

    if pd.api.types.is_numeric_dtype(y_clean):
        if unique_count <= 20:
            return "classification", f"Classification target detected with {unique_count} class(es)."
        return "regression", "Regression target detected because the target is numeric with many unique values."

    return "classification", f"Classification target detected with {unique_count} class(es)."


def validate_target_for_problem(y: pd.Series, problem_type: ProblemType) -> None:
    y_clean = y.dropna()
    if y_clean.nunique() < 2:
        raise ValueError("Target column must have at least 2 unique non-empty values.")
    if problem_type == "regression":
        numeric_y = pd.to_numeric(y_clean, errors="coerce")
        if numeric_y.isna().any():
            raise ValueError("Regression targets must be numeric. Choose a numeric target or use classification.")


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    return numeric_cols, categorical_cols


def build_pipeline(X: pd.DataFrame, problem_type: ProblemType) -> Pipeline:
    numeric_cols, categorical_cols = infer_feature_types(X)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    if problem_type == "classification":
        model = RandomForestClassifier(
            n_estimators=140,
            max_depth=8,
            min_samples_leaf=8,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=1,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=160,
            max_depth=10,
            min_samples_leaf=6,
            random_state=42,
            n_jobs=1,
        )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def _coerce_regression_target(y: pd.Series) -> pd.Series:
    return pd.to_numeric(y, errors="coerce")


def train_model(df: pd.DataFrame, target_col: str, problem_type: ProblemType | str = "auto") -> ModelBundle:
    X, y = split_features_target(df, target_col)

    if problem_type == "auto":
        inferred, _ = infer_problem_type(y)
        if inferred not in {"classification", "regression"}:
            raise ValueError("Could not infer a valid problem type for the selected target.")
        problem_type = inferred

    problem_type = problem_type  # for type checkers
    if problem_type not in {"classification", "regression"}:
        raise ValueError("Problem type must be 'classification' or 'regression'.")

    validate_target_for_problem(y, problem_type)  # type: ignore[arg-type]
    if problem_type == "regression":
        y = _coerce_regression_target(y)

    stratify = None
    if problem_type == "classification":
        class_counts = pd.Series(y).value_counts(dropna=True)
        stratify = y if len(class_counts) >= 2 and int(class_counts.min()) >= 2 else None

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )
    pipeline = build_pipeline(X_train, problem_type)  # type: ignore[arg-type]
    pipeline.fit(X_train, y_train)
    return ModelBundle(
        pipeline=pipeline,
        feature_columns=X.columns.tolist(),
        target_column=target_col,
        problem_type=problem_type,  # type: ignore[arg-type]
        validation_X=X_valid,
        validation_y=y_valid,
    )


def _positive_class_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    try:
        proba = model.predict_proba(X)
    except Exception:
        return None
    if proba.ndim != 2 or proba.shape[1] < 2:
        return None
    return proba[:, 1]


def evaluate_classification_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    y_pred = model.predict(X)

    y_series = pd.Series(y).reset_index(drop=True)
    y_pred_series = pd.Series(y_pred).reset_index(drop=True)
    labels = sorted(pd.concat([y_series, y_pred_series]).dropna().unique().tolist(), key=lambda x: str(x))

    result: dict[str, Any] = {
        "problem_type": "classification",
        "accuracy": accuracy_score(y_series, y_pred_series),
        "precision": precision_score(y_series, y_pred_series, average="weighted", zero_division=0),
        "recall": recall_score(y_series, y_pred_series, average="weighted", zero_division=0),
        "f1": f1_score(y_series, y_pred_series, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_series, y_pred_series, labels=labels).tolist(),
        "class_labels": [str(label) for label in labels],
        "support": int(len(y_series)),
    }

    if y_series.nunique(dropna=True) == 2:
        try:
            result["positive_rate_actual"] = float(pd.to_numeric(y_series, errors="raise").mean())
            result["positive_rate_predicted"] = float(pd.to_numeric(y_pred_series, errors="raise").mean())
        except Exception:
            result["positive_rate_actual"] = None
            result["positive_rate_predicted"] = None
    else:
        result["positive_rate_actual"] = None
        result["positive_rate_predicted"] = None

    proba = _positive_class_probability(model, X)
    if proba is not None and y_series.nunique(dropna=True) == 2:
        try:
            result["roc_auc"] = roc_auc_score(y_series, proba)
        except Exception:
            result["roc_auc"] = None
    else:
        result["roc_auc"] = None
    return result


def evaluate_regression_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    y_true = _coerce_regression_target(pd.Series(y)).reset_index(drop=True)
    y_pred = pd.Series(model.predict(X)).reset_index(drop=True)

    valid_mask = y_true.notna() & y_pred.notna()
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    if len(y_true) == 0:
        raise ValueError("No valid numeric target values available for regression evaluation.")

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    non_zero_mask = y_true.abs() > 1e-9
    mape = float((np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])).mean()) if non_zero_mask.any() else None
    residuals = y_true - y_pred

    return {
        "problem_type": "regression",
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": None if pd.isna(r2) else float(r2),
        "mape": mape,
        "actual_mean": float(y_true.mean()),
        "predicted_mean": float(y_pred.mean()),
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std()),
        "support": int(len(y_true)),
    }


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series, problem_type: ProblemType | str = "classification") -> dict[str, Any]:
    if problem_type == "regression":
        return evaluate_regression_model(model, X, y)
    return evaluate_classification_model(model, X, y)


def prepare_production_features(prod_df: pd.DataFrame, target_col: str, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series | None]:
    y = prod_df[target_col] if target_col in prod_df.columns else None
    X = prod_df.drop(columns=[target_col], errors="ignore")
    id_like = [col for col in X.columns if col.lower().endswith("id") or col.lower() in {"id", "customer_id"}]
    X = X.drop(columns=id_like, errors="ignore")

    # Keep only expected features and create missing expected columns if needed.
    for col in feature_columns:
        if col not in X.columns:
            X[col] = np.nan
    X = X[feature_columns]
    return X, y


def summarize_metric_delta(
    validation_metrics: dict[str, Any],
    production_metrics: dict[str, Any] | None,
    problem_type: ProblemType | str = "classification",
) -> pd.DataFrame:
    if problem_type == "regression":
        metrics = ["mae", "rmse", "r2", "mape", "actual_mean", "predicted_mean"]
    else:
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    rows = []
    for metric in metrics:
        valid_value = validation_metrics.get(metric)
        prod_value = production_metrics.get(metric) if production_metrics else None
        delta = None if valid_value is None or prod_value is None else prod_value - valid_value
        rows.append(
            {
                "metric": metric,
                "validation": valid_value,
                "production": prod_value,
                "delta": delta,
            }
        )
    return pd.DataFrame(rows)
