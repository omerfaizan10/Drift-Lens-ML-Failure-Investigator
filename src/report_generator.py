"""Report generation utilities for DriftLens."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def metrics_table_markdown(metrics: dict[str, Any], problem_type: str = "classification") -> str:
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"] if problem_type == "classification" else [
        "mae",
        "rmse",
        "r2",
        "mape",
        "actual_mean",
        "predicted_mean",
        "residual_mean",
        "residual_std",
    ]
    rows = [f"| {key} | {_fmt(metrics.get(key))} |" for key in keys]
    return "\n".join(["| Metric | Value |", "|---|---:|"] + rows)


def build_markdown_report(
    validation_metrics: dict[str, Any],
    production_metrics: dict[str, Any] | None,
    metric_delta_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    investigation_answer: str,
    problem_type: str = "classification",
) -> str:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    title = "Classification" if problem_type == "classification" else "Regression"
    lines = [
        "# DriftLens ML Failure Investigation Report",
        "",
        f"Generated: {timestamp}",
        f"Problem Type: {title}",
        "",
        "## Validation Metrics",
        metrics_table_markdown(validation_metrics, problem_type),
        "",
    ]

    if production_metrics:
        lines.extend(["## Production Metrics", metrics_table_markdown(production_metrics, problem_type), ""])

    if not metric_delta_df.empty:
        lines.extend(["## Metric Deltas", metric_delta_df.to_markdown(index=False), ""])

    if not drift_df.empty:
        cols = [
            "feature",
            "type",
            "drift_detected",
            "severity_score",
            "p_value",
            "psi",
            "train_summary",
            "production_summary",
        ]
        available_cols = [col for col in cols if col in drift_df.columns]
        lines.extend(["## Drift Results", drift_df[available_cols].head(12).to_markdown(index=False), ""])

    if not importance_df.empty:
        lines.extend(["## Feature Importance", importance_df.head(10).to_markdown(index=False), ""])

    lines.extend(["## AI Investigator", investigation_answer, ""])
    return "\n".join(lines)
