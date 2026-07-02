"""Explainability utilities for DriftLens."""
from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def permutation_feature_importance(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str = "classification",
    max_rows: int = 500,
) -> pd.DataFrame:
    """Compute model-agnostic permutation importance over original input columns.

    Classification uses weighted F1. Regression uses negative MAE, so higher importance
    means the feature matters more to prediction quality.
    """
    if len(X) == 0:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std", "scoring"])

    sample_X = X.sample(min(max_rows, len(X)), random_state=42)
    sample_y = y.loc[sample_X.index]
    scoring = "neg_mean_absolute_error" if problem_type == "regression" else "f1_weighted"

    result = permutation_importance(
        model,
        sample_X,
        sample_y,
        n_repeats=5,
        random_state=42,
        scoring=scoring,
        n_jobs=1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "scoring": scoring,
        }
    )
    return importance_df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
