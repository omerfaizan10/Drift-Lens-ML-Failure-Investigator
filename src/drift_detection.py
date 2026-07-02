"""Data drift detection utilities for DriftLens."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Compute PSI using quantile bins derived from the expected/training data."""
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(expected, quantiles))
    if len(breakpoints) <= 2:
        breakpoints = np.linspace(expected.min(), expected.max(), bins + 1)
        breakpoints = np.unique(breakpoints)
    if len(breakpoints) <= 2:
        return 0.0

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_perc = np.maximum(expected_counts / max(expected_counts.sum(), 1), 1e-6)
    actual_perc = np.maximum(actual_counts / max(actual_counts.sum(), 1), 1e-6)
    return float(np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc)))


def categorical_distribution_shift(train: pd.Series, prod: pd.Series) -> tuple[float, float]:
    train_counts = train.fillna("__MISSING__").astype(str).value_counts()
    prod_counts = prod.fillna("__MISSING__").astype(str).value_counts()
    categories = sorted(set(train_counts.index).union(set(prod_counts.index)))
    table = np.array(
        [
            [train_counts.get(cat, 0) for cat in categories],
            [prod_counts.get(cat, 0) for cat in categories],
        ]
    )
    if table.shape[1] < 2 or table.sum() == 0:
        return np.nan, np.nan
    try:
        chi2, p_value, _, _ = chi2_contingency(table)
    except ValueError:
        return np.nan, np.nan
    return float(chi2), float(p_value)


def detect_drift(train_df: pd.DataFrame, prod_df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
    train = train_df.copy()
    prod = prod_df.copy()
    if target_col:
        train = train.drop(columns=[target_col], errors="ignore")
        prod = prod.drop(columns=[target_col], errors="ignore")
    id_like = [col for col in train.columns if col.lower().endswith("id") or col.lower() in {"id", "customer_id"}]
    train = train.drop(columns=id_like, errors="ignore")
    prod = prod.drop(columns=id_like, errors="ignore")

    common_columns = [col for col in train.columns if col in prod.columns]
    rows = []

    for col in common_columns:
        train_s = train[col]
        prod_s = prod[col]
        train_missing = float(train_s.isna().mean())
        prod_missing = float(prod_s.isna().mean())
        missing_delta = prod_missing - train_missing

        if pd.api.types.is_numeric_dtype(train_s) and pd.api.types.is_numeric_dtype(prod_s):
            clean_train = pd.to_numeric(train_s, errors="coerce").dropna()
            clean_prod = pd.to_numeric(prod_s, errors="coerce").dropna()
            if len(clean_train) > 0 and len(clean_prod) > 0:
                ks_stat, p_value = ks_2samp(clean_train, clean_prod)
                psi = population_stability_index(clean_train, clean_prod)
                train_summary = f"mean={clean_train.mean():.2f}, std={clean_train.std():.2f}"
                prod_summary = f"mean={clean_prod.mean():.2f}, std={clean_prod.std():.2f}"
            else:
                ks_stat, p_value, psi = np.nan, np.nan, np.nan
                train_summary, prod_summary = "not enough data", "not enough data"

            drift_flag = bool(
                (not np.isnan(p_value) and p_value < 0.01 and not np.isnan(ks_stat) and ks_stat > 0.10)
                or (not np.isnan(psi) and psi >= 0.20)
                or abs(missing_delta) >= 0.05
            )
            severity_score = max(
                0 if np.isnan(ks_stat) else ks_stat,
                0 if np.isnan(psi) else min(psi, 1.0),
                abs(missing_delta),
            )
            rows.append(
                {
                    "feature": col,
                    "type": "numeric",
                    "drift_detected": drift_flag,
                    "severity_score": severity_score,
                    "statistic": ks_stat,
                    "p_value": p_value,
                    "psi": psi,
                    "train_missing_rate": train_missing,
                    "production_missing_rate": prod_missing,
                    "missing_delta": missing_delta,
                    "train_summary": train_summary,
                    "production_summary": prod_summary,
                }
            )
        else:
            chi2, p_value = categorical_distribution_shift(train_s, prod_s)
            train_mode = train_s.fillna("__MISSING__").astype(str).mode()
            prod_mode = prod_s.fillna("__MISSING__").astype(str).mode()
            train_summary = f"top={train_mode.iloc[0] if len(train_mode) else 'n/a'}"
            prod_summary = f"top={prod_mode.iloc[0] if len(prod_mode) else 'n/a'}"
            severity_score = 0 if np.isnan(chi2) else min(chi2 / max(len(train_s) + len(prod_s), 1), 1.0)
            drift_flag = bool((not np.isnan(p_value) and p_value < 0.01 and severity_score > 0.02) or abs(missing_delta) >= 0.05)
            rows.append(
                {
                    "feature": col,
                    "type": "categorical",
                    "drift_detected": drift_flag,
                    "severity_score": severity_score,
                    "statistic": chi2,
                    "p_value": p_value,
                    "psi": np.nan,
                    "train_missing_rate": train_missing,
                    "production_missing_rate": prod_missing,
                    "missing_delta": missing_delta,
                    "train_summary": train_summary,
                    "production_summary": prod_summary,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["drift_detected", "severity_score"], ascending=[False, False]).reset_index(drop=True)
