"""Create reproducible sample datasets for DriftLens.

Creates:
- data/sample_train.csv and data/sample_production.csv for classification/churn
- data/sample_regression_train.csv and data/sample_regression_production.csv for regression/customer value
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def make_churn_data(n: int, production: bool = False) -> pd.DataFrame:
    regions = np.array(["North", "South", "East", "West"])
    contract_types = np.array(["Month-to-month", "One year", "Two year"])
    internet_types = np.array(["Fiber", "DSL", "None"])

    if production:
        tenure = rng.gamma(shape=2.0, scale=8.0, size=n).clip(1, 72)
        monthly = rng.normal(82, 22, n).clip(20, 150)
        support = rng.poisson(3.1, n).clip(0, 12)
        region = rng.choice(regions, n, p=[0.18, 0.42, 0.20, 0.20])
        contract = rng.choice(contract_types, n, p=[0.72, 0.20, 0.08])
        internet = rng.choice(internet_types, n, p=[0.68, 0.25, 0.07])
    else:
        tenure = rng.gamma(shape=3.0, scale=10.0, size=n).clip(1, 72)
        monthly = rng.normal(65, 18, n).clip(20, 140)
        support = rng.poisson(1.7, n).clip(0, 10)
        region = rng.choice(regions, n, p=[0.26, 0.25, 0.25, 0.24])
        contract = rng.choice(contract_types, n, p=[0.48, 0.32, 0.20])
        internet = rng.choice(internet_types, n, p=[0.48, 0.38, 0.14])

    contract_risk = np.where(contract == "Month-to-month", 0.9, np.where(contract == "One year", -0.25, -0.75))
    internet_risk = np.where(internet == "Fiber", 0.35, np.where(internet == "DSL", 0.05, -0.30))
    region_risk = np.where(region == "South", 0.25, 0.0)

    logit = (
        -1.2
        - 0.035 * tenure
        + 0.018 * (monthly - 60)
        + 0.32 * support
        + contract_risk
        + internet_risk
        + region_risk
    )
    if production:
        logit += 0.45
    churn = rng.binomial(1, sigmoid(logit))

    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1) + (100_000 if production else 0),
            "tenure_months": tenure.round(1),
            "monthly_charges": monthly.round(2),
            "support_tickets": support,
            "region": region,
            "contract_type": contract,
            "internet_service": internet,
            "churn": churn,
        }
    )


def make_regression_data(n: int, production: bool = False) -> pd.DataFrame:
    regions = np.array(["North", "South", "East", "West"])
    plan_types = np.array(["Basic", "Standard", "Premium"])

    if production:
        tenure = rng.gamma(shape=2.1, scale=9.0, size=n).clip(1, 84)
        monthly = rng.normal(88, 24, n).clip(20, 170)
        support = rng.poisson(3.4, n).clip(0, 14)
        satisfaction = rng.normal(6.3, 1.8, n).clip(1, 10)
        region = rng.choice(regions, n, p=[0.17, 0.43, 0.19, 0.21])
        plan = rng.choice(plan_types, n, p=[0.35, 0.35, 0.30])
        noise = rng.normal(0, 620, n)
    else:
        tenure = rng.gamma(shape=3.2, scale=10.0, size=n).clip(1, 84)
        monthly = rng.normal(68, 17, n).clip(20, 150)
        support = rng.poisson(1.5, n).clip(0, 10)
        satisfaction = rng.normal(7.6, 1.2, n).clip(1, 10)
        region = rng.choice(regions, n, p=[0.26, 0.25, 0.25, 0.24])
        plan = rng.choice(plan_types, n, p=[0.45, 0.38, 0.17])
        noise = rng.normal(0, 420, n)

    plan_bonus = np.where(plan == "Premium", 1250, np.where(plan == "Standard", 550, 0))
    region_bonus = np.where(region == "North", 250, np.where(region == "South", -180, 0))

    # Customer lifetime value is the numeric regression target.
    clv = (
        900
        + tenure * 42
        + monthly * 31
        + satisfaction * 260
        - support * 185
        + plan_bonus
        + region_bonus
        + noise
    ).clip(100, None)

    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1) + (300_000 if production else 200_000),
            "tenure_months": tenure.round(1),
            "monthly_charges": monthly.round(2),
            "support_tickets": support,
            "satisfaction_score": satisfaction.round(1),
            "region": region,
            "plan_type": plan,
            "customer_lifetime_value": clv.round(2),
        }
    )


if __name__ == "__main__":
    make_churn_data(1400, production=False).to_csv(DATA_DIR / "sample_train.csv", index=False)
    make_churn_data(600, production=True).to_csv(DATA_DIR / "sample_production.csv", index=False)
    make_regression_data(1400, production=False).to_csv(DATA_DIR / "sample_regression_train.csv", index=False)
    make_regression_data(600, production=True).to_csv(DATA_DIR / "sample_regression_production.csv", index=False)
    print("Sample datasets created in", DATA_DIR)
