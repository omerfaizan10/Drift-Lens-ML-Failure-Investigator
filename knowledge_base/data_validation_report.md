# Data Validation Rules

## Schema Requirements

The production scoring dataset should contain the same modeling columns as training data. Extra identifier columns are allowed but should not be used as predictive features.

## Numeric Feature Checks

Numeric columns should be monitored for mean shift, standard deviation shift, missing-value changes, outlier rates, Kolmogorov-Smirnov test results, and Population Stability Index.

## Categorical Feature Checks

Categorical columns should be monitored for new categories, changed category proportions, missing-value changes, and chi-square distribution shift.

## Severity Guidance

A PSI value below 0.10 is usually minor. A PSI value from 0.10 to 0.20 should be reviewed. A PSI value above 0.20 is a strong signal that the feature distribution changed enough to threaten model reliability.

## Failure Investigation Checklist

When model performance drops, compare validation and production metrics, identify drifted high-importance features, check missing values, review false negatives, inspect recent data pipeline changes, and decide whether threshold tuning or retraining is required.
