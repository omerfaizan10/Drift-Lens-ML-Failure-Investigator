# Experiment Notes

## Baseline v1

A Random Forest model was selected because it performs well on mixed numeric and categorical customer data and provides robust baseline performance without extensive feature engineering.

## Validation Behavior

During validation, recall was weaker for customers with short tenure and high monthly charges. The model performed best on long-tenure customers with stable contract types.

## Feature Sensitivity

Permutation tests showed that monthly charges, tenure, support ticket count, and payment delay days often explain most of the model behavior. When these distributions change, the model should be considered at risk.

## Retraining Guidance

Retraining should use a time-based split rather than a random split when production behavior changes. The team should compare old and new cohorts and inspect false negatives before approving a model promotion.
