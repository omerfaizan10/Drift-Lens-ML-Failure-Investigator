# Model Card: Customer Churn Classifier

The customer churn classifier predicts whether a customer is likely to leave the service in the next billing period. The current baseline model is a Random Forest classifier trained on customer profile, billing, contract, and support interaction features.

## Intended Use

The model is intended to support retention prioritization. It should not be used as the only reason to deny service, change pricing, or make sensitive decisions about customers.

## Important Features

Historical reviews show that churn predictions are often sensitive to tenure, monthly charges, support tickets, payment delay days, contract type, and internet service type. These features should be monitored carefully after deployment.

## Known Risk Areas

The model can degrade when recent customers behave differently from older training cohorts. Month-to-month customers and customers with high support ticket volume are more likely to be unstable segments. Changes in plan mix, billing policy, or support logging can create data drift.

## Recommended Monitoring

Monitor F1, recall, precision, predicted positive rate, data drift, missing values, and segment-level performance weekly. Raise an incident if F1 drops by more than five percentage points or if important features show strong drift.
