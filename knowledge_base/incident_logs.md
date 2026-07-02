# Historical Incident Logs

## Incident 2026-04: Recall Drop

A previous churn model experienced a recall drop after a promotion campaign changed the customer mix. The production dataset contained more month-to-month customers and more high-charge fiber customers than the training dataset. The fix was to retrain on post-campaign data and tune the classification threshold.

## Incident 2026-05: Missing Payment Data

A payment ingestion issue increased missing values in payment delay days. The model imputed the missing values, but performance dropped because missingness was not random. The fix was to repair the ingestion job and add a missingness monitor.

## Incident 2026-06: Support Ticket Logging Change

A support platform update changed how support tickets were counted. The production support ticket distribution shifted upward, causing the churn classifier to overpredict churn for some customers. The fix was to backfill standardized support ticket counts.
