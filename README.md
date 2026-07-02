# DriftLens — ML Failure Investigator

DriftLens is a Streamlit-based ML failure investigation app that helps explain why a model performs differently after deployment. It supports both **classification** and **regression** datasets.

## What it does

- Upload a training CSV and production CSV
- Select the target column
- Auto-detect or manually choose classification/regression mode
- Train a baseline Random Forest model
- Compare validation vs production performance
- Detect data drift between training and production data
- Compute permutation feature importance
- Use lightweight local RAG over model notes and incident logs
- Generate an evidence-backed investigation report

## Supported problem types

### Classification

Examples:

- churn: 0/1
- fraud: yes/no
- loan_default: default/no_default
- region: North/South/East/West, if region is truly what you want to predict

Metrics:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC for binary targets when available
- Confusion matrix

### Regression

Examples:

- sales
- price
- demand
- monthly revenue
- customer_lifetime_value

Metrics:

- MAE
- RMSE
- R²
- MAPE
- Actual vs predicted plots
- Residual summary

## Built-in sample datasets

The app includes two demo modes:

1. Classification: customer churn
2. Regression: customer lifetime value

## Repository structure

```text
streamlit_app.py
requirements.txt
README.md
create_sample_data.py

src/
  __init__.py
  drift_detection.py
  explainability.py
  modeling.py
  rag_engine.py
  report_generator.py

data/
  sample_train.csv
  sample_production.csv
  sample_regression_train.csv
  sample_regression_production.csv

knowledge_base/
  model_card.md
  experiment_notes.md
  incident_logs.md
  data_validation_report.md
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud

1. Push this full project to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app.
4. Select your GitHub repository.
5. Set the main file path to:

```text
streamlit_app.py
```

6. Click Deploy.

## Notes

This MVP intentionally uses a local TF-IDF RAG engine instead of a paid API, so it can be deployed publicly without API keys.
