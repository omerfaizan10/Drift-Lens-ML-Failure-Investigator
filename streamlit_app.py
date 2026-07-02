from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR / "src"))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.drift_detection import detect_drift
from src.explainability import permutation_feature_importance
from src.modeling import (
    evaluate_model,
    infer_problem_type,
    prepare_production_features,
    summarize_metric_delta,
    train_model,
)
from src.rag_engine import LocalRAG, synthesize_investigation_answer
from src.report_generator import build_markdown_report

DATA_DIR = APP_DIR / "data"

st.set_page_config(
    page_title="DriftLens | ML Failure Investigator",
    page_icon="🔎",
    layout="wide",
)


def load_csv(uploaded_file, fallback_path: Path) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(fallback_path)


def metric_card(label: str, value: Any, delta: Any | None = None, inverse_delta: bool = False) -> None:
    if value is None or pd.isna(value):
        st.metric(label, "n/a")
        return
    delta_text = None
    if delta is not None and not pd.isna(delta):
        # For error metrics such as MAE/RMSE, lower is better. Flip only the visual arrow direction.
        visual_delta = -delta if inverse_delta else delta
        delta_text = f"{visual_delta:+.3f}"
    st.metric(label, f"{float(value):.3f}", delta_text)


def plot_confusion_matrix(cm: list[list[int]], title: str, labels: list[str] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    x_count = len(cm[0]) if cm else 0
    y_count = len(cm)
    ax.set_xticks(range(x_count))
    ax.set_yticks(range(y_count))
    if labels and len(labels) == x_count:
        ax.set_xticklabels(labels, rotation=45, ha="right")
    if labels and len(labels) == y_count:
        ax.set_yticklabels(labels)

    for i in range(y_count):
        for j in range(len(cm[i])):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    fig.tight_layout()
    st.pyplot(fig)


def plot_regression_predictions(model, X: pd.DataFrame, y: pd.Series, title: str) -> None:
    y_true = pd.to_numeric(y, errors="coerce")
    y_pred = pd.Series(model.predict(X), index=X.index)
    valid = y_true.notna() & y_pred.notna()
    if valid.sum() == 0:
        st.info("No valid numeric target values available for this plot.")
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.scatter(y_true[valid], y_pred[valid], alpha=0.65)
    min_value = min(y_true[valid].min(), y_pred[valid].min())
    max_value = max(y_true[valid].max(), y_pred[valid].max())
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    st.pyplot(fig)


def plot_numeric_distribution(train_df: pd.DataFrame, prod_df: pd.DataFrame, feature: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    train_values = pd.to_numeric(train_df[feature], errors="coerce").dropna()
    prod_values = pd.to_numeric(prod_df[feature], errors="coerce").dropna()
    ax.hist(train_values, bins=25, alpha=0.6, label="Training")
    ax.hist(prod_values, bins=25, alpha=0.6, label="Production")
    ax.set_title(f"Distribution comparison: {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)


def default_target_for_columns(columns: list[str], sample_kind: str) -> str:
    preferred = "churn" if sample_kind == "classification" else "customer_lifetime_value"
    if preferred in columns:
        return preferred
    for fallback in ["target", "label", "y", "outcome"]:
        if fallback in columns:
            return fallback
    return columns[0]


def run_investigation(train_df: pd.DataFrame, prod_df: pd.DataFrame, target_col: str, problem_type: str) -> dict[str, Any]:
    bundle = train_model(train_df, target_col, problem_type=problem_type)
    validation_metrics = evaluate_model(bundle.pipeline, bundle.validation_X, bundle.validation_y, bundle.problem_type)
    prod_X, prod_y = prepare_production_features(prod_df, target_col, bundle.feature_columns)
    production_metrics = evaluate_model(bundle.pipeline, prod_X, prod_y, bundle.problem_type) if prod_y is not None else None
    metric_delta_df = summarize_metric_delta(validation_metrics, production_metrics, bundle.problem_type)
    drift_df = detect_drift(train_df, prod_df, target_col=target_col)
    importance_df = permutation_feature_importance(bundle.pipeline, bundle.validation_X, bundle.validation_y, bundle.problem_type)
    return {
        "bundle": bundle,
        "prod_X": prod_X,
        "prod_y": prod_y,
        "validation_metrics": validation_metrics,
        "production_metrics": production_metrics,
        "metric_delta_df": metric_delta_df,
        "drift_df": drift_df,
        "importance_df": importance_df,
        "problem_type": bundle.problem_type,
    }


st.title("🔎 DriftLens — ML Failure Investigator")
st.caption("A Streamlit MVP for classification and regression failure analysis using drift detection, model evaluation, explainability, and local RAG.")

with st.sidebar:
    st.header("Input Data")
    use_sample = st.toggle("Use built-in sample data", value=True)
    sample_kind = "classification"
    train_upload = None
    prod_upload = None

    if use_sample:
        sample_label = st.selectbox(
            "Sample dataset",
            ["Classification: customer churn", "Regression: customer lifetime value"],
        )
        sample_kind = "regression" if sample_label.startswith("Regression") else "classification"
    else:
        train_upload = st.file_uploader("Upload training CSV", type=["csv"])
        prod_upload = st.file_uploader("Upload production CSV", type=["csv"])
        st.info("Your production CSV may include the target column if you want production performance metrics.")

    st.divider()
    st.markdown("**Recommended first run:** use a built-in sample and click **Run investigation**.")

try:
    if use_sample:
        if sample_kind == "classification":
            train_df = pd.read_csv(DATA_DIR / "sample_train.csv")
            prod_df = pd.read_csv(DATA_DIR / "sample_production.csv")
        else:
            train_df = pd.read_csv(DATA_DIR / "sample_regression_train.csv")
            prod_df = pd.read_csv(DATA_DIR / "sample_regression_production.csv")
    elif train_upload and prod_upload:
        train_df = load_csv(train_upload, DATA_DIR / "sample_train.csv")
        prod_df = load_csv(prod_upload, DATA_DIR / "sample_production.csv")
    else:
        st.warning("Upload both CSV files or turn on the sample data.")
        st.stop()
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

st.subheader("1. Data Preview")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Training data**")
    st.dataframe(train_df.head(8), use_container_width=True)
    st.caption(f"Rows: {train_df.shape[0]:,} | Columns: {train_df.shape[1]:,}")
with col_b:
    st.markdown("**Production data**")
    st.dataframe(prod_df.head(8), use_container_width=True)
    st.caption(f"Rows: {prod_df.shape[0]:,} | Columns: {prod_df.shape[1]:,}")

candidate_targets = [
    col for col in train_df.columns
    if not (col.lower().endswith("id") or col.lower() in {"id", "customer_id"})
]
if not candidate_targets:
    st.error("No target candidates found. Your CSV needs at least one non-ID column.")
    st.stop()

default_target = default_target_for_columns(candidate_targets, sample_kind)
target_col = st.selectbox(
    "Select target column",
    candidate_targets,
    index=candidate_targets.index(default_target),
    help="Choose the outcome/label/value the model should predict, such as churn, fraud, default, failure, price, sales, or customer_lifetime_value.",
)

inferred_problem, target_message = infer_problem_type(train_df[target_col])
problem_choice = st.radio(
    "Problem type",
    ["Auto detect", "Classification", "Regression"],
    horizontal=True,
    help="Use Auto detect for most datasets. Override only if you know the selected target type.",
)

if problem_choice == "Auto detect":
    if inferred_problem not in {"classification", "regression"}:
        st.error(target_message)
        st.stop()
    problem_type = inferred_problem
else:
    problem_type = problem_choice.lower()

if problem_type == "classification":
    unique_count = int(train_df[target_col].dropna().nunique())
    if unique_count < 2:
        st.error("Classification targets need at least 2 unique classes.")
        st.stop()
    if unique_count > 50:
        st.warning(f"This target has {unique_count} classes. The app can run, but the results may be hard to interpret.")
    st.success(f"Classification mode selected. {target_message if problem_choice == 'Auto detect' else ''}")
else:
    numeric_target = pd.to_numeric(train_df[target_col], errors="coerce")
    if numeric_target.isna().any():
        st.error("Regression mode requires a numeric target column. Choose a numeric target or switch to classification.")
        st.stop()
    if numeric_target.nunique(dropna=True) < 2:
        st.error("Regression targets need at least 2 unique numeric values.")
        st.stop()
    st.success(f"Regression mode selected. {target_message if problem_choice == 'Auto detect' else ''}")

st.info(f"Selected target: `{target_col}` | Problem type: `{problem_type}`")

if st.button("Run investigation", type="primary"):
    with st.spinner("Training model, detecting drift, and preparing investigation evidence..."):
        st.session_state["results"] = run_investigation(train_df, prod_df, target_col, problem_type)
        st.session_state["target_col"] = target_col
        st.session_state["problem_type"] = problem_type

if "results" not in st.session_state:
    st.info("Click **Run investigation** to start.")
    st.stop()

results = st.session_state["results"]
bundle = results["bundle"]
problem_type = results["problem_type"]
validation_metrics = results["validation_metrics"]
production_metrics = results["production_metrics"]
metric_delta_df = results["metric_delta_df"]
drift_df = results["drift_df"]
importance_df = results["importance_df"]

tab_dashboard, tab_drift, tab_explain, tab_rag, tab_report = st.tabs(
    ["Dashboard", "Drift Detection", "Explainability", "RAG Investigator", "Report"]
)

with tab_dashboard:
    st.subheader("2. Model Performance")
    st.markdown("The model is trained on the training CSV and evaluated on a validation split. If the production file includes the target column, DriftLens also evaluates production performance.")

    delta_lookup = {row["metric"]: row["delta"] for _, row in metric_delta_df.iterrows()}

    if problem_type == "classification":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Validation Accuracy", validation_metrics.get("accuracy"))
        with c2:
            metric_card("Validation F1", validation_metrics.get("f1"))
        with c3:
            metric_card("Production Accuracy", production_metrics.get("accuracy") if production_metrics else None, delta_lookup.get("accuracy"))
        with c4:
            metric_card("Production F1", production_metrics.get("f1") if production_metrics else None, delta_lookup.get("f1"))

        st.markdown("**Metric comparison**")
        st.dataframe(metric_delta_df, use_container_width=True)

        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            plot_confusion_matrix(validation_metrics["confusion_matrix"], "Validation Confusion Matrix", validation_metrics.get("class_labels"))
        with cm_col2:
            if production_metrics:
                plot_confusion_matrix(production_metrics["confusion_matrix"], "Production Confusion Matrix", production_metrics.get("class_labels"))
            else:
                st.info("Production labels not provided, so no production confusion matrix is available.")

    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Validation MAE", validation_metrics.get("mae"))
        with c2:
            metric_card("Validation R²", validation_metrics.get("r2"))
        with c3:
            metric_card("Production MAE", production_metrics.get("mae") if production_metrics else None, delta_lookup.get("mae"), inverse_delta=True)
        with c4:
            metric_card("Production R²", production_metrics.get("r2") if production_metrics else None, delta_lookup.get("r2"))

        st.markdown("**Metric comparison**")
        st.dataframe(metric_delta_df, use_container_width=True)

        plot_col1, plot_col2 = st.columns(2)
        with plot_col1:
            plot_regression_predictions(bundle.pipeline, bundle.validation_X, bundle.validation_y, "Validation: Actual vs Predicted")
        with plot_col2:
            if results["prod_y"] is not None:
                plot_regression_predictions(bundle.pipeline, results["prod_X"], results["prod_y"], "Production: Actual vs Predicted")
            else:
                st.info("Production labels not provided, so no production actual-vs-predicted plot is available.")

with tab_drift:
    st.subheader("3. Data Drift Detection")
    drift_count = int(drift_df["drift_detected"].sum()) if not drift_df.empty else 0
    st.metric("Drifted Features", drift_count)
    st.dataframe(drift_df, use_container_width=True)

    drifted_numeric = drift_df[(drift_df["drift_detected"] == True) & (drift_df["type"] == "numeric")]
    if not drifted_numeric.empty:
        selected_feature = st.selectbox("Visualize a drifted numeric feature", drifted_numeric["feature"].tolist())
        plot_numeric_distribution(train_df, prod_df, selected_feature)
    else:
        st.info("No numeric drifted feature found for histogram visualization.")

with tab_explain:
    st.subheader("4. Model Explainability")
    if problem_type == "regression":
        st.markdown("This MVP uses permutation importance with negative MAE scoring for regression. Higher values mean the feature matters more.")
    else:
        st.markdown("This MVP uses permutation importance with weighted F1 scoring for classification. Higher values mean the feature matters more.")
    st.dataframe(importance_df, use_container_width=True)
    if not importance_df.empty:
        top = importance_df.head(10).sort_values("importance_mean")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(top["feature"], top["importance_mean"])
        ax.set_xlabel("Permutation importance")
        ax.set_title("Top model features")
        st.pyplot(fig)

with tab_rag:
    st.subheader("5. RAG Investigator")
    st.markdown("Ask a question. The app retrieves evidence from the local knowledge base and combines it with the drift and model results.")
    default_question = "Why did the model performance drop in production?" if problem_type == "classification" else "Why did the regression model prediction error increase in production?"
    question = st.text_input("Investigation question", value=default_question)
    rag = LocalRAG(APP_DIR / "knowledge_base")
    retrieval_query = f"{question} {problem_type}"
    if not drift_df.empty:
        retrieval_query += " " + " ".join(drift_df.head(5)["feature"].astype(str).tolist())
    evidence = rag.retrieve(retrieval_query, top_k=4)
    answer = synthesize_investigation_answer(
        question=question,
        validation_metrics=validation_metrics,
        production_metrics=production_metrics,
        metric_delta_df=metric_delta_df,
        drift_df=drift_df,
        importance_df=importance_df,
        evidence=evidence,
        problem_type=problem_type,
    )
    st.markdown(answer)
    st.session_state["investigation_answer"] = answer

with tab_report:
    st.subheader("6. Downloadable Report")
    answer = st.session_state.get("investigation_answer")
    if not answer:
        rag = LocalRAG(APP_DIR / "knowledge_base")
        evidence = rag.retrieve(f"model failure drift production root cause {problem_type}", top_k=4)
        answer = synthesize_investigation_answer(
            question="Why did the model fail?",
            validation_metrics=validation_metrics,
            production_metrics=production_metrics,
            metric_delta_df=metric_delta_df,
            drift_df=drift_df,
            importance_df=importance_df,
            evidence=evidence,
            problem_type=problem_type,
        )
    report_md = build_markdown_report(
        validation_metrics=validation_metrics,
        production_metrics=production_metrics,
        metric_delta_df=metric_delta_df,
        drift_df=drift_df,
        importance_df=importance_df,
        investigation_answer=answer,
        problem_type=problem_type,
    )
    st.markdown(report_md)
    st.download_button(
        label="Download Markdown Report",
        data=report_md,
        file_name="driftlens_failure_report.md",
        mime="text/markdown",
    )
