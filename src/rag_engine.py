"""A lightweight local RAG engine for DriftLens.

This version intentionally avoids paid APIs so the app can be deployed publicly
from GitHub without secrets. It uses TF-IDF retrieval over markdown documents and
combines retrieved evidence with deterministic investigation rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class EvidenceChunk:
    source: str
    text: str
    score: float


def load_knowledge_base(kb_dir: str | Path = "knowledge_base") -> list[tuple[str, str]]:
    kb_path = Path(kb_dir)
    docs: list[tuple[str, str]] = []
    if not kb_path.exists():
        return docs
    for file_path in sorted(kb_path.glob("*.md")):
        docs.append((file_path.name, file_path.read_text(encoding="utf-8")))
    return docs


def chunk_documents(docs: list[tuple[str, str]], max_chars: int = 900) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    for source, text in docs:
        raw_parts = [part.strip() for part in text.split("\n\n") if part.strip()]
        buffer = ""
        for part in raw_parts:
            if len(buffer) + len(part) + 2 <= max_chars:
                buffer = f"{buffer}\n\n{part}".strip()
            else:
                if buffer:
                    chunks.append((source, buffer))
                buffer = part
        if buffer:
            chunks.append((source, buffer))
    return chunks


class LocalRAG:
    def __init__(self, kb_dir: str | Path = "knowledge_base") -> None:
        docs = load_knowledge_base(kb_dir)
        self.chunks = chunk_documents(docs)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = None
        if self.chunks:
            self.matrix = self.vectorizer.fit_transform([text for _, text in self.chunks])

    def retrieve(self, query: str, top_k: int = 4) -> list[EvidenceChunk]:
        if self.matrix is None or not self.chunks:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        top_indices = scores.argsort()[::-1][:top_k]
        return [
            EvidenceChunk(source=self.chunks[i][0], text=self.chunks[i][1], score=float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]


def _fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def _fmt_num(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"


def build_root_cause_hypotheses(
    metric_delta_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    problem_type: str = "classification",
) -> list[dict[str, str]]:
    hypotheses: list[dict[str, str]] = []

    drifted = drift_df[drift_df["drift_detected"] == True].copy() if not drift_df.empty else pd.DataFrame()
    top_important = set(importance_df.head(5)["feature"].tolist()) if not importance_df.empty else set()

    if not drifted.empty:
        important_drift = drifted[drifted["feature"].isin(top_important)]
        if not important_drift.empty:
            features = ", ".join(important_drift.head(4)["feature"].tolist())
            hypotheses.append(
                {
                    "cause": f"High-impact feature drift in {features}",
                    "confidence": "High",
                    "evidence": "These features changed between training and production and also rank high in model importance.",
                    "fix": "Retrain with recent production-like data and verify upstream feature engineering for these columns.",
                }
            )

        features = ", ".join(drifted.head(4)["feature"].tolist())
        hypotheses.append(
            {
                "cause": f"Production data distribution shifted in {features}",
                "confidence": "Medium",
                "evidence": "Statistical drift tests flagged these columns as different from training data.",
                "fix": "Add drift monitoring thresholds and compare recent production cohorts before deployment.",
            }
        )

    if not metric_delta_df.empty:
        if problem_type == "regression":
            mae_row = metric_delta_df[metric_delta_df["metric"] == "mae"]
            rmse_row = metric_delta_df[metric_delta_df["metric"] == "rmse"]
            r2_row = metric_delta_df[metric_delta_df["metric"] == "r2"]
            if not mae_row.empty and pd.notna(mae_row.iloc[0].get("delta")) and mae_row.iloc[0]["delta"] > 0:
                hypotheses.append(
                    {
                        "cause": "Prediction error increased in production",
                        "confidence": "High" if mae_row.iloc[0]["delta"] > 0.1 * max(abs(mae_row.iloc[0]["validation"]), 1) else "Medium",
                        "evidence": f"MAE changed by {mae_row.iloc[0]['delta']:.3f}; higher MAE means larger average prediction error.",
                        "fix": "Inspect residuals by segment, recalibrate the model, and retrain using recent data.",
                    }
                )
            if not rmse_row.empty and pd.notna(rmse_row.iloc[0].get("delta")) and rmse_row.iloc[0]["delta"] > 0:
                hypotheses.append(
                    {
                        "cause": "Large errors or outliers became more common",
                        "confidence": "Medium",
                        "evidence": f"RMSE changed by {rmse_row.iloc[0]['delta']:.3f}; RMSE is sensitive to large errors.",
                        "fix": "Review extreme predictions and check whether new production ranges were missing from training.",
                    }
                )
            if not r2_row.empty and pd.notna(r2_row.iloc[0].get("delta")) and r2_row.iloc[0]["delta"] < -0.05:
                hypotheses.append(
                    {
                        "cause": "Regression fit quality dropped",
                        "confidence": "Medium",
                        "evidence": f"R² changed by {r2_row.iloc[0]['delta']:.3f}, meaning the model explains less variation in production.",
                        "fix": "Compare residual plots and retrain with time-based validation.",
                    }
                )
        else:
            recall_row = metric_delta_df[metric_delta_df["metric"] == "recall"]
            f1_row = metric_delta_df[metric_delta_df["metric"] == "f1"]
            if not recall_row.empty and pd.notna(recall_row.iloc[0].get("delta")) and recall_row.iloc[0]["delta"] < -0.05:
                hypotheses.append(
                    {
                        "cause": "Recall degradation",
                        "confidence": "Medium",
                        "evidence": f"Recall changed by {recall_row.iloc[0]['delta']:.3f}, which means the model is missing more true cases in production.",
                        "fix": "Tune decision thresholds, rebalance the training data, and inspect false negatives by segment.",
                    }
                )
            if not f1_row.empty and pd.notna(f1_row.iloc[0].get("delta")) and f1_row.iloc[0]["delta"] < -0.05:
                hypotheses.append(
                    {
                        "cause": "Overall classification quality dropped",
                        "confidence": "Medium",
                        "evidence": f"F1 changed by {f1_row.iloc[0]['delta']:.3f}, suggesting precision/recall balance is worse in production.",
                        "fix": "Compare confusion matrices and evaluate retraining on a time-based split.",
                    }
                )

    missing_shift = drift_df[drift_df["missing_delta"].abs() >= 0.05] if not drift_df.empty else pd.DataFrame()
    if not missing_shift.empty:
        features = ", ".join(missing_shift.head(4)["feature"].tolist())
        hypotheses.append(
            {
                "cause": f"Data quality regression in {features}",
                "confidence": "Medium",
                "evidence": "Missing-value rates changed by at least five percentage points.",
                "fix": "Audit ingestion, validation, and imputation logic for these fields.",
            }
        )

    if not hypotheses:
        hypotheses.append(
            {
                "cause": "No single obvious failure mode detected",
                "confidence": "Low",
                "evidence": "The available metrics and drift checks did not show a strong signal.",
                "fix": "Add more production labels, segment-level evaluation, and historical incident notes.",
            }
        )

    return hypotheses[:5]


def synthesize_investigation_answer(
    question: str,
    validation_metrics: dict[str, Any],
    production_metrics: dict[str, Any] | None,
    metric_delta_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    evidence: list[EvidenceChunk],
    problem_type: str = "classification",
) -> str:
    hypotheses = build_root_cause_hypotheses(metric_delta_df, drift_df, importance_df, problem_type)
    top_drift = drift_df[drift_df["drift_detected"] == True].head(5) if not drift_df.empty else pd.DataFrame()

    lines = [
        "## Investigation Summary",
        f"**Question:** {question}",
        "",
        "### Performance Signal",
    ]

    if problem_type == "regression":
        lines.append(f"- Validation MAE: **{_fmt_num(validation_metrics.get('mae'))}**")
        lines.append(f"- Validation R²: **{_fmt_num(validation_metrics.get('r2'))}**")
        if production_metrics:
            lines.append(f"- Production MAE: **{_fmt_num(production_metrics.get('mae'))}**")
            lines.append(f"- Production R²: **{_fmt_num(production_metrics.get('r2'))}**")
            if production_metrics.get("mae") is not None and validation_metrics.get("mae") is not None:
                lines.append(f"- MAE Delta: **{production_metrics['mae'] - validation_metrics['mae']:+.3f}**")
        else:
            lines.append("- Production labels were not provided, so regression failure analysis is based on drift and model sensitivity.")
    else:
        lines.append(f"- Validation F1: **{_fmt_pct(validation_metrics.get('f1'))}**")
        if production_metrics:
            lines.append(f"- Production F1: **{_fmt_pct(production_metrics.get('f1'))}**")
            if production_metrics.get("f1") is not None and validation_metrics.get("f1") is not None:
                lines.append(f"- F1 Delta: **{production_metrics['f1'] - validation_metrics['f1']:+.3f}**")
        else:
            lines.append("- Production labels were not provided, so classification failure analysis is based on drift and model sensitivity.")

    lines.extend(["", "### Most Likely Root Causes"])
    for idx, h in enumerate(hypotheses, start=1):
        lines.append(f"{idx}. **{h['cause']}** — Confidence: **{h['confidence']}**")
        lines.append(f"   - Evidence: {h['evidence']}")
        lines.append(f"   - Recommended fix: {h['fix']}")

    if not top_drift.empty:
        lines.extend(["", "### Top Drift Signals"])
        for _, row in top_drift.iterrows():
            psi = row.get("psi")
            psi_txt = "n/a" if pd.isna(psi) else f"{psi:.3f}"
            p_value = row.get("p_value")
            p_txt = "n/a" if pd.isna(p_value) else f"{p_value:.3g}"
            lines.append(
                f"- **{row['feature']}** ({row['type']}): {row['train_summary']} → {row['production_summary']}; p={p_txt}; PSI={psi_txt}"
            )

    if not importance_df.empty:
        lines.extend(["", "### Important Model Features"])
        for _, row in importance_df.head(5).iterrows():
            lines.append(f"- **{row['feature']}**: permutation importance {row['importance_mean']:.4f}")

    if evidence:
        lines.extend(["", "### Retrieved Knowledge Base Evidence"])
        for item in evidence:
            cleaned = item.text.replace("\n", " ")
            if len(cleaned) > 420:
                cleaned = cleaned[:420].rstrip() + "..."
            lines.append(f"- **{item.source}** — relevance {item.score:.2f}: {cleaned}")

    if problem_type == "regression":
        lines.extend(
            [
                "",
                "### Next Actions",
                "1. Compare residuals across important business segments.",
                "2. Check whether production values moved outside training ranges.",
                "3. Retrain on recent production-like data and compare MAE/RMSE/R².",
                "4. Add automated drift thresholds and residual monitoring before the next deployment.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "### Next Actions",
                "1. Validate the top drifted features against the production data pipeline.",
                "2. Review false negatives and false positives by business segment.",
                "3. Retrain on a recent time-based sample and compare it against the current model.",
                "4. Add automated drift thresholds before the next deployment.",
            ]
        )
    return "\n".join(lines)
