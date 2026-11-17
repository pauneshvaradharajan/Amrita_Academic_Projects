from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef)
from sklearn.metrics.pairwise import cosine_similarity
from starlette.concurrency import run_in_threadpool

from webapp.phase1_runner import run_phase1_from_uploads

APP_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
REFERENCE = json.loads((APP_DIR / "phase1_reference.json").read_text(encoding="utf-8"))
KB_TEXT = (APP_DIR / "knowledge_base.md").read_text(encoding="utf-8")
PLOTS_DIR = APP_DIR / "assets" / "phase1_plots"
NOTEBOOK_PATH = APP_DIR / "assets" / "Phase1_Final_Four_Modality_Experiment_preview.ipynb"
PLOT_FILES = {
    "01_class_distribution.png", "02_oof_confusion_matrix.png",
    "03_metrics_and_fold_stability.png", "04_per_class_metrics.png",
}
CLASS_NAMES = ["Low", "Medium", "High"]
ORIGINAL_TO_GROUPED = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PREDICTION_ROWS = 25_000
MAX_PHASE1_ROWS = 1_800
MAX_PHASE1_FEATURES = 5_000
SESSION_TTL_SECONDS = 30 * 60
sessions: dict[str, dict[str, Any]] = {}
phase1_jobs: dict[str, dict[str, Any]] = {}
llm_calls: deque[float] = deque()

app = FastAPI(title="MultiPENG Engagement Research Assistant", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    dataset_id: str | None = None
    use_gemini: bool = False
    history: list[dict[str, str]] = Field(default_factory=list, max_length=12)


def expire_sessions() -> None:
    cutoff = time.time() - SESSION_TTL_SECONDS
    for key in [key for key, value in sessions.items() if value["created"] < cutoff]:
        sessions.pop(key, None)
    for key in list(phase1_jobs):
        item = phase1_jobs[key]
        if item.get("finished_at", time.time()) < cutoff:
            phase1_jobs.pop(key, None)


def retrieve(question: str, limit: int = 4) -> list[dict[str, Any]]:
    chunks = [part.strip() for part in re.split(r"\n(?=#{1,3} )", KB_TEXT) if len(part.strip()) > 40]
    if not chunks:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10_000)
    matrix = vectorizer.fit_transform(chunks + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    top = np.argsort(scores)[::-1][:limit]
    return [{"title": chunks[i].splitlines()[0].lstrip("# ")[:120],
             "text": chunks[i][:2_500], "score": round(float(scores[i]), 3)}
            for i in top if scores[i] > 0]


def _to_class(value: Any, numeric_is_grouped: bool) -> int:
    text = str(value).strip().lower()
    aliases = {"low": 0, "low (0/1)": 0, "medium": 1, "mid": 1, "high": 2,
               "high (3/4)": 2, "0": 0, "1": 1, "2": 2, "3": 2, "4": 2}
    try:
        numeric_value = float(text)
        numeric = int(numeric_value) if numeric_value.is_integer() else None
    except ValueError:
        numeric = None
    if numeric is not None:
        if numeric not in ORIGINAL_TO_GROUPED:
            raise ValueError(f"Unrecognized engagement label {value!r}. Use Low/Medium/High or numeric labels 0–4.")
        return numeric if numeric_is_grouped and numeric <= 2 else ORIGINAL_TO_GROUPED[numeric]
    if text not in aliases:
        raise ValueError(f"Unrecognized engagement label {value!r}. Use Low/Medium/High or numeric labels 0–4.")
    return aliases[text]


def parse_prediction_csv(payload: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(payload), low_memory=False)
    if frame.empty:
        raise ValueError("The CSV has no data rows.")
    if len(frame) > MAX_PREDICTION_ROWS:
        raise ValueError(f"This public app accepts up to {MAX_PREDICTION_ROWS:,} prediction rows per upload.")
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    source_columns = list(frame.columns)
    aliases = {
        "participant_id": ["participant_id", "participant", "subject_id", "player_id"],
        "session_id": ["session_id", "submission_id", "session", "submission"],
        "predicted_engagement": ["predicted_engagement", "predicted_class", "prediction", "engagement_class", "engagement"],
        "timestamp": ["timestamp", "datetime", "session_time", "time"],
        "actual_grouped": ["actual_engagement_grouped_0_2", "actual_grouped", "actual_class_0_2"],
        "actual_original": ["actual_engagement_original_0_4", "actual_engagement", "actual_label", "ground_truth", "true_label"],
    }
    rename: dict[str, str] = {}
    for target, names in aliases.items():
        source = next((name for name in names if name in frame.columns), None)
        if source and source != target:
            rename[source] = target
    frame = frame.rename(columns=rename)
    if "predicted_engagement" not in frame:
        raise ValueError("Add a predicted_engagement or prediction column to the CSV.")
    numeric_prediction_values = pd.to_numeric(frame["predicted_engagement"], errors="coerce").dropna()
    grouped_predictions = (
        "actual_grouped" in frame.columns
        or frame["predicted_engagement"].astype(str).str.lower().isin(["medium", "mid", "high"]).any()
        or (not numeric_prediction_values.empty and numeric_prediction_values.between(0, 2).all())
    )
    frame["predicted_label"] = frame["predicted_engagement"].map(lambda x: _to_class(x, grouped_predictions))
    actual_source = "actual_grouped" if "actual_grouped" in frame else "actual_original" if "actual_original" in frame else None
    if actual_source:
        frame["actual_label"] = frame[actual_source].map(lambda x: _to_class(x, actual_source == "actual_grouped"))
    if "timestamp" in frame:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame.attrs["source_columns"] = source_columns
    return frame


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    counts = np.bincount(frame["predicted_label"].to_numpy(dtype=int), minlength=3)
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "participants": int(frame["participant_id"].nunique()) if "participant_id" in frame else None,
        "columns": frame.attrs.get("source_columns", list(frame.columns)),
        "class_counts": {CLASS_NAMES[i]: int(counts[i]) for i in range(3)},
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "unique_sessions": int(frame["session_id"].nunique()) if "session_id" in frame else None,
    }
    if "actual_label" in frame:
        actual = frame["actual_label"].to_numpy(dtype=int)
        predicted = frame["predicted_label"].to_numpy(dtype=int)
        matrix = confusion_matrix(actual, predicted, labels=[0, 1, 2])
        report = classification_report(actual, predicted, labels=[0, 1, 2], target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        actual_counts = np.bincount(actual, minlength=3)
        majority = int(actual_counts.argmax())
        summary["evaluation"] = {
            "n_labeled": int(len(actual)), "accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(actual, predicted, average="weighted", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
            "mcc": float(matthews_corrcoef(actual, predicted)),
            "majority_baseline_accuracy": float(np.mean(actual == majority)),
            "majority_class": CLASS_NAMES[majority], "confusion_matrix": matrix.tolist(),
            "actual_counts": {CLASS_NAMES[i]: int(actual_counts[i]) for i in range(3)},
            "per_class": {name: {"precision": report[name]["precision"], "recall": report[name]["recall"],
                                 "f1": report[name]["f1-score"], "support": report[name]["support"]}
                          for name in CLASS_NAMES},
        }
    return summary


def deterministic_answer(question: str, summary: dict[str, Any] | None, sources: list[dict[str, Any]]) -> tuple[str, list[str]]:
    q = re.sub(r"\s+", " ", question.lower()).strip()
    tools: list[str] = []
    uploaded = bool(summary and summary.get("columns"))
    if not uploaded:
        summary = {"rows": REFERENCE["sessions"], "participants": REFERENCE["participants"],
                   "class_counts": {"Low": 97, "Medium": 156, "High": 620}}
        tools.append("saved Phase 1 aggregate results")

    def has(*terms: str) -> bool:
        return any(term in q for term in terms)

    def saved_confusion() -> list[list[int]]:
        return REFERENCE["confusion_matrix"]["rows_actual_columns_predicted"]

    def confusion_table(matrix: list[list[int]]) -> str:
        rows = ["| Actual \\ Predicted | Low | Medium | High |", "|---|---:|---:|---:|"]
        rows.extend(f"| {CLASS_NAMES[i]} | {matrix[i][0]} | {matrix[i][1]} | {matrix[i][2]} |" for i in range(3))
        return "\n".join(rows)

    if has("why did i build", "why did i create", "why build", "why create", "project purpose", "purpose of this project", "what is this project", "what are you", "who are you", "what is your purpose", "what is the aim"):
        answer = ("You built this project to estimate a game player's engagement from four synchronized signals—EEG, eye tracking, facial behavior, and heart rate—and make the results easier for researchers to inspect. "
                  "Phase 1 predicts Low, Medium, or High engagement and saves evaluation evidence. Phase 2 analyzes those saved results, retrieves project notes, and explains the evidence in plain English. "
                  "It supports research; it does not read a player's mind or make clinical conclusions.")
        tools.append("project-purpose notes")

    elif has("novelty", "contribution", "what is new", "what is novel"):
        answer = ("The project's contribution is the **integrated and auditable workflow**: four-modality Phase 1 evaluation is connected to a Phase 2 assistant that calculates from saved evidence, retrieves project notes, and explains results with limitations. "
                  "The novelty claim is the end-to-end decision-support integration—not a new classifier or proven superiority over existing models.")
        tools.append("project contribution notes")

    elif has("fold", "cross-validation", "cross validation", "grouped", "out-of-fold", "oof", "unseen participant", "split", "validation", "generaliz"):
        answer = ("Phase 1 used nested participant-grouped cross-validation: **5 outer folds** produced out-of-fold (OOF) predictions, and **3 inner folds** selected the model within each outer training portion. "
                  "Participants in an outer validation fold were held out from that fold's training data. Each of the 873 prepared sessions received an OOF prediction. "
                  "This estimates performance on new participants, but it is **not a separate locked test set**. The Phase 2 DSS displays the saved run; it does not rerun or independently verify the original folds.")
        tools.append("saved validation protocol")

    elif has("column", "schema", "field name", "list fields"):
        if uploaded:
            columns = summary.get("columns", [])
            answer = f"The uploaded CSV has **{summary['rows']:,} rows** and **{len(columns)} columns**. Its column names are:\n\n" + "\n".join(f"- `{column}`" for column in columns)
            tools.append("uploaded CSV schema")
        else:
            answer = ("No Phase 1 prediction CSV is loaded, so I cannot truthfully list its exact columns. The saved Phase 1 reference contains aggregate results and plots, not the original session-level table. "
                      "Upload the Phase 1 prediction CSV in **Analyze a Phase 1 CSV** to display its exact schema. The upload needs a `predicted_engagement` column; actual-label columns are needed to calculate performance metrics.")
            tools.append("saved-file scope and upload schema requirements")

    elif has("dataset", "data set", "multipeng", "modalit", "what data", "how many participants", "how many sessions"):
        if uploaded:
            cols = summary.get("columns", [])
            answer = f"The currently uploaded file contains **{summary['rows']:,} rows** across **{len(cols)} columns**. "
        else:
            answer = "The project uses MultiPENG gameplay-session data with four prepared modalities: **EEG** (brain-signal features), **eye tracking** (gaze/ocular behavior), **facial behavior/OpenFace** (face and head features), and **heart rate**. The current prediction target groups the original 0–4 labels into **Low (0/1), Medium (2), and High (3/4)**. "
        answer += ("The project abstract describes the original dataset as **900 sessions from 39 participants**; the prepared Phase 1 evaluation saved here covers **873 usable sessions from 38 participants**. "
                   "Those are different scopes. The DSS stores only aggregate Phase 1 evidence by default, not the raw multimodal feature tables.")
        if uploaded and has("column", "schema", "fields"):
            answer += " Uploaded columns: " + ", ".join(f"`{c}`" for c in summary.get("columns", [])) + "."
        tools.append("dataset notes and saved Phase 1 scope")

    elif has("where stored", "where is it stored", "where is that file", "where is this file", "where is that stored", "where are results", "where are my saved results", "where are files", "where saved", "where is the data", "storage", "database"):
        if has("knowledge", "rag", "vector", "retrieval"):
            answer = ("The RAG knowledge source is `phase2_agentic_engagement_dss/knowledge_base.md` in the GitHub repository. The app reads this file locally and uses TF-IDF to retrieve relevant passages; it does **not** use a separate vector database. "
                      "The saved Phase 1 summary is in `phase2_agentic_engagement_dss/phase1_reference.json`, with figures in `phase2_agentic_engagement_dss/assets/phase1_plots/`. "
                      "Uploaded prediction or modality CSVs are held temporarily in server memory and expire; they are not written to the repository or a database.")
        else:
            answer = ("The saved Phase 1 aggregate metrics and class counts are in `phase2_agentic_engagement_dss/phase1_reference.json`; saved figures are in `phase2_agentic_engagement_dss/assets/phase1_plots/`; and the notebook preview is under `phase2_agentic_engagement_dss/assets/`. "
                      "An uploaded CSV is kept temporarily in server memory and expires; the app does not write it to GitHub or a database.")
        tools.append("repository file locations and upload-storage policy")

    elif has("evaluate the dss", "evaluate this dss", "dss evaluation", "evaluate this assistant", "assistant evaluation", "evaluation questions"):
        answer = ("Evaluate the DSS against a fixed set of questions with independently prepared reference answers. Check: **(1)** schema, types, row counts, and missingness; **(2)** whether tables and plots match independent calculations; **(3)** whether method explanations and limitations are correct; **(4)** whether independent verification reduces errors; and **(5)** task time and manual effort. "
                  "Report each category separately. Phase 1 model accuracy is not the same as the correctness of DSS explanations.")
        tools.append("DSS evaluation protocol notes")

    elif has("rag", "retrieval augmented", "knowledge base", "knowledge_base", "llm", "gemini", "agentic", "agent"):
        answer = ("The DSS uses a bounded, tool-assisted workflow. It routes supported questions to local calculations (such as metrics, class counts, or a confusion matrix). "
                  "For project questions, TF-IDF retrieval selects relevant passages from the local `knowledge_base.md` file. "
                  "If Gemini is enabled and a key is configured, it receives the question, aggregate summaries, and retrieved notes to explain the evidence; raw CSV rows, participant IDs, and sensor features are excluded. "
                  "Without a working Gemini key, the local answer path is used. This is a bounded decision-support assistant, not an unrestricted autonomous agent.")
        tools.append("DSS workflow and RAG notes")

    elif has("imbalance", "class balance", "balanced data", "data balanced", "balance data", "oversampl", "minority class", "majority class"):
        if uploaded and summary.get("evaluation"):
            counts = summary["evaluation"].get("actual_counts", {})
            scope = "actual labels in the uploaded evaluation file"
        else:
            counts = {"Low": 97, "Medium": 156, "High": 620}
            scope = "saved Phase 1 actual-label support"
        total = max(sum(counts.values()), 1)
        answer = f"Class balance in the {scope}:\n\n" + "\n".join(f"- **{name}:** {count:,} ({count/total:.1%})" for name, count in counts.items())
        answer += ("\n\nRandom oversampling was compared with class weighting. The selected RBF-SVM pipeline used random oversampling **only inside each training fold**; validation folds kept their original class distribution. "
                   "Oversampling helps expose the learner to more minority examples, but it did not solve the low Low/Medium recall, so class imbalance remains a limitation.")
        tools.append("class-count calculation and saved balancing configuration")

    elif has("preprocess", "cleaning", "imput", "scal", "anova", "top 50", "feature selection", "selected features", "50%"):
        answer = ("Inside each training fold, the Phase 1 pipeline removed invalid, constant, and duplicate features; median-imputed missing values; standardized features; then used ANOVA to retain the top **50% of features per modality**. "
                  "These steps were fitted within training data to avoid using held-out participants when preprocessing or selecting features. "
                  "The prepared inputs were already session-level feature tables, so arbitrary column positions were not treated as time steps.")
        tools.append("saved preprocessing and feature-selection configuration")

    elif has("confusion", "mistake", "error", "misclassif"):
        matrix = summary["evaluation"]["confusion_matrix"] if uploaded and summary.get("evaluation") else saved_confusion()
        answer = "Confusion matrix (**actual classes are rows; predictions are columns**):\n\n" + confusion_table(matrix)
        errors = [(matrix[i][j], i, j) for i in range(3) for j in range(3) if i != j]
        errors.sort(reverse=True)
        if errors and errors[0][0] > 0:
            count, actual_i, predicted_i = errors[0]
            answer += f"\n\nThe largest error type is **{CLASS_NAMES[actual_i]} predicted as {CLASS_NAMES[predicted_i]} ({count} sessions)**. The saved run's main issue is confusing minority Low/Medium sessions with High."
        answer += "\n\nThese are uploaded-file calculations." if uploaded and summary.get("evaluation") else "\n\nThese are saved grouped OOF results, not an independent test result."
        tools.append("uploaded or saved confusion matrix")

    elif has("verify", "prove", "can i trust", "is it correct", "is the result valid", "is the accuracy correct"):
        answer = ("The DSS can independently recalculate accuracy and F1 from an uploaded file that contains paired actual and predicted labels, and you can compare those values with a separate calculation. "
                  "It cannot prove that each model prediction is true or reconstruct the Phase 1 grouped folds from aggregate summaries alone. The saved reference is aggregate-only, so row-level verification requires the original OOF prediction file.")
        tools.append("verification scope and uploaded-label calculation")

    elif has("model", "algorithm", "svm", "random forest", "classifier", "which one selected", "which did you select", "why selected", "why choose", "why chosen"):
        answer = ("The final saved Phase 1 selection was **RBF-SVM with random oversampling (C=1)** in each of the 5 outer folds. **Random Forest** was the comparison model. "
                  "An SVM finds a boundary between classes; the RBF kernel lets that boundary model nonlinear patterns. RBF-SVM was retained because it had the strongest inner-fold Macro-F1 under the model-selection procedure. "
                  "The results use engineered session-level features, not a sequential neural network. "
                  "The gated four-encoder MLP was defined but not executed in the saved CPU run.")
        if has("performance", "result", "compare", "accuracy", "f1", "score"):
            m, b = REFERENCE["overall_metrics"], REFERENCE["majority_reference_metrics"]
            answer += (f"\n\nSaved grouped OOF performance: accuracy **{m['Accuracy']:.1%}**, Macro-F1 **{m['Macro-F1']:.3f}**, Weighted-F1 **{m['Weighted-F1']:.3f}**. "
                       f"Majority-only accuracy is **{b['Accuracy']:.1%}**, so accuracy is lower than that baseline even though Macro-F1 and Weighted-F1 are higher.")
        tools.append("saved model-selection record")

    elif has("accuracy", "macro", "f1", "performance", "metric", "score", "result", "compare"):
        evaluation = summary.get("evaluation") if uploaded else None
        if has("what is", "what does", "meaning", "explain", "define") and has("macro", "weighted", "mcc", "accuracy", "balanced"):
            answer = ("Accuracy is the share of sessions classified correctly. Macro-F1 averages the F1 score of each class equally, so it reveals weak minority-class performance. "
                      "Weighted-F1 averages class F1 scores by class size, so the High class contributes more. Balanced accuracy averages class recall. MCC summarizes agreement between actual and predicted classes from -1 to +1. "
                      "For this run, Macro-F1 is **0.371** and accuracy is **65.75%**.")
            tools.append("metric definitions and saved Phase 1 values")
        elif evaluation:
            answer = (f"Calculated from **{evaluation['n_labeled']:,} uploaded rows with actual labels**: accuracy **{evaluation['accuracy']:.1%}**, Macro-F1 **{evaluation['macro_f1']:.3f}**, Weighted-F1 **{evaluation['weighted_f1']:.3f}**, balanced accuracy **{evaluation['balanced_accuracy']:.3f}**, and MCC **{evaluation['mcc']:.3f}**. "
                      f"The majority-only accuracy baseline on these rows is **{evaluation['majority_baseline_accuracy']:.1%}**.")
            tools.append("metrics calculated from uploaded actual and predicted labels")
        else:
            m, b = REFERENCE["overall_metrics"], REFERENCE["majority_reference_metrics"]
            answer = (f"Saved grouped OOF Phase 1 results: accuracy **{m['Accuracy']:.1%}**, Macro-F1 **{m['Macro-F1']:.3f}**, Weighted-F1 **{m['Weighted-F1']:.3f}**, balanced accuracy **{m['Balanced accuracy']:.3f}**, and MCC **{m['MCC']:.3f}**. "
                      f"The majority-only accuracy reference is **{b['Accuracy']:.1%}**. So accuracy is below that baseline, while Macro-F1 and Weighted-F1 are higher. Low/Medium recall is weak. This is nested participant-grouped cross-validation, not a locked test score.")
            tools.append("saved Phase 1 metrics")

    elif has("distribution", "class count", "how many low", "how many medium", "how many high", "percentage"):
        if uploaded:
            counts = summary.get("class_counts", {})
            kind = "predicted"
        else:
            counts = {"Low": 97, "Medium": 156, "High": 620}
            kind = "actual label"
        total = max(sum(counts.values()), 1)
        answer = f"{kind.title()} class counts in {'the uploaded file' if uploaded else 'the saved Phase 1 dataset'} ({total:,} sessions):\n\n" + "\n".join(f"- **{name}:** {count:,} ({count/total:.1%})" for name, count in counts.items())
        if not uploaded:
            answer += "\n\nThese are actual class supports, not predicted counts. The saved reference does not include the full session-level table."
        elif summary.get("evaluation"):
            answer += "\n\nThe uploaded file also contains actual labels; ask for performance metrics to compare predictions against them."
        else:
            answer += "\n\nThese are predicted labels only, so this distribution cannot establish accuracy."
        tools.append("class-count calculation")

    elif has("duplicate", "missing", "data quality", "quality check", "invalid", "data type"):
        if uploaded:
            answer = (f"The uploaded file has **{summary['rows']:,} rows**, **{summary['missing_cells']:,} missing cells**, and **{summary['duplicate_rows']:,} fully duplicated rows**. "
                      + (f"It contains **{summary['unique_sessions']:,} unique session IDs**." if summary.get("unique_sessions") is not None else "It has no recognized session-ID column, so session uniqueness cannot be checked."))
            tools.append("uploaded CSV quality checks")
        else:
            answer = ("The saved Phase 1 reference contains aggregate metrics, counts, and plots—not all original rows or columns—so the DSS cannot calculate duplicate, missing-value, or data-type counts from it. "
                      "Upload the session-level CSV in **Analyze a Phase 1 CSV** for those checks.")
            tools.append("saved-file scope")

    elif has("phase 1", "phase1", "phase 2", "phase2", "architecture", "workflow", "methodology", "input", "output", "how does this work"):
        answer = ("**Phase 1** takes aligned EEG, eye-tracking, face/OpenFace, and HR feature tables plus engagement labels; it cleans and scales features, selects features within participant-grouped folds, compares RBF-SVM with Random Forest, and saves OOF predictions, metrics, and plots. "
                  "**Phase 2** reads the saved aggregate result files from the repository or an uploaded Phase 1 prediction CSV. It computes requested statistics locally, retrieves matching project notes with RAG, and can optionally use Gemini to explain those results. "
                  "The DSS does not silently retrain the model when you ask a question; a new Phase 1 run requires the four modality CSVs in the Run Phase 1 section.")
        tools.append("Phase 1/Phase 2 workflow notes")

    elif has("can you", "capabilit", "cannot", "can't", "limit", "reliable", "trust", "correct", "verify", "valid"):
        answer = ("The DSS can summarize the saved Phase 1 evidence, calculate metrics from an uploaded CSV that contains actual and predicted labels, show plots and tables, and retrieve project notes. "
                  "It cannot prove that a prediction is true, independently reproduce Phase 1 folds from aggregate files, infer missing columns, or diagnose a player's mental or health state. "
                  "To verify a reported accuracy, upload the original paired actual/predicted OOF labels and compare DSS calculations with an independent calculation from that same file. The repository's saved reference is aggregate-only, so it cannot trace a displayed metric back to every session row on its own.")
        tools.append("DSS capability and limitation notes")

    else:
        # Do not surface a loosely matching passage as though it answered the question.
        # RAG evidence is useful only when both relevance and lexical overlap are clear.
        query_terms = {term for term in re.findall(r"[a-z0-9_]+", q) if len(term) > 3}
        relevant = [source for source in sources
                    if float(source.get("score", 0)) >= 0.12
                    and query_terms.intersection(set(re.findall(r"[a-z0-9_]+", str(source.get("text", "")).lower())))]
        if relevant:
            answer = _grounded_excerpt(question, relevant)
            tools.append("local RAG retrieval from knowledge_base.md")
        else:
            answer = ("I don’t have enough project evidence to answer that specific question reliably. I can answer about the dataset, modalities, Phase 1 models and validation, class balance, preprocessing, results, confusion matrix, or the DSS/RAG/LLM workflow. "
                      "If you meant a previous topic, mention it in the question; otherwise upload the relevant Phase 1 CSV for file-specific analysis.")
    return answer, tools


def _grounded_excerpt(question: str, sources: list[dict[str, Any]]) -> str:
    """Return a concise evidence-led response instead of raw retrieved chunks."""
    query_terms = {term for term in re.findall(r"[a-z0-9]+", question.lower()) if len(term) > 2}
    candidates: list[tuple[float, str]] = []
    for source in sources[:3]:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", source.get("text", "")):
            sentence = sentence.strip(" #-\t")
            if len(sentence) < 35:
                continue
            terms = {term for term in re.findall(r"[a-z0-9]+", sentence.lower()) if len(term) > 2}
            overlap = len(query_terms & terms) / max(len(query_terms), 1)
            score = overlap + float(source.get("score", 0))
            candidates.append((score, sentence))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    for _, sentence in candidates:
        if sentence not in selected:
            selected.append(sentence)
        if len(selected) == 4:
            break
    if not selected:
        return "I found a related project note, but it does not contain enough detail to answer this confidently. Please narrow the question to a specific project topic."
    return "Based on the project notes:\n\n" + "\n\n".join(f"- {sentence}" for sentence in selected)


def resolve_follow_up(question: str, history: list[dict[str, str]]) -> str:
    """Add the last user topic only when the new prompt clearly depends on it."""
    q = question.strip()
    normalized = q.lower().strip(" .?!")
    words = normalized.split()
    follow_up = (normalized in {"why", "how", "what about that", "what about it", "tell me more", "explain more", "more details", "continue"}
                 or any(normalized.startswith(prefix) for prefix in (
                     "explain that", "tell me more about it", "what does that", "how does that", "why is that",
                     "where is that", "where is this", "what is that", "what was that", "show that", "list those",
                 ))
                 or (len(words) <= 7 and any(term in words for term in ("that", "this", "it", "those", "there"))))
    if not follow_up:
        return q
    previous_users = [str(item.get("content", "")).strip() for item in history
                      if item.get("role") == "user" and str(item.get("content", "")).strip()
                      and str(item.get("content", "")).strip().lower().strip(" .?!") != normalized]
    return f"{previous_users[-1]} {q}" if previous_users else q


def gemini_answer(question: str, summary: dict[str, Any] | None, sources: list[dict[str, Any]], history: list[dict[str, str]]) -> tuple[str | None, str]:
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        return None, "Gemini is not configured; this answer uses local calculations and retrieved notes."
    now = time.time()
    while llm_calls and now - llm_calls[0] > 3600:
        llm_calls.popleft()
    if len(llm_calls) >= 30:
        return None, "The shared Gemini limit has been reached for now; showing the local evidence-based answer."
    try:
        from google import genai
        client = genai.Client(api_key=key)
        context = {"data_summary": summary or {"saved_run": REFERENCE["overall_metrics"]},
                   "retrieved_project_notes": [item["text"] for item in sources]}
        recent = "\n".join(f"{m.get('role','user')}: {str(m.get('content',''))[:700]}" for m in history[-6:])
        prompt = ("You are a careful research decision-support assistant for a multimodal engagement project. "
                  "Answer the user's exact latest question directly in the first sentence. Do not substitute a retrieved passage for an answer, and do not paste long source text. "
                  "Use simple language and provide enough detail to satisfy the requested format (including every requested list item or line). "
                  "Answer from the supplied computed results and retrieved project notes only. Do not invent results. "
                  "If evidence is missing, say so. Distinguish predictions from ground truth and grouped OOF results from a locked test. "
                  "Format requested lists, tables, or line counts in Markdown. If the user asks about the dataset, explain the four modalities and distinguish original dataset size from prepared Phase 1 evaluation size. "
                  "Use the previous conversation only to resolve references such as 'that' or 'tell me more'; otherwise prioritize the latest question.\n\n"
                  f"Evidence JSON:\n{json.dumps(context, default=str)[:18_000]}\n\n"
                  f"Recent conversation:\n{recent}\n\nQuestion: {question}")
        response = client.models.generate_content(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), contents=prompt)
        answer = (response.text or "").strip()
        if answer:
            llm_calls.append(now)
            return answer, "Gemini explained the computed results and retrieved project notes."
    except Exception as exc:
        return None, f"Gemini is temporarily unavailable ({type(exc).__name__}); showing the local evidence-based answer."
    return None, "Gemini returned an empty answer; showing the local evidence-based answer."


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": "MultiPENG public research assistant",
            "answer_engine": "structured-evidence-v2",
            "gemini_configured": bool(os.getenv("GEMINI_API_KEY", "").strip())}


@app.get("/api/reference")
async def reference() -> dict[str, Any]:
    return REFERENCE


@app.get("/api/plots/{filename}")
async def plot(filename: str) -> FileResponse:
    if filename not in PLOT_FILES:
        raise HTTPException(status_code=404, detail="Plot not found")
    path = PLOTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Saved plot not available")
    return FileResponse(path, media_type="image/png")


@app.get("/api/notebook")
async def notebook_index() -> dict[str, Any]:
    data = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = []
    number = 0
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        number += 1
        source = "".join(cell.get("source", []))
        first = next((line.strip().lstrip("# ") for line in source.splitlines() if line.strip()), "Code")
        cells.append({"index": number, "title": first[:100]})
    return {"cells": cells}


@app.get("/api/notebook/{cell_index}")
async def notebook_cell(cell_index: int) -> dict[str, Any]:
    data = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = [cell for cell in data.get("cells", []) if cell.get("cell_type") == "code"]
    if cell_index < 1 or cell_index > len(cells):
        raise HTTPException(status_code=404, detail="Notebook cell not found")
    cell = cells[cell_index - 1]
    output_text: list[str] = []
    images: list[str] = []
    for output in cell.get("outputs", []):
        if output.get("text"):
            output_text.append("".join(output["text"]) if isinstance(output["text"], list) else str(output["text"]))
        payload = output.get("data", {})
        plain = payload.get("text/plain")
        if plain:
            output_text.append("".join(plain) if isinstance(plain, list) else str(plain))
        image = payload.get("image/png")
        if image:
            images.append("".join(image) if isinstance(image, list) else str(image))
    return {"source": "".join(cell.get("source", [])), "output": "\n".join(output_text), "images": images[:2]}


@app.post("/api/upload-predictions")
async def upload_predictions(file: UploadFile = File(...)) -> dict[str, Any]:
    expire_sessions()
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Prediction CSV must be no larger than 20 MB.")
    try:
        frame = parse_prediction_csv(payload)
        summary = summarize(frame)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if len(sessions) >= 50:
        expire_sessions()
        if len(sessions) >= 50:
            raise HTTPException(status_code=429, detail="The temporary analysis workspace is busy. Try again shortly.")
    frame_bytes = int(frame.memory_usage(index=True, deep=True).sum())
    cached_bytes = sum(int(item["frame"].memory_usage(index=True, deep=True).sum()) for item in sessions.values())
    if frame_bytes > 30 * 1024 * 1024 or cached_bytes + frame_bytes > 120 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="The expanded CSV is too large for the public temporary workspace. Reduce rows or columns and upload again.")
    dataset_id = uuid.uuid4().hex
    sessions[dataset_id] = {"created": time.time(), "frame": frame, "summary": summary}
    return {"dataset_id": dataset_id, "summary": summary, "message": "Uploaded data is held temporarily in memory and expires after 30 minutes."}


@app.get("/api/dataset/{dataset_id}/records")
async def dataset_records(dataset_id: str, limit: int = 100) -> dict[str, Any]:
    item = sessions.get(dataset_id)
    if not item:
        raise HTTPException(status_code=404, detail="This temporary dataset session expired. Upload the CSV again.")
    frame = item["frame"].head(max(1, min(limit, 250)))
    visible = [column for column in ["participant_id", "session_id", "timestamp", "actual_label", "predicted_label"] if column in frame]
    clean = frame[visible].copy()
    if "timestamp" in clean:
        clean["timestamp"] = clean["timestamp"].astype(str)
    clean = clean.rename(columns={"actual_label": "actual_engagement", "predicted_label": "predicted_engagement"})
    return {"columns": clean.columns.tolist(), "rows": clean.replace({np.nan: None}).to_dict(orient="records")}


@app.get("/api/dataset/{dataset_id}/export")
async def dataset_export(dataset_id: str) -> JSONResponse:
    item = sessions.get(dataset_id)
    if not item:
        raise HTTPException(status_code=404, detail="This temporary dataset session expired. Upload the CSV again.")
    return JSONResponse(item["summary"])


@app.post("/api/chat")
async def chat(payload: ChatRequest) -> dict[str, Any]:
    question = payload.question.strip()
    item = sessions.get(payload.dataset_id or "")
    summary = item["summary"] if item else None
    routed_question = resolve_follow_up(question, payload.history)
    sources = retrieve(routed_question)
    answer, used = deterministic_answer(routed_question, summary, sources)
    if routed_question != question:
        used.append("conversation follow-up context")
    mode = "Local evidence-based answer; arithmetic and data summaries are calculated by the app."
    if payload.use_gemini:
        generated, status = await run_in_threadpool(gemini_answer, question, summary, sources, payload.history)
        if generated:
            answer, mode = generated, status
        else:
            mode = status
    return {"answer": answer, "analysis_steps": used, "sources": sources, "answer_source": mode,
            "chart": build_chat_chart(question, summary), "dataset_summary": summary}


def build_chat_chart(question: str, summary: dict[str, Any] | None) -> dict[str, Any] | None:
    q = question.lower()
    if not any(word in q for word in ["plot", "chart", "graph", "visual"]):
        return None
    if not summary:
        counts = REFERENCE.get("class_counts_actual", {})
        labels = ["Low", "Medium", "High"]
        vals = [counts.get("Low (0/1)", 97), counts.get("Medium (2)", 156), counts.get("High (3/4)", 620)]
    else:
        labels = CLASS_NAMES
        vals = [summary["class_counts"].get(label, 0) for label in labels]
    return {"type": "bar", "labels": labels, "values": vals, "title": "Engagement class counts"}


@app.post("/api/phase1/run")
async def start_phase1(request: Request, eeg: UploadFile = File(...), eye: UploadFile = File(...), face: UploadFile = File(...), hr: UploadFile = File(...)) -> dict[str, str]:
    expire_sessions()
    if any(job.get("status") == "running" for job in phase1_jobs.values()):
        raise HTTPException(status_code=429, detail="A Phase 1 run is already active. Please wait for it to finish.")
    files = {"EEG": eeg, "Eye": eye, "Face": face, "HR": hr}
    contents: dict[str, bytes] = {}
    total = 0
    for name, upload in files.items():
        raw = await upload.read(MAX_UPLOAD_BYTES + 1)
        total += len(raw)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"{name} CSV must be no larger than 20 MB.")
        contents[name] = raw
    if total > 60 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Combined modality upload must be no larger than 60 MB.")
    for name, raw in contents.items():
        try:
            preview = pd.read_csv(io.BytesIO(raw), low_memory=False)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read {name} CSV: {exc}") from exc
        if len(preview) > MAX_PHASE1_ROWS:
            raise HTTPException(status_code=413, detail=f"{name} CSV exceeds the {MAX_PHASE1_ROWS:,}-session public-run limit.")
        if preview.shape[1] - 3 > MAX_PHASE1_FEATURES:
            raise HTTPException(status_code=413, detail=f"{name} CSV exceeds the {MAX_PHASE1_FEATURES:,}-feature public-run limit.")
    wrappers = {name: type("UploadBytes", (), {"getvalue": lambda self, b=raw: b})() for name, raw in contents.items()}
    job_id = uuid.uuid4().hex
    phase1_jobs[job_id] = {"status": "running", "progress": "Validating the four modality files…", "created": time.time()}
    asyncio.create_task(execute_phase1_job(job_id, wrappers))
    return {"job_id": job_id, "message": "The grouped Phase 1 run has started."}


async def execute_phase1_job(job_id: str, wrappers: dict[str, Any]) -> None:
    def update(message: str) -> None:
        if job_id in phase1_jobs:
            phase1_jobs[job_id]["progress"] = message
    try:
        result = await run_in_threadpool(run_phase1_from_uploads, wrappers, update)
        result["predictions_csv"] = result["predictions"].to_csv(index=False)
        result["predictions"] = result["predictions"].head(0).to_dict(orient="records")
        phase1_jobs[job_id].update({"status": "complete", "progress": "The grouped run finished.", "result": result, "finished_at": time.time()})
    except Exception as exc:
        phase1_jobs[job_id].update({"status": "error", "progress": str(exc), "finished_at": time.time()})


@app.get("/api/phase1/status/{job_id}")
async def phase1_status(job_id: str) -> dict[str, Any]:
    job = phase1_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Run not found or expired.")
    result = job.get("result")
    safe_result = None
    if result:
        safe_result = {key: value for key, value in result.items() if key not in {"predictions_csv", "predictions"}}
        for key in ["fold_table", "search_table", "class_table"]:
            if key in safe_result and hasattr(safe_result[key], "to_dict"):
                safe_result[key] = safe_result[key].replace({np.nan: None}).to_dict(orient="records")
    return {"status": job["status"], "progress": job["progress"], "result": safe_result}


@app.get("/api/phase1/download/{job_id}")
async def phase1_download(job_id: str) -> JSONResponse:
    job = phase1_jobs.get(job_id)
    if not job or job.get("status") != "complete":
        raise HTTPException(status_code=404, detail="Completed predictions are not available.")
    return JSONResponse({"csv": job["result"]["predictions_csv"]})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
