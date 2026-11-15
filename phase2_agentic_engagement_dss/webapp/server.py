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
    q = question.lower()
    tools: list[str] = []
    if not summary:
        summary = {"rows": REFERENCE["sessions"], "participants": REFERENCE["participants"],
                   "class_counts": {"Low": 97, "Medium": 156, "High": 620}}
        tools.append("saved Phase 1 aggregate results")
    if any(k in q for k in ["column", "schema", "field", "feature"]):
        cols = summary.get("columns", ["actual_engagement", "predicted_engagement", "session_count"])
        answer = "The available columns are:\n\n" + "\n".join(f"- `{c}`" for c in cols)
        tools.append("CSV schema")
    elif any(k in q for k in ["confusion", "mistake", "error", "misclassif"]):
        if summary.get("evaluation"):
            cm = summary["evaluation"]["confusion_matrix"]
            rows = ["| Actual \\ Predicted | Low | Medium | High |", "|---|---:|---:|---:|"]
            rows += [f"| {CLASS_NAMES[i]} | {cm[i][0]} | {cm[i][1]} | {cm[i][2]} |" for i in range(3)]
            answer = "Confusion matrix (rows are actual classes; columns are predicted classes):\n\n" + "\n".join(rows)
            tools.append("confusion matrix")
        else:
            answer = "The saved Phase 1 run reports an out-of-fold confusion matrix. It shows the strongest recognition for High sessions; Low and Medium are more often confused. Use the saved grouped OOF plot to inspect counts. This is cross-validation evidence, not a separate test-set result."
            tools.append("saved Phase 1 confusion matrix")
    elif any(k in q for k in ["accuracy", "macro", "f1", "performance", "metric", "score"]):
        evaluation = summary.get("evaluation")
        if evaluation:
            answer = (f"From {evaluation['n_labeled']:,} rows with actual labels: accuracy **{evaluation['accuracy']:.1%}**, Macro-F1 **{evaluation['macro_f1']:.3f}**, Weighted-F1 **{evaluation['weighted_f1']:.3f}**, balanced accuracy **{evaluation['balanced_accuracy']:.3f}**, and MCC **{evaluation['mcc']:.3f}**. The majority-only accuracy baseline is **{evaluation['majority_baseline_accuracy']:.1%}**. These are calculated from the uploaded labels and predictions.")
            tools.append("local metric calculation")
        else:
            m = REFERENCE["overall_metrics"]
            b = REFERENCE["majority_reference_metrics"]
            answer = (f"The saved participant-grouped nested cross-validation run reports **{m['Accuracy']:.1%} accuracy**, **{m['Macro-F1']:.3f} Macro-F1**, and **{m['Weighted-F1']:.3f} Weighted-F1**. The majority-only accuracy reference is **{b['Accuracy']:.1%}**. This is out-of-fold validation across the 873 sessions, not a locked test-set result; Low and Medium recall are weak.")
            tools.append("saved Phase 1 metrics")
    elif any(k in q for k in ["distribution", "class count", "imbalance", "majority"]):
        counts = summary["class_counts"]
        total = max(sum(counts.values()), 1)
        answer = f"This file contains **{summary['rows']:,} rows**. Predicted engagement counts are:\n\n" + "\n".join(f"- **{name}:** {count:,} ({count/total:.1%})" for name, count in counts.items())
        answer += "\n\nThese are predicted class counts; they do not alone establish model accuracy."
        tools.append("class distribution calculation")
    elif any(k in q for k in ["duplicate", "missing", "quality", "data type"]):
        answer = f"The uploaded file has **{summary['rows']:,} rows**, **{summary['missing_cells']:,} missing cells**, and **{summary['duplicate_rows']:,} completely duplicated rows**. Session uniqueness is checked only when the file includes a session identifier."
        tools.append("CSV quality checks")
    else:
        if sources:
            answer = "Here is the project evidence that best matches your question:\n\n" + "\n\n".join(f"**{item['title']}**\n{item['text']}" for item in sources[:2])
            tools.append("local RAG retrieval from knowledge_base.md")
        else:
            answer = "I could not find enough project evidence to answer that reliably. Try asking about the dataset columns, class distribution, model metrics, confusion matrix, validation method, or preprocessing."
    return answer, tools


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
                  "Answer from the supplied computed results and retrieved project notes only. Do not invent results. "
                  "If evidence is missing, say so. Distinguish predictions from ground truth and grouped OOF results from a locked test. "
                  "Format requested lists, tables, or line counts in Markdown. Explain simply.\n\n"
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
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "MultiPENG public research assistant"}


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
    sources = retrieve(question)
    answer, used = deterministic_answer(question, summary, sources)
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
