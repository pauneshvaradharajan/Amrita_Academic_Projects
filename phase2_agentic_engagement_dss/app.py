from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


APP_DIR = Path(__file__).parent
KB_PATH = APP_DIR / "knowledge_base.md"
CLASSES = ["Low", "Medium", "High"]
COLORS = {"Low": "#4C78A8", "Medium": "#F2A541", "High": "#59A14F"}

st.set_page_config(page_title="MultiPENG Engagement Decision Support", page_icon="🧠", layout="wide")


def make_demo_data(seed: int = 42) -> pd.DataFrame:
    """Create clearly labelled fictional sessions for a safe first-time demo."""
    rng = np.random.default_rng(seed)
    labels = np.array(["Low"] * 10 + ["Medium"] * 16 + ["High"] * 64)
    rng.shuffle(labels)
    rows = []
    for i, label in enumerate(labels):
        participant = f"Demo-P{i % 18 + 1:02d}"
        session_number = i // 18 + 1
        rows.append({
            "participant_id": participant,
            "session_id": f"D{i + 1:03d}",
            "session_number": session_number,
            "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(days=int(i)),
            "game": "Fictional Demo Game",
            "difficulty": ["Easy", "Medium", "Hard"][i % 3],
            "predicted_engagement": label,
            "confidence": float(rng.uniform(0.55, 0.98)),
        })
    return pd.DataFrame(rows)


def canonical_class(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    if text in {"low", "0", "1", "class0", "class1", "low01"}:
        return "Low"
    if text in {"medium", "mid", "2", "class2"}:
        return "Medium"
    if text in {"high", "3", "4", "class3", "class4", "high34"}:
        return "High"
    raise ValueError(f"Unknown engagement value: {value!r}. Use Low/Medium/High or numeric labels 0–4.")


def load_prediction_csv(file) -> pd.DataFrame:
    raw = pd.read_csv(file)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    aliases = {
        "participant_id": ["participant_id", "participant", "subject_id", "player_id"],
        "session_id": ["session_id", "submission_id", "session", "submission"],
        "predicted_engagement": ["predicted_engagement", "predicted_class", "prediction", "engagement_class", "engagement"],
        "timestamp": ["timestamp", "datetime", "session_time", "time"],
        "game": ["game", "game_name"],
        "difficulty": ["difficulty", "difficulty_level"],
        "confidence": ["confidence", "prediction_confidence", "max_probability"],
    }
    rename = {}
    for canonical, options in aliases.items():
        found = next((name for name in options if name in raw.columns), None)
        if found and found != canonical:
            rename[found] = canonical
    raw = raw.rename(columns=rename)
    if "predicted_engagement" not in raw.columns:
        raise ValueError("CSV needs a predicted_engagement, prediction, engagement_class, or engagement column.")
    raw["predicted_engagement"] = raw["predicted_engagement"].map(canonical_class)
    if "participant_id" not in raw.columns:
        raw["participant_id"] = "Unknown"
    if "session_id" not in raw.columns:
        raw["session_id"] = [f"S{i + 1:04d}" for i in range(len(raw))]
    if "confidence" in raw.columns:
        raw["confidence"] = pd.to_numeric(raw["confidence"], errors="coerce")
        if raw["confidence"].dropna().gt(1).any():
            raw["confidence"] = raw["confidence"] / 100.0
        raw["confidence"] = raw["confidence"].clip(0, 1)
    if "timestamp" in raw.columns:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    return raw


def summarize_data(df: pd.DataFrame) -> dict:
    counts = df["predicted_engagement"].value_counts().reindex(CLASSES, fill_value=0)
    result = {
        "sessions": int(len(df)),
        "participants": int(df["participant_id"].nunique()),
        "class_counts": {k: int(v) for k, v in counts.items()},
        "class_percent": {k: round(float(v / max(len(df), 1) * 100), 1) for k, v in counts.items()},
    }
    if "confidence" in df and df["confidence"].notna().any():
        result["mean_confidence"] = round(float(df["confidence"].mean()), 3)
    if "timestamp" in df and df["timestamp"].notna().any():
        result["date_range"] = [str(df["timestamp"].min().date()), str(df["timestamp"].max().date())]
    return result


def create_report(df: pd.DataFrame, summary: dict) -> str:
    lines = [
        "# MultiPENG Engagement Summary",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Sessions analyzed: {summary['sessions']}",
        f"Participants represented: {summary['participants']}",
        "",
        "## Predicted engagement distribution",
    ]
    for name in CLASSES:
        lines.append(f"- {name}: {summary['class_counts'][name]} ({summary['class_percent'][name]}%)")
    if "mean_confidence" in summary:
        lines.extend(["", f"Mean model confidence: {summary['mean_confidence']:.3f}"])
    lines.extend(["", "These are summaries of model predictions, not direct measurements of a person's internal state."])
    return "\n".join(lines)


@st.cache_data(show_spinner=False)
def load_knowledge() -> list[str]:
    text = KB_PATH.read_text(encoding="utf-8")
    return [chunk.strip() for chunk in re.split(r"\n(?=## )", text) if chunk.strip()]


@st.cache_resource(show_spinner=False)
def make_retriever(chunks: tuple[str, ...]):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, matrix


def retrieve_context(question: str, k: int = 3) -> list[tuple[str, float]]:
    chunks = load_knowledge()
    vectorizer, matrix = make_retriever(tuple(chunks))
    q = vectorizer.transform([question])
    scores = cosine_similarity(q, matrix).ravel()
    order = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in order if scores[i] > 0]


def run_analysis_tool(tool_name: str, df: pd.DataFrame) -> dict:
    summary = summarize_data(df)
    if tool_name == "participant_summary":
        counts = df.groupby(["participant_id", "predicted_engagement"]).size().unstack(fill_value=0)
        counts = counts.reindex(columns=CLASSES, fill_value=0)
        counts["sessions"] = counts.sum(axis=1)
        return {
            "tool": tool_name,
            "participants": int(len(counts)),
            "class_session_totals": {name: int(counts[name].sum()) for name in CLASSES},
            "note": "Participant identifiers and individual rows are excluded from the language-model context.",
        }
    if tool_name == "time_trend" and "timestamp" in df and df["timestamp"].notna().any():
        dated = df.dropna(subset=["timestamp"]).copy()
        dated["date"] = dated["timestamp"].dt.date.astype(str)
        trend = dated.groupby(["date", "predicted_engagement"]).size().unstack(fill_value=0)
        trend = trend.reindex(columns=CLASSES, fill_value=0).tail(30)
        return {"tool": tool_name, "daily_counts": trend.reset_index().to_dict("records")}
    return {"tool": "class_distribution", "summary": summary}


def choose_tools(question: str, df: pd.DataFrame) -> list[str]:
    q = question.lower()
    tools = []
    if any(x in q for x in ["participant", "player", "person", "individual"]):
        tools.append("participant_summary")
    if any(x in q for x in ["trend", "over time", "timeline", "change", "date"]):
        if "timestamp" in df and df["timestamp"].notna().any():
            tools.append("time_trend")
    if any(x in q for x in ["class", "distribution", "engagement", "summary", "overall", "compare", "report"]) or not tools:
        tools.append("class_distribution")
    return list(dict.fromkeys(tools))


def deterministic_answer(question: str, results: list[dict], context: list[tuple[str, float]], df: pd.DataFrame) -> str:
    summary = summarize_data(df)
    distribution = ", ".join(f"{k}: {summary['class_counts'][k]} ({summary['class_percent'][k]}%)" for k in CLASSES)
    pieces = [f"I analyzed {summary['sessions']} sessions from {summary['participants']} participants. Predicted classes: {distribution}."]
    for result in results:
        if result["tool"] == "participant_summary":
            pieces.append("The participant analysis uses anonymous totals and does not expose individual identities.")
        if result["tool"] == "time_trend":
            pieces.append("A date-based count table was prepared from rows with valid timestamps.")
    if context:
        pieces.append("Project note: " + context[0][0].splitlines()[0].lstrip("# ") + ".")
    pieces.append("Interpret predictions as model estimates, not ground truth or clinical conclusions.")
    return " ".join(pieces)


def sanitize_question(question: str, df: pd.DataFrame) -> str:
    safe_question = question
    for column in ["participant_id", "session_id"]:
        if column in df:
            for identifier in df[column].dropna().astype(str).unique():
                if identifier and identifier.lower() != "unknown":
                    safe_question = re.sub(rf"(?<!\w){re.escape(identifier)}(?!\w)", "[record]", safe_question, flags=re.IGNORECASE)
    return safe_question


def llm_answer(question: str, results: list[dict], context: list[tuple[str, float]], df: pd.DataFrame, allow_llm: bool) -> tuple[str, str]:
    api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    if not allow_llm:
        return deterministic_answer(question, results, context, df), "Gemini is off; no user question or uploaded data was sent to an external LLM."
    if not api_key:
        return deterministic_answer(question, results, context, df), "No GEMINI_API_KEY is configured; showing the data-grounded explanation without an LLM."
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        kb = "\n\n".join(f"[Source {i + 1}] {chunk}" for i, (chunk, _) in enumerate(context)) or "No matching project source was retrieved."
        payload = json.dumps(results, ensure_ascii=False, default=str)
        prompt = f"""You are a careful research decision-support assistant for a multimodal player engagement project.
Answer the user's question in plain English. Use only the computed result JSON and retrieved project notes below.
Do not claim causation, clinical status, or that predictions are ground truth. If information is absent, say so.
For background claims, cite source headings exactly as [Source 1], etc. Clearly distinguish predicted classes from true labels.

User question: {sanitize_question(question, df)}

Computed aggregate results (authoritative for counts; contains no session rows or identifiers):
{payload}

Retrieved project notes:
{kb}
"""
        response = client.models.generate_content(
            model=st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=700),
        )
        answer = (response.text or "").strip()
        if not answer:
            raise RuntimeError("The language model returned an empty answer.")
        return answer, "Gemini generated this explanation from aggregate results and retrieved project notes."
    except Exception as exc:
        return deterministic_answer(question, results, context, df), f"LLM unavailable ({type(exc).__name__}); used the data-grounded fallback."


def run_agent(question: str, df: pd.DataFrame, allow_llm: bool) -> tuple[str, list[str], list[tuple[str, float]], str]:
    selected = choose_tools(question, df)
    results = [run_analysis_tool(name, df) for name in selected]
    context = retrieve_context(question)
    answer, status = llm_answer(question, results, context, df, allow_llm)
    return answer, selected, context, status


st.markdown("""
<style>
.stApp{background:#fff;color:#202123}[data-testid="stHeader"]{background:rgba(255,255,255,.94)}[data-testid="stSidebar"]{background:#f7f7f8;border-right:1px solid #e7e7e8}[data-testid="stSidebar"]>div:first-child{padding-top:1.1rem}.block-container{max-width:920px;padding-top:2rem;padding-bottom:7rem}.brand{font-size:1.05rem;font-weight:700;letter-spacing:-.02em;margin:.1rem 0 .15rem}.brand-sub{color:#777b82;font-size:.78rem;margin-bottom:1.2rem}.welcome{padding:3.2rem 1rem 1.6rem;text-align:center}.welcome h1{font-size:2.1rem;letter-spacing:-.045em;font-weight:650;color:#202123;margin-bottom:.45rem}.welcome p{color:#777b82;font-size:1rem}.stChatMessage{border:0;padding-top:.75rem;padding-bottom:.75rem}[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"]{line-height:1.65}[data-testid="stChatInput"]{border:1px solid #d9dadd;border-radius:24px;box-shadow:0 3px 12px rgba(0,0,0,.06)}[data-testid="stChatInput"] textarea{padding:.9rem 1rem}.stButton button{border-radius:18px}.block-container div[data-testid="stHorizontalBlock"] button{background:#fff!important;color:#35373b!important;border:1px solid #e5e5e6!important;min-height:4.5rem;white-space:normal;text-align:left}[data-testid="stMetric"]{background:#fafafa;border:1px solid #ededee;border-radius:13px;padding:.75rem .9rem}.demo-note{background:#fff8e6;border:1px solid #f0dfb3;border-radius:10px;padding:.65rem .8rem;color:#66551f;font-size:.82rem}.footnote{color:#8a8d92;font-size:.76rem;text-align:center;padding:1rem 0 0}div[data-testid="stExpander"]{border:1px solid #e9e9ea;border-radius:12px}

/* High-contrast text for the white workspace */
.stApp, .stApp p, .stApp span, .stApp label, .stApp li, .stApp small,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stText"], .stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stMetricLabel"], .stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricDelta"], .stApp [data-testid="stExpander"] summary,
.stApp [data-testid="stTabs"] button, .stApp [role="radiogroup"],
.stApp [role="radiogroup"] *, .stApp [role="switch"] { color:#111 !important; }
.brand-sub, .welcome p { color:#222 !important; }
.welcome h1 { color:#111 !important; }
.block-container div[data-testid="stHorizontalBlock"] button,
.block-container div[data-testid="stHorizontalBlock"] button * { color:#111 !important; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * { color:#111 !important; }
[data-testid="stChatInput"] textarea { color:#111 !important; caret-color:#111 !important; }
[data-testid="stChatInput"] textarea::placeholder { color:#444 !important; opacity:1 !important; }
.stApp input, .stApp textarea { color:#111 !important; }
.footnote { color:#333 !important; }

</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="brand">◉ MultiPENG</div><div class="brand-sub">Engagement research assistant</div>', unsafe_allow_html=True)
    if st.button("＋  New conversation", use_container_width=True):
        st.session_state.pop("chat_history", None)
        st.session_state.pop("suggested_question", None)
        st.rerun()
    st.divider()
    st.markdown("**Workspace data**")
    data_mode = st.radio("Data source", ["Fictional demo data", "Upload Phase 1 predictions"], index=0, label_visibility="collapsed")
    if data_mode == "Upload Phase 1 predictions":
        uploaded = st.file_uploader("Prediction summary CSV", type=["csv"], help="Upload session-level predictions only. Do not upload raw EEG, face, eye, or HR recordings to this public demo.")
        st.caption("Uploaded data is used only for this app session.")
    else:
        uploaded = None
    allow_llm = st.toggle("Enable Gemini responses", value=False, help="When enabled, your question, aggregate counts and retrieved project notes are sent to Google Gemini. Session rows, participant IDs and raw sensor features are excluded.")
    st.caption("Local analytics and retrieval work when Gemini is off.")

try:
    if data_mode == "Upload Phase 1 predictions" and uploaded is not None:
        data = load_prediction_csv(uploaded)
        st.session_state["dataset"] = data
        st.session_state["is_demo"] = False
    elif data_mode == "Upload Phase 1 predictions" and "dataset" in st.session_state and not st.session_state.get("is_demo", True):
        data = st.session_state["dataset"]
    else:
        data = make_demo_data()
        st.session_state["dataset"] = data
        st.session_state["is_demo"] = True
except Exception as exc:
    st.error(f"Could not load the prediction file: {exc}")
    st.stop()

summary = summarize_data(data)
if st.session_state.get("is_demo", False):
    st.sidebar.markdown('<div class="demo-note">Demo workspace uses fictional sessions, not actual MultiPENG participant results.</div>', unsafe_allow_html=True)

history = st.session_state.get("chat_history", [])
if not history:
    st.markdown('<div class="welcome"><h1>What would you like to understand?</h1><p>Ask about engagement patterns, model outputs, or the evidence behind them.</p></div>', unsafe_allow_html=True)
    suggestions = ["Summarize the engagement distribution", "Which groups or classes need attention?", "Explain the limitations of these predictions"]
    cols = st.columns(3)
    for i, (col, prompt) in enumerate(zip(cols, suggestions)):
        with col:
            if st.button(prompt, key=f"suggestion_{i}", use_container_width=True):
                st.session_state["suggested_question"] = prompt
                st.rerun()
else:
    st.markdown("### ◉ MultiPENG research assistant")

question = st.chat_input("Message MultiPENG research assistant…")
question = st.session_state.pop("suggested_question", None) or question
if question:
    history = st.session_state.setdefault("chat_history", [])
    history.append({"role": "user", "content": question})
    with st.spinner("Reviewing session data and searching project sources…"):
        answer, tools_used, sources, llm_status = run_agent(question, data, allow_llm)
    history.append({"role": "assistant", "content": answer, "tools": tools_used, "sources": sources, "status": llm_status})

for message in st.session_state.get("chat_history", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            with st.expander("Evidence, sources and analysis steps"):
                st.caption("Analysis tools: " + (", ".join(message.get("tools", [])) or "none"))
                st.caption("Response mode: " + message.get("status", "Data-grounded local response."))
                if message.get("sources"):
                    for j, (chunk, score) in enumerate(message["sources"], 1):
                        title = chunk.splitlines()[0].lstrip("# ")
                        with st.expander(f"Source {j}: {title} · relevance {score:.2f}"):
                            st.markdown(chunk)
                else:
                    st.write("No project passage matched this question. The answer uses computed session summaries only.")

with st.expander("Explore analytics, records and exports"):
    metric_cols = st.columns(4)
    metric_cols[0].metric("Sessions", f"{summary['sessions']:,}")
    metric_cols[1].metric("Participants", f"{summary['participants']:,}")
    metric_cols[2].metric("Most common prediction", max(summary["class_counts"], key=summary["class_counts"].get))
    metric_cols[3].metric("Mean confidence", f"{summary['mean_confidence']:.1%}" if "mean_confidence" in summary else "Not supplied")
    chart_tab, records_tab, export_tab = st.tabs(["Patterns", "Prediction records", "Export"])
    with chart_tab:
        left, right = st.columns(2)
        with left:
            counts = pd.DataFrame({"Engagement": CLASSES, "Sessions": [summary["class_counts"][c] for c in CLASSES]})
            fig = px.bar(counts, x="Engagement", y="Sessions", color="Engagement", color_discrete_map=COLORS, title="Predicted engagement classes")
            fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width="stretch")
        with right:
            if "timestamp" in data and data["timestamp"].notna().any():
                by_day = data.dropna(subset=["timestamp"]).assign(Date=lambda x: x["timestamp"].dt.date.astype(str)).groupby(["Date", "predicted_engagement"]).size().reset_index(name="Sessions")
                fig = px.line(by_day, x="Date", y="Sessions", color="predicted_engagement", markers=True, color_discrete_map=COLORS, title="Predictions over time")
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Add a timestamp column to the prediction CSV to see time trends.")
    with records_tab:
        visible_cols = [c for c in ["participant_id", "session_id", "session_number", "timestamp", "game", "difficulty", "predicted_engagement", "confidence"] if c in data.columns]
        st.dataframe(data[visible_cols], width="stretch", hide_index=True)
    with export_tab:
        report = create_report(data, summary)
        st.download_button("Download analysis summary", report, file_name="multipeng_engagement_summary.md", mime="text/markdown")
        template = pd.DataFrame(columns=["participant_id", "session_id", "predicted_engagement", "confidence", "timestamp", "game", "difficulty"])
        st.download_button("Download prediction CSV template", template.to_csv(index=False), file_name="phase1_prediction_template.csv", mime="text/csv")
        st.caption("This app summarizes stored predictions. It does not train or change the Phase 1 classifier.")

st.markdown('<div class="footnote">Research decision support only · predictions can be wrong · not for clinical or high-impact decisions</div>', unsafe_allow_html=True)
