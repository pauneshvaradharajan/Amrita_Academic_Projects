# MultiPENG Agentic Engagement Decision Support

This Streamlit app is the Phase 2 decision-support layer for the MultiPENG project. It summarizes session-level Phase 1 predictions, computes evaluation metrics when actual labels are present, displays charts, retrieves project notes (RAG), and can use Gemini to explain computed results when an API key is configured. A bounded agent routes questions to data-analysis tools and retrieves project context; it does not alter labels or train the Phase 1 model.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens on the current Phase 1 run. It loads a privacy-safe aggregate confusion-count file and the four exact plots saved from Phase 1, so you can ask questions without uploading the prepared sensor CSVs or the session-level prediction file. The current aggregate file contains nine actual/predicted class pairs and their counts; it contains no participant or session identifiers. You can switch to **Upload another Phase 1 CSV** for a different session-level prediction file. That file needs a `predicted_engagement` column; actual labels are needed for model evaluation.

For evaluation, include both predictions and actual labels. The Phase 1 export columns `actual_engagement_grouped_0_2` and `actual_engagement_original_0_4` are recognized. The grouped label is used for evaluation (0/1/2 = Low/Medium/High); the original 0–4 label is mapped with 0/1 = Low, 2 = Medium, and 3/4 = High. The app reports accuracy, Macro-F1, Weighted-F1, balanced accuracy, MCC, per-class scores, and a confusion matrix. If actual labels are absent, it will not claim to know model accuracy.

Gemini is off by default. With Gemini off, deterministic explanations use computed summaries and RAG results locally. With Gemini enabled, the app sends the current question, aggregate analysis results, and retrieved project notes to Google Gemini; session rows, participant IDs, and raw sensor features are excluded.

## Enable Gemini explanations

Create `.streamlit/secrets.toml` locally:

```toml
GEMINI_API_KEY = "your-key"
GEMINI_MODEL = "gemini-3.8-flash"
```

Never commit this file or an API key. In Streamlit Community Cloud, add these values under the app's Secrets settings.

## Data handling

The app does not create or display fictional sessions. The default Phase 1 reference stores aggregate class-pair counts only. Uploaded predictions are held in the Streamlit session and are not written to the repository or a database. Do not upload raw physiological recordings or identifiable participant data to a public deployment. For persistent real research records, connect a private, access-controlled database before enabling persistence.

## Current experiment context

The current Phase 1 model-selection run covers 873 sessions from 38 participants. Nested participant-grouped OOF metrics: 65.75% accuracy, 0.371 Macro-F1, 0.614 Weighted-F1, 0.372 balanced accuracy, and 0.096 MCC. Majority-only accuracy is 71.02%, so this run has not improved accuracy over the baseline; Low and Medium recall remain weak. It is not an independent test result. Update `phase1_reference.json`, the saved plots, and `knowledge_base.md` together whenever validated Phase 1 results change.
