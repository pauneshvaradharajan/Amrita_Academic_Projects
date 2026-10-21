# MultiPENG Agentic Engagement Decision Support

This Streamlit app is the Phase 2 decision-support layer for the MultiPENG project. It summarizes session-level predictions, displays charts, retrieves project notes (RAG), and uses Gemini to explain computed results when an API key is configured. A bounded agent selects from predefined analysis tools; it does not alter labels or train the Phase 1 model.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in fictional demo mode. To use Phase 1 results, upload a session-level prediction CSV with `predicted_engagement`; optional columns are `participant_id`, `session_id`, `confidence`, `timestamp`, `game`, and `difficulty`.

## Enable Gemini explanations

Create `.streamlit/secrets.toml` locally:

```toml
GEMINI_API_KEY = "your-key"
GEMINI_MODEL = "gemini-2.5-flash"
```

Never commit this file or an API key. In Streamlit Community Cloud, add these values under the app's Secrets settings.

## Data handling

The public app starts with fictional sample sessions. Uploaded predictions are held in the Streamlit session and are not written to the repository or a database. Do not upload raw physiological recordings or identifiable participant data to a public deployment. For persistent real research records, connect a private, access-controlled database before enabling persistence.

## Current experiment context

The current Phase 1 SVM uses the mapping 0/1 → Low, 2 → Medium, and 3/4 → High, with participant-grouped nested five-fold validation as the primary unseen-participant estimate. The knowledge base contains the current reported results and should be updated when the model or dataset changes.

