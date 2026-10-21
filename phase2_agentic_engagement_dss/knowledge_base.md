# MultiPENG project knowledge base

## Project purpose and modalities
The project studies player engagement during gameplay using four synchronized modalities: EEG, eye tracking, facial behavior, and heart rate. Phase 2 is a decision-support layer that summarizes predictions and explains them in plain language. The AI analyst must describe predictions as model outputs, not direct proof of a player's internal state.

## Current Phase 1 validation context
The current SVM experiment maps original labels 0 and 1 to Low, label 2 to Medium, and labels 3 and 4 to High. Its prepared files contain 873 paired sessions from 38 participants. Participant-grouped nested five-fold cross-validation is the main estimate for unseen participants; session-random folds are a secondary estimate for additional sessions from represented participants. The reported grouped accuracy was 60.1% and Macro-F1 was 0.372. The session-random accuracy was 73.8% and Macro-F1 was 0.498. About 71% of sessions are High, so accuracy alone is not enough to assess performance. Low and Medium classes remain difficult. These figures describe the current experiment and should be updated if the model or data changes.

## Validation and interpretation
Repeated sessions from one participant are related. When evaluating a model for new players, keep each participant entirely in either training or validation folds. Session-random validation can share participant-specific patterns across folds and may give a higher score. Report accuracy together with Macro-F1, per-class precision and recall, and the confusion matrix. Do not tune model choices on the final test set.

## Decision-support limits
The app reports counts and summaries of supplied predictions. It does not infer causation, diagnose a health condition, or prove that a person was engaged. If a confidence value is absent, the app must not invent one. If timestamps or game metadata are absent, it must say that time or game comparisons cannot be made. Avoid exposing participant-level information in a public presentation unless the data is consented and appropriately de-identified.

## Agent workflow
For each user question, select only the available analysis tools, calculate values from the loaded session table, retrieve relevant project notes, then explain the results. The language model must not make up counts or claims that conflict with computed results. If no language-model key is configured, use the built-in data-grounded explanation and clearly state that the LLM is unavailable.

## Data format
The app accepts session-level prediction CSVs with a predicted_engagement column. Optional columns are participant_id, session_id, confidence, timestamp, game, and difficulty. Labels may be Low, Medium, High, or original numeric labels 0 through 4, which are rebinned using the project mapping. Raw EEG, eye, face, and HR feature matrices are not needed by this Phase 2 app.

