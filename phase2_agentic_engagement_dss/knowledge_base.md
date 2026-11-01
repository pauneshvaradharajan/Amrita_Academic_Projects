# MultiPENG project knowledge base

## Project purpose and four modalities
Project title: An Agentic AI Decision Support Framework for Intelligent Multimodal Engagement Analytics using Deep Learning, RAG, and Large Language Models. Phase 1 predicts a player's engagement using synchronized EEG, eye tracking, facial behavior/OpenFace, and heart rate features. Phase 2 explains Phase 1 output to a researcher in plain English, shows evidence and charts, and supports follow-up questions. The Phase 2 app does not train the Phase 1 classifier or read raw sensor matrices.

## Engagement labels and class imbalance
The current grouped task maps original labels 0 and 1 to Low, original label 2 to Medium, and original labels 3 and 4 to High. In grouped numeric columns, 0=Low, 1=Medium, and 2=High. These two encodings are different: grouped class 1 is Medium, but original label 1 is Low. The app must prefer an explicit grouped-label column when one is supplied. Historical data has substantially fewer Low and Medium examples than High. A high accuracy can therefore hide weak minority-class recall; interpret accuracy with Macro-F1, per-class recall, balanced accuracy, MCC, and a confusion matrix.

## Current Phase 1 results and validation
The current Phase 1 reference covers 873 sessions from 38 participants and is included with this app as aggregate actual/predicted class counts and four saved plots. It contains no session or participant identifiers. The target maps original labels 0/1 to Low, 2 to Medium, and 3/4 to High.

The experiment used nested 5-fold participant-grouped out-of-fold evaluation with 3 inner folds for candidate selection. It evaluated RBF-SVM and Random Forest with training-fold-only class weighting and random oversampling options. It used fold-fitted cleaning, imputation, scaling, and ANOVA selection of the top 50% of features per modality. The gated four-encoder MLP was defined but not executed in this CPU run. This is cross-validation evidence, not an untouched final test.

Pooled results: accuracy 65.75%, Macro-F1 0.371, Weighted-F1 0.614, balanced accuracy 0.372, and MCC 0.096. The majority-only accuracy reference is 71.02%; therefore the model did not improve accuracy over that reference. Macro-F1 and balanced accuracy are higher than the majority reference, but Low recall is 0.134, Medium recall is 0.103, and High recall is 0.879. The main observed limitation remains weak minority-class recognition.

The app should calculate metrics from the current aggregate confusion counts or an uploaded prediction file and distinguish this participant-grouped out-of-fold estimate from an independent test score. The earlier 60.1% grouped SVM and 73.8% session-random results are historical and must not be presented as the current result.

## What the evaluation metrics mean
Accuracy is the fraction of all sessions predicted correctly. Macro-F1 computes F1 for each class and averages the classes equally, so it highlights poor minority-class performance. Weighted-F1 averages class F1 weighted by the number of examples. Recall tells how many actual examples of a class the model finds. Balanced accuracy averages class recall. MCC summarizes agreement between actual and predicted labels from -1 to +1, with 0 near chance. The confusion matrix has actual classes as rows and predictions as columns. Majority baseline accuracy is the score from always predicting the most frequent actual class on those same labeled rows. Model accuracy should be compared with that baseline.

## Validation and generalization
Sessions from the same participant can share personal patterns. For claims about new players, keep participants separate between training and validation. Session-random splits can leak participant-specific information and overstate generalization to unseen people. Use the protocol written in the uploaded result file where available. The Phase 2 app only reports the supplied predictions and labels; it cannot independently verify how the Phase 1 folds were created.

## Phase 2 answer workflow: analytics, RAG, and LLM
For each user question, the bounded assistant routes to relevant predefined tools such as class distribution, performance metrics, error analysis, file-quality checks, recommendations, and trend summaries. It computes tool results from the currently uploaded prediction CSV. RAG retrieves matching passages from this local project knowledge base using TF-IDF text similarity. If Gemini is enabled and GEMINI_API_KEY is configured, Gemini writes a natural-language explanation from aggregate computed results and retrieved passages. Raw records, participant IDs, and raw EEG/eye/face/HR features must not be sent in the prompt. If no key exists or the API call fails, use the local data-grounded assistant and tell the user which response mode was used. The workflow is tool-assisted and bounded; do not describe it as an unrestricted autonomous agent.

## CSV upload format and label safety
The app requires a `predicted_engagement` column. The Phase 1 export may also contain `actual_engagement_grouped_0_2`, `actual_engagement_original_0_4`, `participant_id`, `session_id`, `prediction_method`, `prediction_protocol`, and other metadata. The grouped actual label is the preferred ground truth for three-class evaluation. The original label is mapped using 0/1=Low, 2=Medium, 3/4=High. Predictions alone do not provide accuracy: actual labels are required. Missing confidence, dates, or game metadata must never be invented.

## Decision-support limits and privacy
Engagement estimates are model outputs, not direct measurements of a player's internal state. They do not prove causation, diagnose a health condition, or justify a high-impact decision about a person. Point out missing data, uncertainty, imbalance, and weak per-class results. The app's uploaded prediction file remains in the app session and is not written to its repository or database. Do not upload raw sensor data or identifiable participant information to the public app.
