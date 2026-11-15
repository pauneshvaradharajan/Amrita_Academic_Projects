from __future__ import annotations
import io
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
def run_phase1_from_uploads(uploaded_files: dict[str, object], progress=None) -> dict:
    """Run nested participant-grouped model selection on the four uploaded prepared tables."""
    modalities = ["EEG", "Eye", "Face", "HR"]
    keys = ["participant_id", "submission_id"]
    labels = ["engagement"]
    label_map = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
    stage_rows: list[dict[str, str]] = []

    def add_stage(name: str, code: str, output: str) -> None:
        stage_rows.append({"name": name, "code": code.strip(), "output": output.strip()})

    def report(message: str) -> None:
        if progress:
            progress(message)

    raw: dict[str, pd.DataFrame] = {}
    for modality in modalities:
        raw[modality] = pd.read_csv(io.BytesIO(uploaded_files[modality].getvalue()), low_memory=False)
        raw[modality].columns = [str(column).strip() for column in raw[modality].columns]
        missing = [column for column in keys + labels if column not in raw[modality].columns]
        if missing:
            raise ValueError(f"{modality} CSV is missing required columns: {', '.join(missing)}")
        if raw[modality].duplicated(keys).any():
            raise ValueError(f"{modality} CSV has duplicate participant_id/submission_id pairs.")

    inventory = pd.DataFrame([
        {"Modality": modality, "Rows": len(frame), "Columns": frame.shape[1],
         "Missing cells": int(frame.isna().sum().sum()), "Duplicate sessions": int(frame.duplicated(keys).sum())}
        for modality, frame in raw.items()
    ])
    base = raw["EEG"][keys + labels].copy()
    for modality, frame in raw.items():
        check = base.merge(frame[keys + labels], on=keys, how="outer", suffixes=("_EEG", "_other"), indicator=True, validate="one_to_one")
        if not check["_merge"].eq("both").all() or not check["engagement_EEG"].eq(check["engagement_other"]).all():
            raise ValueError(f"{modality} rows or labels do not align exactly with EEG. Check participant/session keys before running.")
    add_stage(
        "Read and synchronize the four tables",
        "read_csv() each modality; require participant_id, submission_id, engagement; validate one-to-one keys and matching labels.",
        inventory.to_string(index=False) + f"\n\nSynchronization passed: {len(base):,} sessions, {base.participant_id.nunique():,} participants; labels and keys match across modalities.",
    )

    feature_columns: dict[str, list[str]] = {}
    wide = base.copy()
    quality_rows = []
    excluded_pattern = re.compile(r"timestamp|counter|frame.?id|participant.?id|submission.?id", re.I)
    for modality, frame in raw.items():
        candidates = [
            column for column in frame.columns
            if column not in keys + labels
            and not excluded_pattern.search(column)
            and pd.api.types.is_numeric_dtype(frame[column])
            and "engagement" not in column.lower()
        ]
        if not candidates:
            raise ValueError(f"{modality} has no numeric predictor features after excluding identifiers and engagement-derived columns.")
        feature_columns[modality] = candidates
        block = frame[keys + candidates].copy()
        block[candidates] = block[candidates].replace([np.inf, -np.inf], np.nan)
        numeric = block[candidates]
        constant_count = int(sum(numeric[column].nunique(dropna=True) <= 1 for column in candidates))
        quality_rows.append({
            "Modality": modality, "Numeric features": len(candidates),
            "Missing feature cells": int(numeric.isna().sum().sum()),
            "Infinite values replaced": int(np.isinf(frame[candidates].to_numpy(dtype="float64", na_value=np.nan)).sum()),
            "Constant features": constant_count,
        })
        wide = wide.merge(block.rename(columns={column: f"{modality}__{column}" for column in candidates}), on=keys, validate="one_to_one")

    original = pd.to_numeric(wide.engagement, errors="raise").astype(int).to_numpy()
    if not set(np.unique(original)).issubset(label_map):
        raise ValueError("engagement must use the original integer labels 0, 1, 2, 3, 4.")
    y = np.asarray([label_map[value] for value in original], dtype=int)
    groups = wide.participant_id.astype(str).to_numpy()
    class_names = ["Low (0/1)", "Medium (2)", "High (3/4)"]
    class_counts = np.bincount(y, minlength=3)
    class_table = pd.DataFrame({"Class": class_names, "Sessions": class_counts, "Share": class_counts / len(y)})
    if len(np.unique(groups)) < 5 or np.any(class_counts == 0):
        raise ValueError("Nested 5-fold grouped evaluation requires at least five participants and all three grouped classes.")
    quality_table = pd.DataFrame(quality_rows)
    add_stage(
        "Audit features and label imbalance",
        "Exclude IDs, non-numeric columns, engagement-derived predictors; replace infinities; map 0/1→Low, 2→Medium, 3/4→High.",
        quality_table.to_string(index=False) + "\n\n" + class_table.to_string(index=False) + f"\n\nLargest/smallest class ratio: {class_counts.max()/class_counts.min():.2f}:1",
    )

    def prepare(train_indices: np.ndarray, eval_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        train_blocks, eval_blocks = [], []
        for modality in modalities:
            cols = [f"{modality}__{column}" for column in feature_columns[modality]]
            train_frame = wide.iloc[train_indices][cols].replace([np.inf, -np.inf], np.nan).astype("float32")
            eval_frame = wide.iloc[eval_indices][cols].replace([np.inf, -np.inf], np.nan).astype("float32")
            usable = [column for column in cols if train_frame[column].notna().any() and train_frame[column].nunique(dropna=True) > 1]
            train_frame, eval_frame = train_frame[usable], eval_frame[usable]
            if train_frame.empty:
                raise ValueError(f"No usable training features remain for {modality} in a fold.")
            duplicate_mask = ~train_frame.T.duplicated().to_numpy()
            train_frame, eval_frame = train_frame.loc[:, duplicate_mask], eval_frame.loc[:, duplicate_mask]
            imputer = SimpleImputer(strategy="median")
            train_values, eval_values = imputer.fit_transform(train_frame), imputer.transform(eval_frame)
            scaler = StandardScaler()
            train_values, eval_values = scaler.fit_transform(train_values), scaler.transform(eval_values)
            selector = SelectPercentile(score_func=f_classif, percentile=50)
            train_values = selector.fit_transform(train_values, y[train_indices])
            eval_values = selector.transform(eval_values)
            train_blocks.append(train_values.astype("float32"))
            eval_blocks.append(eval_values.astype("float32"))
        return np.concatenate(train_blocks, axis=1), np.concatenate(eval_blocks, axis=1)

    def oversample(x_values: np.ndarray, y_values: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        counts = np.bincount(y_values, minlength=3)
        target = int(counts.max())
        indices = []
        for class_id in range(3):
            class_indices = np.flatnonzero(y_values == class_id)
            if not len(class_indices):
                raise ValueError(f"A training fold does not contain class {class_id}; cannot oversample it.")
            indices.extend(class_indices.tolist())
            if len(class_indices) < target:
                indices.extend(rng.choice(class_indices, target - len(class_indices), replace=True).tolist())
        indices = np.asarray(indices)
        rng.shuffle(indices)
        return x_values[indices], y_values[indices]

    candidates = [
        ("RBF-SVM weighted C=1", "svm_weighted"),
        ("RBF-SVM oversampled C=1", "svm_oversampled"),
        ("RBF-SVM unweighted C=2", "svm_unweighted"),
        ("Random Forest balanced", "rf_balanced"),
    ]

    def fit_predict(name: str, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, seed: int) -> np.ndarray:
        if name == "svm_oversampled":
            x_train, y_train = oversample(x_train, y_train, seed)
        if name.startswith("svm_"):
            class_weight = "balanced" if name == "svm_weighted" else None
            c_value = 2.0 if name == "svm_unweighted" else 1.0
            model = SVC(C=c_value, kernel="rbf", gamma="scale", class_weight=class_weight, random_state=seed)
        else:
            model = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
        model.fit(x_train, y_train)
        return model.predict(x_eval)

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.full(len(y), -1, dtype=int)
    fold_rows, search_rows = [], []
    progress_logs = []
    for fold_number, (outer_train, outer_eval) in enumerate(outer.split(np.zeros(len(y)), y, groups), start=1):
        report(f"Outer fold {fold_number}/5: comparing models using inner participant-grouped validation.")
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42 + fold_number)
        inner_splits = list(inner_cv.split(np.zeros(len(outer_train)), y[outer_train], groups[outer_train]))
        mean_scores = {}
        for candidate_name, candidate_key in candidates:
            scores = []
            for inner_number, (inner_train_rel, inner_eval_rel) in enumerate(inner_splits, start=1):
                inner_train = outer_train[inner_train_rel]
                inner_eval = outer_train[inner_eval_rel]
                x_train, x_eval = prepare(inner_train, inner_eval)
                predicted = fit_predict(candidate_key, x_train, y[inner_train], x_eval, 1000 + fold_number * 10 + inner_number)
                scores.append(f1_score(y[inner_eval], predicted, average="macro", zero_division=0))
            mean_scores[candidate_name] = float(np.mean(scores))
            search_rows.append({"Outer fold": fold_number, "Candidate": candidate_name, "Inner Macro-F1": mean_scores[candidate_name]})
        selected_name = max(mean_scores, key=mean_scores.get)
        selected_key = dict(candidates)[selected_name]
        x_train, x_eval = prepare(outer_train, outer_eval)
        predicted = fit_predict(selected_key, x_train, y[outer_train], x_eval, 5000 + fold_number)
        oof[outer_eval] = predicted
        fold_result = {
            "Fold": fold_number, "Selected model": selected_name,
            "Accuracy": accuracy_score(y[outer_eval], predicted),
            "Macro-F1": f1_score(y[outer_eval], predicted, average="macro", zero_division=0),
            "Weighted-F1": f1_score(y[outer_eval], predicted, average="weighted", zero_division=0),
            "MCC": matthews_corrcoef(y[outer_eval], predicted),
        }
        fold_rows.append(fold_result)
        progress_logs.append(f"Fold {fold_number}: selected {selected_name} (inner Macro-F1={mean_scores[selected_name]:.3f}); outer Accuracy={fold_result['Accuracy']:.3f}, Macro-F1={fold_result['Macro-F1']:.3f}.")

    if np.any(oof < 0):
        raise RuntimeError("Some sessions did not receive an out-of-fold prediction.")
    overall = {
        "Sessions": int(len(y)), "Participants": int(pd.Series(groups).nunique()),
        "Accuracy": float(accuracy_score(y, oof)),
        "Macro-F1": float(f1_score(y, oof, average="macro", zero_division=0)),
        "Weighted-F1": float(f1_score(y, oof, average="weighted", zero_division=0)),
        "Balanced accuracy": float(balanced_accuracy_score(y, oof)),
        "MCC": float(matthews_corrcoef(y, oof)),
    }
    majority_prediction = np.full(len(y), int(class_counts.argmax()), dtype=int)
    majority_accuracy = float(accuracy_score(y, majority_prediction))
    majority_metrics = {
        "Accuracy": majority_accuracy,
        "Macro-F1": float(f1_score(y, majority_prediction, average="macro", zero_division=0)),
        "Weighted-F1": float(f1_score(y, majority_prediction, average="weighted", zero_division=0)),
        "Balanced accuracy": float(balanced_accuracy_score(y, majority_prediction)),
        "MCC": float(matthews_corrcoef(y, majority_prediction)),
        "Sessions": len(y), "Participants": pd.Series(groups).nunique(),
    }
    fold_table = pd.DataFrame(fold_rows)
    search_table = pd.DataFrame(search_rows)
    report_text = classification_report(y, oof, labels=[0, 1, 2], target_names=class_names, zero_division=0)
    matrix = confusion_matrix(y, oof, labels=[0, 1, 2])
    predictions = pd.DataFrame({
        "participant_id": groups, "session_id": wide.submission_id.astype(str),
        "actual_engagement_grouped_0_2": y, "actual_engagement_original_0_4": original,
        "predicted_engagement": oof, "predicted_engagement_name": [class_names[index] for index in oof],
        "validation_fold": np.zeros(len(y), dtype=int),
    })
    for fold_number, (outer_train, outer_eval) in enumerate(outer.split(np.zeros(len(y)), y, groups), start=1):
        predictions.loc[outer_eval, "validation_fold"] = fold_number

    add_stage(
        "Fold-local preprocessing and feature selection",
        "For each training fold: drop constant/duplicate columns → median imputation → standard scaling → ANOVA top 50% per modality.",
        "All preprocessing was fitted on each training split only. Validation/test-fold rows were not resampled or used to fit preprocessing.",
    )
    add_stage(
        "Nested grouped model search",
        "Outer: StratifiedGroupKFold(5) by participant. Inner: StratifiedGroupKFold(3). Select the candidate by inner Macro-F1; evaluate once on outer held-out participants.",
        "\n".join(progress_logs) + "\n\nInner-fold candidate table:\n" + search_table.round(3).to_string(index=False),
    )
    add_stage(
        "Final out-of-fold metrics",
        "Compute Accuracy, Macro-F1, Weighted-F1, balanced accuracy and MCC from outer-fold predictions; compare accuracy with the majority-only reference.",
        pd.DataFrame([overall, majority_metrics], index=["Grouped model", "Majority-only reference"]).round(3).to_string(),
    )
    add_stage(
        "Per-class report and confusion matrix",
        "Use outer out-of-fold labels only; rows in the confusion matrix are actual classes and columns are predictions.",
        report_text + "\nConfusion matrix (actual rows × predicted columns):\n" + pd.DataFrame(matrix, index=class_names, columns=class_names).to_string(),
    )
    return {
        "stages": stage_rows, "overall": overall, "majority_accuracy": majority_accuracy,
        "fold_table": fold_table, "search_table": search_table, "predictions": predictions,
        "class_table": class_table, "confusion_matrix": matrix.tolist(), "class_names": class_names,
    }


