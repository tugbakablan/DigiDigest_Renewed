from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: index for index, aa in enumerate(AMINO_ACIDS)}
HYDROPHOBICITY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}


def sequence_features(sequence: str) -> np.ndarray:
    sequence = sequence.strip().upper()
    length = len(sequence)
    composition = np.zeros(len(AMINO_ACIDS), dtype=float)
    dipeptides = np.zeros(len(AMINO_ACIDS) ** 2, dtype=float)
    for aa in sequence:
        composition[AA_INDEX[aa]] += 1.0
    composition /= max(length, 1)
    for left, right in zip(sequence, sequence[1:]):
        dipeptides[AA_INDEX[left] * len(AMINO_ACIDS) + AA_INDEX[right]] += 1.0
    if length > 1:
        dipeptides /= length - 1

    n_terminal = np.zeros(2 * len(AMINO_ACIDS), dtype=float)
    c_terminal = np.zeros(3 * len(AMINO_ACIDS), dtype=float)
    for position, aa in enumerate(sequence[:2]):
        n_terminal[position * len(AMINO_ACIDS) + AA_INDEX[aa]] = 1.0
    for position, aa in enumerate(reversed(sequence[-3:])):
        c_terminal[position * len(AMINO_ACIDS) + AA_INDEX[aa]] = 1.0

    hydrophobicity = np.mean([HYDROPHOBICITY[aa] for aa in sequence])
    approximate_charge = (
        sequence.count("K") + sequence.count("R") + 0.1 * sequence.count("H")
        - sequence.count("D") - sequence.count("E")
    )
    summary = np.array(
        [
            length,
            math.log1p(length),
            hydrophobicity,
            approximate_charge,
            sum(sequence.count(aa) for aa in "FWY") / length,
            sequence.count("P") / length,
            sequence.count("G") / length,
            sum(sequence.count(aa) for aa in "DE") / length,
            sum(sequence.count(aa) for aa in "KRH") / length,
        ],
        dtype=float,
    )
    return np.concatenate([summary, composition, dipeptides, n_terminal, c_terminal])


def feature_matrix(sequences: list[str]) -> np.ndarray:
    return np.vstack([sequence_features(sequence) for sequence in sequences])


def levenshtein_with_limit(left: str, right: str, limit: int) -> int:
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for row_index, left_char in enumerate(left, start=1):
        current = [row_index]
        row_minimum = row_index
        for column_index, right_char in enumerate(right, start=1):
            value = min(
                current[-1] + 1,
                previous[column_index] + 1,
                previous[column_index - 1] + (left_char != right_char),
            )
            current.append(value)
            row_minimum = min(row_minimum, value)
        if row_minimum > limit:
            return limit + 1
        previous = current
    return previous[-1]


def cluster_sequences(sequences: list[str], identity_threshold: float = 0.8) -> dict[str, int]:
    unique_sequences = sorted(set(sequences), key=lambda item: (len(item), item))
    representatives: dict[int, list[tuple[int, str]]] = defaultdict(list)
    assignments: dict[str, int] = {}
    next_cluster = 0
    for sequence in unique_sequences:
        matched_cluster: int | None = None
        maximum_length_difference = max(1, int(math.floor((1.0 - identity_threshold) * len(sequence))))
        for candidate_length in range(
            max(1, len(sequence) - maximum_length_difference),
            len(sequence) + maximum_length_difference + 1,
        ):
            for cluster_id, representative in representatives.get(candidate_length, []):
                allowed = int(math.floor((1.0 - identity_threshold) * max(len(sequence), len(representative))))
                if levenshtein_with_limit(sequence, representative, allowed) <= allowed:
                    matched_cluster = cluster_id
                    break
            if matched_cluster is not None:
                break
        if matched_cluster is None:
            matched_cluster = next_cluster
            next_cluster += 1
            representatives[len(sequence)].append((matched_cluster, sequence))
        assignments[sequence] = matched_cluster
    return assignments


def build_length_matched_classifier_data(data_dir: Path, random_state: int = 42) -> pd.DataFrame:
    positives = pd.read_csv(data_dir / "positive_evidence_catalog.csv")
    audit = pd.read_csv(data_dir / "ai4aceip_label_audit.csv")
    negatives = audit[
        (audit["ai4aceip_label"] == 0)
        & (~audit["label_conflict"].astype(bool))
        & (audit["standard_sequence"].astype(bool))
    ].copy()

    positives = positives[
        positives["standard_sequence"].astype(bool) & positives["length"].between(5, 30)
    ].copy()
    negatives = negatives[negatives["length"].between(5, 30)].copy()
    positives["label"] = 1
    negatives["label"] = 0
    positives["source_role"] = "positive_evidence"
    negatives["source_role"] = "AI4ACEIP_candidate_negative"

    samples = []
    for length in sorted(set(positives["length"]) & set(negatives["length"])):
        positive_group = positives[positives["length"] == length]
        negative_group = negatives[negatives["length"] == length]
        sample_size = min(len(positive_group), len(negative_group))
        if sample_size < 10:
            continue
        samples.append(positive_group.sample(sample_size, random_state=random_state + int(length)))
        samples.append(negative_group.sample(sample_size, random_state=random_state + 1000 + int(length)))

    result = pd.concat(samples, ignore_index=True)[["sequence", "length", "label", "source_role"]]
    result = result.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    clusters = cluster_sequences(result["sequence"].tolist(), identity_threshold=0.8)
    result["similarity_cluster"] = result["sequence"].map(clusters)
    return result


def classification_models(random_state: int = 42):
    return {
        "length_only_logistic": Pipeline(
            [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=random_state))]
        ),
        "sequence_logistic": Pipeline(
            [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=4000, C=0.25, random_state=random_state))]
        ),
        "sequence_extra_trees": ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def evaluate_classifiers(dataset: pd.DataFrame, random_state: int = 42):
    sequences = dataset["sequence"].tolist()
    full_features = feature_matrix(sequences)
    length_features = dataset[["length"]].to_numpy(dtype=float)
    labels = dataset["label"].to_numpy(dtype=int)
    groups = dataset["similarity_cluster"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    metric_rows = []
    prediction_rows = []

    for model_name, estimator in classification_models(random_state).items():
        features = length_features if model_name == "length_only_logistic" else full_features
        for fold, (train_index, test_index) in enumerate(splitter.split(features, labels, groups), start=1):
            model = clone(estimator)
            model.fit(features[train_index], labels[train_index])
            probabilities = model.predict_proba(features[test_index])[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            metric_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "n_train": len(train_index),
                    "n_test": len(test_index),
                    "roc_auc": roc_auc_score(labels[test_index], probabilities),
                    "pr_auc": average_precision_score(labels[test_index], probabilities),
                    "balanced_accuracy": balanced_accuracy_score(labels[test_index], predictions),
                    "f1": f1_score(labels[test_index], predictions),
                    "mcc": matthews_corrcoef(labels[test_index], predictions),
                }
            )
            for row_index, probability in zip(test_index, probabilities):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "sequence": sequences[row_index],
                        "length": int(dataset.iloc[row_index]["length"]),
                        "label": int(labels[row_index]),
                        "probability": float(probability),
                        "similarity_cluster": int(groups[row_index]),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows), full_features, labels


def build_regression_data(data_dir: Path) -> pd.DataFrame:
    records = pd.read_csv(data_dir / "ic50_records_clean.csv")
    records = records[
        records["regression_eligible"].astype(bool)
        & records["standard_sequence"].astype(bool)
        & records["length"].between(2, 30)
        & records["pIC50"].notna()
    ].copy()
    grouped = records.groupby("sequence", as_index=False).agg(
        length=("length", "first"),
        pIC50=("pIC50", "median"),
        pIC50_min=("pIC50", "min"),
        pIC50_max=("pIC50", "max"),
        measurement_count=("pIC50", "size"),
        source_dataset_count=("source_dataset", "nunique"),
    )
    clusters = cluster_sequences(grouped["sequence"].tolist(), identity_threshold=0.8)
    grouped["similarity_cluster"] = grouped["sequence"].map(clusters)
    grouped["measurement_span"] = grouped["pIC50_max"] - grouped["pIC50_min"]
    return grouped.sort_values("sequence").reset_index(drop=True)


def regression_models(random_state: int = 42):
    return {
        "sequence_ridge": Pipeline([( "scale", StandardScaler()), ("model", Ridge(alpha=25.0))]),
        "sequence_extra_trees": ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def evaluate_regressors(dataset: pd.DataFrame, random_state: int = 42):
    sequences = dataset["sequence"].tolist()
    features = feature_matrix(sequences)
    targets = dataset["pIC50"].to_numpy(dtype=float)
    groups = dataset["similarity_cluster"].to_numpy()
    strata = pd.qcut(targets, q=5, labels=False, duplicates="drop")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    metric_rows = []
    prediction_rows = []
    for model_name, estimator in regression_models(random_state).items():
        for fold, (train_index, test_index) in enumerate(splitter.split(features, strata, groups), start=1):
            model = clone(estimator)
            model.fit(features[train_index], targets[train_index])
            predictions = model.predict(features[test_index])
            correlation = spearmanr(targets[test_index], predictions).statistic
            metric_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "n_train": len(train_index),
                    "n_test": len(test_index),
                    "mae_pIC50": mean_absolute_error(targets[test_index], predictions),
                    "rmse_pIC50": math.sqrt(mean_squared_error(targets[test_index], predictions)),
                    "r2": r2_score(targets[test_index], predictions),
                    "spearman": correlation,
                }
            )
            for row_index, prediction in zip(test_index, predictions):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "sequence": sequences[row_index],
                        "length": int(dataset.iloc[row_index]["length"]),
                        "pIC50_observed": float(targets[row_index]),
                        "pIC50_predicted": float(prediction),
                        "similarity_cluster": int(groups[row_index]),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows), features, targets


def metric_summary(metrics: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return metrics.groupby("model")[columns].agg(["mean", "std"]).round(4)


def write_model_report(
    path: Path,
    classifier_data: pd.DataFrame,
    classifier_metrics: pd.DataFrame,
    regression_data: pd.DataFrame,
    regression_metrics: pd.DataFrame,
    selected_classifier: str,
    selected_regressor: str,
) -> None:
    classifier_summary = metric_summary(
        classifier_metrics, ["roc_auc", "pr_auc", "balanced_accuracy", "f1", "mcc"]
    )
    regression_summary = metric_summary(
        regression_metrics, ["mae_pIC50", "rmse_pIC50", "r2", "spearman"]
    )
    text = f"""# ACE baseline model report

## Classification data

- Records: {len(classifier_data):,}
- Positive records: {(classifier_data['label'] == 1).sum():,}
- Candidate-negative records: {(classifier_data['label'] == 0).sum():,}
- Length scope: {classifier_data['length'].min()}–{classifier_data['length'].max()} amino acids
- Similarity clusters: {classifier_data['similarity_cluster'].nunique():,}
- Selected exploratory model: `{selected_classifier}`

The classes were sampled to the same count at each exact peptide length. This
removes the strongest length shortcut but does not turn AI4ACEIP candidate
negatives into experimentally confirmed negatives.

### Clustered five-fold cross-validation

```
{classifier_summary.to_string()}
```

The length-only model is reported as a shortcut diagnostic. A useful sequence
model should materially outperform it.

## pIC50 regression data

- Unique peptide targets: {len(regression_data):,}
- Length scope: {regression_data['length'].min()}–{regression_data['length'].max()} amino acids
- Similarity clusters: {regression_data['similarity_cluster'].nunique():,}
- Sequences with repeated measurements: {(regression_data['measurement_count'] > 1).sum():,}
- Selected exploratory model: `{selected_regressor}`

Repeated exact measurements were summarized by their median at sequence level.
The minimum, maximum, count and span remain in `regression_dataset.csv` for
heterogeneity review.

### Clustered five-fold cross-validation

```
{regression_summary.to_string()}
```

## Use restrictions

1. Do not use the classifier outside {classifier_data['length'].min()}–{classifier_data['length'].max()} amino acids.
2. Do not interpret its probability as clinical efficacy or absorption.
3. Do not label unknown DigiDigest fragments as negative.
4. Do not claim INFOGEST validity from BIOPEP rule-based fragments.
5. Treat both fitted models as baselines until external and laboratory validation.
6. Rank candidates with separate activity, digestion, transport and safety evidence rather than one unexplained probability.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    classifier_data = build_length_matched_classifier_data(args.data_dir, args.random_state)
    classifier_metrics, classifier_predictions, classifier_features, classifier_labels = evaluate_classifiers(
        classifier_data, args.random_state
    )
    classifier_means = classifier_metrics.groupby("model")["mcc"].mean()
    selected_classifier = classifier_means.drop("length_only_logistic").idxmax()
    classifier_estimator = classification_models(args.random_state)[selected_classifier]
    classifier_estimator.fit(classifier_features, classifier_labels)
    joblib.dump(
        {
            "model": classifier_estimator,
            "model_name": selected_classifier,
            "minimum_length": 5,
            "maximum_length": int(classifier_data["length"].max()),
            "feature_version": "composition_dipeptide_terminal_v1",
            "negative_evidence": "AI4ACEIP candidate negatives after exact positive-conflict removal",
        },
        args.model_dir / "ace_classifier.joblib",
    )

    regression_data = build_regression_data(args.data_dir)
    regression_metrics, regression_predictions, regression_features, regression_targets = evaluate_regressors(
        regression_data, args.random_state
    )
    selected_regressor = regression_metrics.groupby("model")["mae_pIC50"].mean().idxmin()
    regression_estimator = regression_models(args.random_state)[selected_regressor]
    regression_estimator.fit(regression_features, regression_targets)
    joblib.dump(
        {
            "model": regression_estimator,
            "model_name": selected_regressor,
            "minimum_length": 2,
            "maximum_length": 30,
            "feature_version": "composition_dipeptide_terminal_v1",
            "target": "median pIC50 from exact normalized records",
        },
        args.model_dir / "pic50_regressor.joblib",
    )

    classifier_data.to_csv(args.data_dir / "classifier_dataset.csv", index=False)
    classifier_metrics.to_csv(args.report_dir / "classifier_metrics.csv", index=False)
    classifier_predictions.to_csv(args.report_dir / "classifier_oof_predictions.csv", index=False)
    regression_data.to_csv(args.data_dir / "regression_dataset.csv", index=False)
    regression_metrics.to_csv(args.report_dir / "regression_metrics.csv", index=False)
    regression_predictions.to_csv(args.report_dir / "regression_oof_predictions.csv", index=False)
    write_model_report(
        args.report_dir / "baseline_model_report.md",
        classifier_data,
        classifier_metrics,
        regression_data,
        regression_metrics,
        selected_classifier,
        selected_regressor,
    )
    metadata = {
        "random_state": args.random_state,
        "classifier_model": selected_classifier,
        "regression_model": selected_regressor,
        "classifier_records": len(classifier_data),
        "regression_records": len(regression_data),
    }
    (args.report_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
