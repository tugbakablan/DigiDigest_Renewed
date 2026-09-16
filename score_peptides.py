from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from train_baselines import feature_matrix


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    parts: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(parts).upper()))
            header = line[1:].strip()
            parts = []
        else:
            parts.append(line)
    if header is not None:
        records.append((header, "".join(parts).upper()))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    classifier_bundle = joblib.load(args.model_dir / "ace_classifier.joblib")
    regressor_bundle = joblib.load(args.model_dir / "pic50_regressor.joblib")
    records = read_fasta(args.fasta)
    sequences = [sequence for _, sequence in records]
    features = feature_matrix(sequences)
    classifier_indices = [
        index
        for index, sequence in enumerate(sequences)
        if classifier_bundle["minimum_length"] <= len(sequence) <= classifier_bundle["maximum_length"]
    ]
    regression_indices = [
        index
        for index, sequence in enumerate(sequences)
        if regressor_bundle["minimum_length"] <= len(sequence) <= regressor_bundle["maximum_length"]
    ]
    classifier_predictions = {
        index: float(probability)
        for index, probability in zip(
            classifier_indices,
            classifier_bundle["model"].predict_proba(features[classifier_indices])[:, 1],
        )
    }
    regression_predictions = {
        index: float(prediction)
        for index, prediction in zip(
            regression_indices,
            regressor_bundle["model"].predict(features[regression_indices]),
        )
    }

    rows = []
    for index, (record_id, sequence) in enumerate(records):
        classifier_in_scope = index in classifier_predictions
        regression_in_scope = index in regression_predictions
        probability = classifier_predictions.get(index)
        predicted_pic50_raw = regression_predictions.get(index)
        ace_prediction_positive = probability is not None and probability >= 0.5
        predicted_pic50 = predicted_pic50_raw if ace_prediction_positive else None
        predicted_ic50_um = 10 ** (6.0 - predicted_pic50) if predicted_pic50 is not None else None
        rows.append(
            {
                "record_id": record_id,
                "sequence": sequence,
                "length": len(sequence),
                "ace_probability_exploratory": probability,
                "classifier_in_scope": classifier_in_scope,
                "predicted_pIC50_exploratory": predicted_pic50,
                "predicted_IC50_uM_exploratory": predicted_ic50_um,
                "regression_in_scope": predicted_pic50 is not None,
                "ace_prediction_positive": ace_prediction_positive,
                "ace_assessment_status": (
                    "predicted_ACE_candidate" if ace_prediction_positive
                    else "ACE_classification_negative" if classifier_in_scope
                    else "ACE_assay_required"
                ),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
