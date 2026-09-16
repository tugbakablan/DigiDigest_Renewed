from __future__ import annotations

import numpy as np
import pandas as pd

from train_baselines import feature_matrix


FEATURE_GROUPS = {
    "Global physicochemical features": slice(0, 9),
    "Amino-acid composition": slice(9, 29),
    "Dipeptide pattern": slice(29, 429),
    "N-terminal pattern": slice(429, 469),
    "C-terminal pattern": slice(469, 529),
}


def reference_feature_vector(dataset: pd.DataFrame) -> np.ndarray:
    return feature_matrix(dataset["sequence"].astype(str).tolist()).mean(axis=0)


def explain_classifier(bundle: dict, sequence: str, reference: np.ndarray) -> list[dict]:
    vector = feature_matrix([sequence])[0]
    baseline = float(bundle["model"].predict_proba(vector.reshape(1, -1))[0, 1])
    rows = []
    for label, feature_slice in FEATURE_GROUPS.items():
        ablated = vector.copy()
        ablated[feature_slice] = reference[feature_slice]
        changed = float(bundle["model"].predict_proba(ablated.reshape(1, -1))[0, 1])
        rows.append({"Feature group": label, "Influence (percentage points)": 100 * (baseline - changed)})
    return sorted(rows, key=lambda row: abs(row["Influence (percentage points)"]), reverse=True)


def explain_regressor(bundle: dict, sequence: str, reference: np.ndarray) -> list[dict]:
    vector = feature_matrix([sequence])[0]
    baseline = float(bundle["model"].predict(vector.reshape(1, -1))[0])
    rows = []
    for label, feature_slice in FEATURE_GROUPS.items():
        ablated = vector.copy()
        ablated[feature_slice] = reference[feature_slice]
        changed = float(bundle["model"].predict(ablated.reshape(1, -1))[0])
        rows.append({"Feature group": label, "Influence on predicted pIC50": baseline - changed})
    return sorted(rows, key=lambda row: abs(row["Influence on predicted pIC50"]), reverse=True)


def summarize_classifier_explanation(
    sequence: str,
    probability: float,
    explanation_rows: list[dict],
    threshold: float = 0.5,
    predicted_ic50: float | None = None,
) -> str:
    """Return a concise, numerical description of local classifier behavior."""
    supporting = [
        row for row in explanation_rows
        if float(row["Influence (percentage points)"]) > 0
    ]
    opposing = [
        row for row in explanation_rows
        if float(row["Influence (percentage points)"]) < 0
    ]
    supporting.sort(key=lambda row: float(row["Influence (percentage points)"]), reverse=True)
    opposing.sort(key=lambda row: float(row["Influence (percentage points)"]))

    decision = "above" if probability >= threshold else "below"
    status = "a predicted ACE candidate" if probability >= threshold else "classification negative"
    sentences = [
        f"For {sequence}, the classifier reports {100 * probability:.1f}% ACE probability, "
        f"which is {decision} the {100 * threshold:.0f}% decision threshold; the model status is {status}."
    ]
    if supporting:
        row = supporting[0]
        sentences.append(
            f"The strongest supporting feature group is {row['Feature group'].lower()} "
            f"(+{float(row['Influence (percentage points)']):.1f} percentage points versus the training reference)."
        )
    if opposing:
        row = opposing[0]
        sentences.append(
            f"The strongest opposing feature group is {row['Feature group'].lower()} "
            f"({float(row['Influence (percentage points)']):.1f} percentage points)."
        )
    if predicted_ic50 is not None:
        sentences.append(
            f"Because the classification is positive, the separate potency model estimates an IC50 of {predicted_ic50:.1f} µM."
        )
    else:
        sentences.append("No predicted IC50 is reported because the ACE classification is not positive.")
    return " ".join(sentences)


def classifier_explanation_card(
    probability: float,
    explanation_rows: list[dict],
    threshold: float = 0.5,
    predicted_ic50: float | None = None,
) -> dict[str, str]:
    """Build a scan-friendly numerical XAI card for laboratory users."""
    supporting = sorted(
        (
            row for row in explanation_rows
            if float(row["Influence (percentage points)"]) > 0
        ),
        key=lambda row: float(row["Influence (percentage points)"]),
        reverse=True,
    )
    opposing = sorted(
        (
            row for row in explanation_rows
            if float(row["Influence (percentage points)"]) < 0
        ),
        key=lambda row: float(row["Influence (percentage points)"]),
    )
    margin = 100 * (probability - threshold)
    absolute_margin = abs(margin)
    if absolute_margin < 5:
        margin_label = "Borderline"
    elif absolute_margin < 15:
        margin_label = "Moderate margin"
    else:
        margin_label = "Clear margin"

    support_text = "No feature group produced a positive local effect relative to the training reference."
    if supporting:
        row = supporting[0]
        support_text = (
            f"{row['Feature group']} had the strongest positive effect, increasing the ACE probability "
            f"by {float(row['Influence (percentage points)']):.1f} percentage points."
        )
    opposition_text = "No feature group produced a negative local effect relative to the training reference."
    if opposing:
        row = opposing[0]
        opposition_text = (
            f"{row['Feature group']} had the strongest negative effect, decreasing the ACE probability "
            f"by {abs(float(row['Influence (percentage points)'])):.1f} percentage points."
        )

    if margin_label == "Borderline":
        interpretation = (
            f"The result is only {absolute_margin:.1f} percentage points "
            f"{'above' if margin >= 0 else 'below'} the decision threshold. It is borderline and should not be interpreted as strong ACE evidence."
        )
    elif probability >= threshold:
        interpretation = (
            f"The result is {absolute_margin:.1f} percentage points above the decision threshold and supports prioritization as a predicted ACE candidate."
        )
    else:
        interpretation = (
            f"The result is {absolute_margin:.1f} percentage points below the decision threshold and does not support ACE-candidate classification."
        )

    potency_interpretation = (
        f"The separate potency model estimates IC50 at {predicted_ic50:.1f} µM; this is a model estimate, not a measured value."
        if predicted_ic50 is not None
        else "Predicted IC50 is not reported because the ACE classification is not positive."
    )

    return {
        "Decision": "Predicted ACE candidate" if probability >= threshold else "Classification negative",
        "Probability": f"{100 * probability:.1f}%",
        "Threshold margin": f"{margin:+.1f} pp · {margin_label}",
        "Main support": support_text,
        "Main opposition": opposition_text,
        "Predicted potency": f"IC50 {predicted_ic50:.1f} µM" if predicted_ic50 is not None else "Not reported",
        "Laboratory interpretation": interpretation,
        "Potency interpretation": potency_interpretation,
    }
