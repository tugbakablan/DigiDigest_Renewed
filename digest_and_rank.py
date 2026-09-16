from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from score_peptides import read_fasta
from train_baselines import feature_matrix


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class Fragment:
    protein_id: str
    sequence: str
    start: int  # one-based, inclusive
    end: int  # one-based, inclusive
    stage: str
    enzymes: str


def validate_protein(sequence: str) -> None:
    invalid = sorted(set(sequence) - STANDARD_AA)
    if invalid:
        raise ValueError(f"Only the 20 standard amino acids are supported; found: {','.join(invalid)}")


def pepsin_ph_gt_2_cut(sequence: str, index: int) -> bool:
    """Broad PeptideCutter-style pepsin approximation for gastric pH > 2.

    ``index`` is the zero-based residue on the N-terminal side of the bond.
    At this pH pepsin specificity is broad; aromatic/hydrophobic P1 residues are
    used and cleavage immediately before proline is blocked. This is a
    deterministic hypothesis generator, not a kinetic INFOGEST simulation.
    """
    return sequence[index] in "FLWY" and (index + 1 == len(sequence) or sequence[index + 1] != "P")


def trypsin_cut(sequence: str, index: int) -> bool:
    return sequence[index] in "KR" and (index + 1 == len(sequence) or sequence[index + 1] != "P")


def chymotrypsin_high_cut(sequence: str, index: int) -> bool:
    return sequence[index] in "FYW" and (index + 1 == len(sequence) or sequence[index + 1] != "P")


def cleavage_products(fragment: Fragment, cut_functions, stage: str, enzymes: str) -> list[Fragment]:
    cut_after = [
        index + 1
        for index in range(len(fragment.sequence) - 1)
        if any(cut_function(fragment.sequence, index) for cut_function in cut_functions)
    ]
    boundaries = [0, *cut_after, len(fragment.sequence)]
    products: list[Fragment] = []
    for left, right in zip(boundaries, boundaries[1:]):
        if left == right:
            continue
        products.append(
            Fragment(
                protein_id=fragment.protein_id,
                sequence=fragment.sequence[left:right],
                start=fragment.start + left,
                end=fragment.start + right - 1,
                stage=stage,
                enzymes=enzymes,
            )
        )
    return products


def staged_digest(protein_id: str, sequence: str, coordinate_start: int = 1) -> list[Fragment]:
    sequence = sequence.upper().strip()
    validate_protein(sequence)
    parent = Fragment(
        protein_id,
        sequence,
        coordinate_start,
        coordinate_start + len(sequence) - 1,
        "input",
        "none",
    )
    gastric = cleavage_products(parent, [pepsin_ph_gt_2_cut], "gastric", "pepsin_ph_gt_2")
    intestinal: list[Fragment] = []
    for gastric_fragment in gastric:
        intestinal.extend(
            cleavage_products(
                gastric_fragment,
                [trypsin_cut, chymotrypsin_high_cut],
                "intestinal",
                "trypsin+chymotrypsin_high_specificity",
            )
        )
    return [*gastric, *intestinal]


def load_evidence(data_dir: Path) -> tuple[set[str], dict[str, float]]:
    positives = pd.read_csv(data_dir / "positive_evidence_catalog.csv")
    regression = pd.read_csv(data_dir / "regression_dataset.csv")
    positive_sequences = set(positives["sequence"].astype(str))
    measured_pic50 = dict(zip(regression["sequence"].astype(str), regression["pIC50"].astype(float)))
    return positive_sequences, measured_pic50


def annotate_and_rank(
    fragments: list[Fragment], model_dir: Path, data_dir: Path, target_stage: str
) -> pd.DataFrame:
    classifier_bundle = joblib.load(model_dir / "ace_classifier.joblib")
    regressor_bundle = joblib.load(model_dir / "pic50_regressor.joblib")
    positive_sequences, measured_pic50 = load_evidence(data_dir)

    rows = [fragment.__dict__ | {"length": len(fragment.sequence)} for fragment in fragments]
    table = pd.DataFrame(rows)
    unique_sequences = table["sequence"].drop_duplicates().tolist()
    valid_sequences = [sequence for sequence in unique_sequences if len(sequence) >= 1]
    features = feature_matrix(valid_sequences)
    feature_lookup = {sequence: features[index] for index, sequence in enumerate(valid_sequences)}

    classifier_sequences = [
        sequence for sequence in unique_sequences
        if classifier_bundle["minimum_length"] <= len(sequence) <= classifier_bundle["maximum_length"]
    ]
    regression_sequences = [
        sequence for sequence in unique_sequences
        if regressor_bundle["minimum_length"] <= len(sequence) <= regressor_bundle["maximum_length"]
    ]
    classifier_predictions = {}
    if classifier_sequences:
        matrix = pd.DataFrame([feature_lookup[s] for s in classifier_sequences]).to_numpy()
        values = classifier_bundle["model"].predict_proba(matrix)[:, 1]
        classifier_predictions = dict(zip(classifier_sequences, values.astype(float)))
    regression_predictions = {}
    if regression_sequences:
        matrix = pd.DataFrame([feature_lookup[s] for s in regression_sequences]).to_numpy()
        values = regressor_bundle["model"].predict(matrix)
        regression_predictions = dict(zip(regression_sequences, values.astype(float)))

    table["known_ace_positive_exact_match"] = table["sequence"].isin(positive_sequences)
    table["measured_median_pIC50"] = table["sequence"].map(measured_pic50)
    table["ace_probability_exploratory"] = table["sequence"].map(classifier_predictions)
    table["predicted_pIC50_model_raw"] = table["sequence"].map(regression_predictions)
    table["measured_median_IC50_uM"] = table["measured_median_pIC50"].map(
        lambda value: 10 ** (6.0 - value) if pd.notna(value) else None
    )
    table["classifier_in_scope"] = table["ace_probability_exploratory"].notna()
    table["ace_prediction_positive"] = (
        table["classifier_in_scope"] & (table["ace_probability_exploratory"] >= 0.5)
    )
    table["ace_supported"] = (
        table["known_ace_positive_exact_match"] | table["ace_prediction_positive"]
    )
    table["predicted_pIC50_exploratory"] = table["predicted_pIC50_model_raw"].where(
        table["ace_prediction_positive"]
    )
    table["predicted_IC50_uM_exploratory"] = table["predicted_pIC50_exploratory"].map(
        lambda value: 10 ** (6.0 - value) if pd.notna(value) else None
    )
    table = table.drop(columns=["predicted_pIC50_model_raw"])
    table["regression_in_scope"] = table["predicted_pIC50_exploratory"].notna()
    table["ace_assessment_status"] = table.apply(_ace_assessment_status, axis=1)
    table["systemic_size_category"] = table["length"].map(
        lambda length: (
            "too_small_for_peptide_model" if length < 2 else
            "small_possible" if length <= 3 else
            "preferred_screening_window" if length <= 12 else
            "limited_intact_absorption_expected"
        )
    )
    table["systemic_size_priority"] = table["length"].map(
        lambda length: 3 if length < 2 else 1 if length <= 3 else 0 if length <= 12 else 2
    )
    table["evidence_tier"] = 5
    table.loc[table["classifier_in_scope"] & ~table["ace_prediction_positive"], "evidence_tier"] = 4
    table.loc[table["regression_in_scope"], "evidence_tier"] = 3
    table.loc[table["known_ace_positive_exact_match"], "evidence_tier"] = 2
    table.loc[table["measured_median_pIC50"].notna(), "evidence_tier"] = 1
    table["activity_sort_value"] = table["measured_median_pIC50"].fillna(
        table["predicted_pIC50_exploratory"]
    )

    target = table[table["stage"] == target_stage].copy()
    occurrence_counts = target.groupby("sequence").size().rename("target_stage_occurrences")
    target = target.drop_duplicates("sequence").join(occurrence_counts, on="sequence")
    target = target.sort_values(
        ["evidence_tier", "systemic_size_priority", "activity_sort_value", "ace_probability_exploratory", "length"],
        ascending=[True, True, False, False, True],
        na_position="last",
    )
    target["priority_rank"] = range(1, len(target) + 1)
    table = table.merge(
        target[["sequence", "priority_rank", "target_stage_occurrences"]],
        on="sequence",
        how="left",
    )
    table.loc[table["stage"] != target_stage, ["priority_rank", "target_stage_occurrences"]] = None
    return table.sort_values(["protein_id", "stage", "start", "end"]).reset_index(drop=True)


def _ace_assessment_status(row: pd.Series) -> str:
    if pd.notna(row["measured_median_pIC50"]):
        return "known_ACE_with_measured_potency"
    if bool(row["known_ace_positive_exact_match"]):
        return "known_ACE_without_measured_IC50"
    if bool(row["ace_prediction_positive"]):
        return "predicted_ACE_candidate"
    if bool(row["classifier_in_scope"]):
        return "ACE_classification_negative"
    return "ACE_assay_required"


def build_ranked_candidates(table: pd.DataFrame, target_stage: str) -> pd.DataFrame:
    """Collapse target-stage occurrences into one row per ranked peptide."""
    target_occurrences = table[table["stage"] == target_stage].copy()
    locations = (
        target_occurrences.groupby("sequence", sort=False)
        .apply(
            lambda group: ";".join(
                f"{row.protein_id}:{int(row.start)}-{int(row.end)}"
                for row in group.itertuples()
            ),
            include_groups=False,
        )
        .rename("protein_locations")
    )
    ranked = (
        target_occurrences.sort_values("priority_rank")
        .drop_duplicates("sequence")
        .join(locations, on="sequence")
        .reset_index(drop=True)
    )
    ranked["priority_rank"] = ranked["priority_rank"].astype(int)
    ranked["candidate_role"] = ranked["known_ace_positive_exact_match"].map(
        {True: "positive_control_candidate", False: "novel_discovery_candidate"}
    )
    ranked["novel_candidate_rank"] = pd.NA
    novel_mask = ranked["candidate_role"] == "novel_discovery_candidate"
    ranked.loc[novel_mask, "novel_candidate_rank"] = range(1, int(novel_mask.sum()) + 1)
    ranked["control_rank"] = pd.NA
    control_mask = ranked["candidate_role"] == "positive_control_candidate"
    ranked.loc[control_mask, "control_rank"] = range(1, int(control_mask.sum()) + 1)
    if len(ranked) == 1:
        ranked["relative_lab_priority_percentile"] = 100.0
    else:
        ranked["relative_lab_priority_percentile"] = (
            100.0 * (len(ranked) - ranked["priority_rank"]) / (len(ranked) - 1)
        ).round(1)
    ranked["classifier_scope_note"] = ranked.apply(
        lambda row: (
            "in_scope_5_to_18_aa"
            if row["classifier_in_scope"]
            else f"not_applied_length_{int(row['length'])}_aa_model_scope_5_to_18_aa"
        ),
        axis=1,
    )
    ranked["ranking_reason"] = ranked.apply(_ranking_reason, axis=1)
    return ranked


def _ranking_reason(row: pd.Series) -> str:
    if pd.notna(row["measured_median_pIC50"]):
        return "exact_ACE_match_with_measured_pIC50"
    if bool(row["known_ace_positive_exact_match"]):
        return "exact_ACE_positive_evidence_match"
    if bool(row["classifier_in_scope"]):
        return "novel_candidate_ranked_by_model_predictions_and_size"
    if bool(row["regression_in_scope"]):
        return "novel_candidate_ranked_by_pIC50_model_and_size_classifier_not_applicable"
    return "insufficient_model_scope_ranked_last"


def main() -> None:
    parser = argparse.ArgumentParser(description="Staged theoretical GI digestion and ACE candidate ranking")
    parser.add_argument("--fasta", type=Path, required=True, help="Protein FASTA")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-stage", choices=["gastric", "intestinal"], default="intestinal")
    parser.add_argument(
        "--sequence-start",
        type=int,
        default=1,
        help="One-based first residue to digest; useful for removing a signal peptide",
    )
    parser.add_argument(
        "--sequence-end",
        type=int,
        default=None,
        help="One-based inclusive last residue to digest; defaults to the sequence end",
    )
    parser.add_argument(
        "--selection-count",
        type=int,
        default=0,
        help="Optional convenience shortlist size; 0 writes only the complete ranking",
    )
    args = parser.parse_args()

    fragments: list[Fragment] = []
    for protein_id, sequence in read_fasta(args.fasta):
        if args.sequence_start < 1:
            raise ValueError("--sequence-start must be at least 1")
        sequence_end = args.sequence_end or len(sequence)
        if sequence_end < args.sequence_start or sequence_end > len(sequence):
            raise ValueError("Selected sequence region is outside the FASTA sequence")
        selected_sequence = sequence[args.sequence_start - 1 : sequence_end]
        fragments.extend(staged_digest(protein_id, selected_sequence, args.sequence_start))
    table = annotate_and_rank(fragments, args.model_dir, args.data_dir, args.target_stage)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "all_staged_fragments.csv", index=False)
    ranked = build_ranked_candidates(table, args.target_stage)
    ranked.to_csv(args.output_dir / "ranked_ace_candidates.csv", index=False)
    if args.selection_count > 0:
        ranked.head(args.selection_count).to_csv(
            args.output_dir / f"selected_first_{args.selection_count}_candidates.csv", index=False
        )


if __name__ == "__main__":
    main()
