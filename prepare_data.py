from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def read_tsv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    for column in frame.columns:
        frame[column] = frame[column].astype(str).str.strip()
    return frame


def read_fasta(path: Path) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    header: str | None = None
    sequence_parts: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append({"record_id": header, "sequence": "".join(sequence_parts).upper()})
            header = line[1:].strip()
            sequence_parts = []
        else:
            sequence_parts.append(line)
    if header is not None:
        records.append({"record_id": header, "sequence": "".join(sequence_parts).upper()})
    return pd.DataFrame(records)


def first_number(value: object) -> float | None:
    match = NUMBER_RE.search(str(value).replace(",", "."))
    return float(match.group()) if match else None


def parse_ic50(raw_value: object, molecular_weight: object) -> dict[str, object]:
    raw = str(raw_value).strip()
    result: dict[str, object] = {
        "ic50_raw": raw,
        "ic50_numeric_raw": None,
        "ic50_unit_raw": "",
        "ic50_relation": "missing",
        "ic50_uM": None,
        "conversion_status": "missing",
        "regression_eligible": False,
    }
    if not raw or raw.upper() == "ND":
        return result

    normalized = raw.replace("µ", "μ").replace("Î¼", "μ")
    numbers = [float(item) for item in NUMBER_RE.findall(normalized.replace(",", "."))]
    if not numbers:
        result["conversion_status"] = "unparsed_no_number"
        return result

    if normalized.lstrip().startswith(">"):
        relation = "greater_than"
    elif normalized.lstrip().startswith("<"):
        relation = "less_than"
    elif re.search(r"\d\s*[-–]\s*\d", normalized):
        relation = "range"
    elif len(numbers) > 1:
        relation = "multiple_values"
    else:
        relation = "exact"

    lower = normalized.lower().replace(" ", "")
    if "%" in lower:
        unit = "%"
    elif "mg/ml" in lower:
        unit = "mg/mL"
    elif "μg/ml" in lower or "ug/ml" in lower:
        unit = "ug/mL"
    elif "nm" in lower:
        unit = "nM"
    elif "mm" in lower:
        unit = "mM"
    elif "μm/l" in lower:
        unit = "uM_per_L_label"
    elif "μm" in lower or "um" in lower:
        unit = "uM"
    else:
        unit = "unknown"

    value = numbers[0]
    mw = first_number(molecular_weight)
    converted: float | None = None
    status = "unsupported_unit"
    if unit in {"uM", "uM_per_L_label"}:
        converted = value
        status = "normalized_to_uM"
    elif unit == "mM":
        converted = value * 1000.0
        status = "converted_mM_to_uM"
    elif unit == "nM":
        converted = value / 1000.0
        status = "converted_nM_to_uM"
    elif unit == "ug/mL" and mw and mw > 0:
        converted = value * 1000.0 / mw
        status = "converted_mass_concentration_to_uM"
    elif unit == "mg/mL" and mw and mw > 0:
        converted = value * 1_000_000.0 / mw
        status = "converted_mass_concentration_to_uM"
    elif unit == "%":
        status = "percent_inhibition_not_ic50"

    eligible = relation == "exact" and converted is not None and converted > 0 and math.isfinite(converted)
    result.update(
        {
            "ic50_numeric_raw": value,
            "ic50_unit_raw": unit,
            "ic50_relation": relation,
            "ic50_uM": converted,
            "conversion_status": status,
            "regression_eligible": eligible,
        }
    )
    return result


def valid_standard_sequence(sequence: object) -> bool:
    text = str(sequence).strip().upper()
    return bool(text) and set(text).issubset(STANDARD_AA)


def summarize_source(name: str, frame: pd.DataFrame, sequence_column: str = "sequence") -> dict[str, object]:
    sequences = frame[sequence_column].astype(str).str.strip().str.upper()
    lengths = sequences.str.len()
    return {
        "source": name,
        "record_count": len(frame),
        "unique_sequence_count": sequences.nunique(),
        "duplicate_record_count": len(frame) - sequences.nunique(),
        "invalid_standard_sequence_count": int((~sequences.map(valid_standard_sequence)).sum()),
        "minimum_length": int(lengths.min()) if len(lengths) else None,
        "median_length": float(lengths.median()) if len(lengths) else None,
        "maximum_length": int(lengths.max()) if len(lengths) else None,
    }


def prepare_ic50(aht_ic50: pd.DataFrame, biopep: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, row in aht_ic50.iterrows():
        sequence = row["seq"].strip().upper()
        record = {
            "source_dataset": "AHTPDB_IC50",
            "source_record_id": row.get("id", ""),
            "sequence": sequence,
            "length": len(sequence),
            "molecular_weight_raw": row.get("molwt", ""),
            "biological_source": row.get("source", ""),
            "assay": row.get("assay", ""),
            "method": row.get("method", ""),
            "standard_sequence": valid_standard_sequence(sequence),
        }
        record.update(parse_ic50(row.get("ic50", ""), row.get("molwt", "")))
        records.append(record)

    for _, row in biopep.iterrows():
        sequence = str(row["sequence"]).strip().upper()
        raw_ic50 = row.get("ic50_um", "")
        record = {
            "source_dataset": "BIOPEP_previous",
            "source_record_id": row.get("external_id", ""),
            "sequence": sequence,
            "length": len(sequence),
            "molecular_weight_raw": row.get("chemical_mass", ""),
            "biological_source": "",
            "assay": "",
            "method": "",
            "standard_sequence": valid_standard_sequence(sequence),
        }
        if pd.isna(raw_ic50):
            parsed = parse_ic50("", row.get("chemical_mass", ""))
        else:
            parsed = parse_ic50(f"{raw_ic50} uM", row.get("chemical_mass", ""))
        record.update(parsed)
        records.append(record)

    output = pd.DataFrame(records)
    output["pIC50"] = output["ic50_uM"].map(
        lambda value: 6.0 - math.log10(value) if pd.notna(value) and value > 0 else None
    )
    output["exact_sequence_source_key"] = output["source_dataset"] + ":" + output["source_record_id"].astype(str)
    return output.sort_values(["sequence", "source_dataset", "source_record_id"], kind="stable")


def build_positive_catalog(
    aht_all: pd.DataFrame,
    biopep: pd.DataFrame,
    ai_positive: pd.DataFrame,
    ic50_clean: pd.DataFrame,
) -> pd.DataFrame:
    evidence: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    for sequence in aht_all["seq"].astype(str).str.strip().str.upper():
        evidence[sequence].add("AHTPDB")
        counts[sequence] += 1
    for sequence in biopep["sequence"].astype(str).str.strip().str.upper():
        evidence[sequence].add("BIOPEP_previous")
        counts[sequence] += 1
    for sequence in ai_positive["sequence"].astype(str).str.strip().str.upper():
        evidence[sequence].add("AI4ACEIP_positive")
        counts[sequence] += 1

    exact_ic50 = set(
        ic50_clean.loc[ic50_clean["regression_eligible"].astype(bool), "sequence"].astype(str)
    )
    rows = []
    for sequence in sorted(evidence):
        rows.append(
            {
                "sequence": sequence,
                "length": len(sequence),
                "standard_sequence": valid_standard_sequence(sequence),
                "positive_evidence_sources": ";".join(sorted(evidence[sequence])),
                "positive_source_count": len(evidence[sequence]),
                "source_record_count": counts[sequence],
                "has_regression_eligible_ic50": sequence in exact_ic50,
                "label_status": "positive_evidence",
            }
        )
    return pd.DataFrame(rows)


def audit_ai4aceip_labels(
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    aht_sequences: set[str],
    biopep_sequences: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, frame in [(1, positives), (0, negatives)]:
        grouped = frame.groupby("sequence", sort=True).size()
        for sequence, occurrence_count in grouped.items():
            in_aht = sequence in aht_sequences
            in_biopep = sequence in biopep_sequences
            conflict = label == 0 and (in_aht or in_biopep)
            rows.append(
                {
                    "sequence": sequence,
                    "length": len(sequence),
                    "ai4aceip_label": label,
                    "ai4aceip_occurrence_count": int(occurrence_count),
                    "standard_sequence": valid_standard_sequence(sequence),
                    "present_in_ahtpdb_positive": in_aht,
                    "present_in_biopep_previous_positive": in_biopep,
                    "label_conflict": conflict,
                    "negative_evidence_status": (
                        "conflicts_with_positive_evidence"
                        if conflict
                        else "candidate_negative_not_experimentally_confirmed"
                        if label == 0
                        else "not_applicable"
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["ai4aceip_label", "sequence"], ascending=[False, True])


def build_fragment_catalog(digestion: pd.DataFrame, positive_sequences: set[str]) -> pd.DataFrame:
    aggregates: dict[str, dict[str, object]] = {}
    for _, row in digestion.iterrows():
        parent = str(row.get("sequence", "")).strip().upper()
        fragments_raw = row.get("digestion_fragments", "")
        if pd.isna(fragments_raw):
            continue
        known_ace = {
            item.strip().upper()
            for item in str(row.get("known_ace_active_fragments", "")).split(",")
            if item.strip() and str(row.get("known_ace_active_fragments", "")).lower() != "nan"
        }
        for fragment in [item.strip().upper() for item in str(fragments_raw).split(",") if item.strip()]:
            item = aggregates.setdefault(
                fragment,
                {
                    "fragment": fragment,
                    "length": len(fragment),
                    "parent_sequences": set(),
                    "occurrence_count": 0,
                    "old_biopep_known_ace_fragment": False,
                },
            )
            item["parent_sequences"].add(parent)
            item["occurrence_count"] += 1
            item["old_biopep_known_ace_fragment"] = bool(
                item["old_biopep_known_ace_fragment"] or fragment in known_ace
            )

    rows = []
    for fragment, item in aggregates.items():
        rows.append(
            {
                "fragment": fragment,
                "length": item["length"],
                "standard_sequence": valid_standard_sequence(fragment),
                "parent_count": len(item["parent_sequences"]),
                "occurrence_count": item["occurrence_count"],
                "parent_sequences": ";".join(sorted(item["parent_sequences"])),
                "old_biopep_known_ace_fragment": item["old_biopep_known_ace_fragment"],
                "present_in_current_positive_catalog": fragment in positive_sequences,
                "activity_evidence_status": (
                    "positive_evidence"
                    if item["old_biopep_known_ace_fragment"] or fragment in positive_sequences
                    else "unlabeled_not_negative"
                ),
                "eligible_for_peptide_activity_scoring": len(fragment) >= 2,
                "digestion_evidence_type": "BIOPEP_rule_based_combined_enzyme_simulation",
            }
        )
    return pd.DataFrame(rows).sort_values(["length", "fragment"], kind="stable")


def write_report(
    output_path: Path,
    source_summary: pd.DataFrame,
    positive_catalog: pd.DataFrame,
    ic50_clean: pd.DataFrame,
    label_audit: pd.DataFrame,
    fragments: pd.DataFrame,
) -> None:
    conflicts = label_audit[label_audit["label_conflict"]]
    candidate_negatives = label_audit[
        (label_audit["ai4aceip_label"] == 0) & (~label_audit["label_conflict"])
    ]
    exact_ic50 = ic50_clean[ic50_clean["regression_eligible"]]
    fragment_positive = fragments[fragments["activity_evidence_status"] == "positive_evidence"]
    fragment_unlabeled = fragments[fragments["activity_evidence_status"] == "unlabeled_not_negative"]

    text = f"""# ACE peptide data quality report

## Prepared collections

- Positive-evidence catalogue: {len(positive_catalog):,} unique sequences
- Regression-eligible IC50 records: {len(exact_ic50):,} records across {exact_ic50['sequence'].nunique():,} unique sequences
- AI4ACEIP candidate negatives after exact positive-conflict removal: {len(candidate_negatives):,} unique sequences
- AI4ACEIP exact negative/positive-evidence conflicts: {len(conflicts):,} unique sequences
- Previous DigiDigest theoretical fragments: {len(fragments):,} unique sequences
- Theoretical fragments with positive ACE evidence: {len(fragment_positive):,}
- Theoretical fragments with unknown activity: {len(fragment_unlabeled):,}

## Decisions

1. Do not treat database absence as a negative label.
2. Do not train the activity model on single-residue digestion products.
3. Keep repeated IC50 measurements at record level; assay context can change the value.
4. Exclude censored, ranged, percent-inhibition and unparsed values from the first IC50 regression model.
5. Treat AI4ACEIP negatives as candidate negatives until their evidence provenance is verified.
6. Match or reweight positive and negative samples by peptide length before classification.
7. Split model development data by sequence-similarity clusters, not random rows.
8. Treat previous BIOPEP digestion as rule-based reference output, not experimental validation.

## Next build step

Create two separately evaluated baselines:

- ACE activity classifier using positive evidence and a reviewed, length-matched negative subset.
- pIC50 regression model using only exact, normalized, positive IC50 records.

The digestion engine should remain a separate module. It will digest full FASTA proteins stage by stage and send the resulting peptides to the activity models.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    aht_small = read_tsv(input_dir / "ahtpdb_small_peptides.txt")
    aht_long = read_tsv(input_dir / "ahtpdb_long_peptides.txt")
    aht_ic50 = read_tsv(input_dir / "ahtpdb_ic50.txt")
    aht_all = pd.concat([aht_small, aht_long], ignore_index=True)

    biopep = pd.read_excel(input_dir / "step1_biopep_with_ic50.xlsx")
    toxicity = pd.read_excel(input_dir / "02_ACE_TOXICITY(1).xlsx")
    digestion = pd.read_excel(input_dir / "03_ACE_WITH_DIGESTION_FRAGMENTS(2).xlsx")
    ai_positive = read_fasta(input_dir / "pos.fa")
    ai_negative = read_fasta(input_dir / "neg.fa")

    for frame in [biopep, toxicity, digestion]:
        frame.columns = [str(column).strip() for column in frame.columns]
    for frame in [ai_positive, ai_negative]:
        frame["sequence"] = frame["sequence"].astype(str).str.strip().str.upper()

    source_summary = pd.DataFrame(
        [
            summarize_source("AHTPDB_small", aht_small, "seq"),
            summarize_source("AHTPDB_long", aht_long, "seq"),
            summarize_source("AHTPDB_ic50", aht_ic50, "seq"),
            summarize_source("BIOPEP_previous", biopep),
            summarize_source("BIOPEP_toxicity_enriched", toxicity),
            summarize_source("BIOPEP_digestion_enriched", digestion),
            summarize_source("AI4ACEIP_positive", ai_positive),
            summarize_source("AI4ACEIP_negative", ai_negative),
        ]
    )

    ic50_clean = prepare_ic50(aht_ic50, toxicity)
    positive_catalog = build_positive_catalog(aht_all, biopep, ai_positive, ic50_clean)
    positive_sequences = set(positive_catalog["sequence"])
    label_audit = audit_ai4aceip_labels(
        ai_positive,
        ai_negative,
        set(aht_all["seq"].astype(str).str.strip().str.upper()),
        set(biopep["sequence"].astype(str).str.strip().str.upper()),
    )
    conflicts = label_audit[label_audit["label_conflict"]].copy()
    fragments = build_fragment_catalog(digestion, positive_sequences)

    source_summary.to_csv(output_dir / "source_summary.csv", index=False)
    positive_catalog.to_csv(output_dir / "positive_evidence_catalog.csv", index=False)
    ic50_clean.to_csv(output_dir / "ic50_records_clean.csv", index=False)
    label_audit.to_csv(output_dir / "ai4aceip_label_audit.csv", index=False)
    conflicts.to_csv(output_dir / "ai4aceip_negative_conflicts.csv", index=False)
    fragments.to_csv(output_dir / "digestion_fragment_catalog.csv", index=False)
    write_report(
        output_dir / "data_quality_report.md",
        source_summary,
        positive_catalog,
        ic50_clean,
        label_audit,
        fragments,
    )


if __name__ == "__main__":
    main()

