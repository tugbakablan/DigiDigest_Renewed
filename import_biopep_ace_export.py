from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
SOURCE_NAME = "BIOPEP_UWM_2026_export"


def read_biopep_export(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(path, skiprows=3)
    required = {"ID", "Name", "Sequence", "Activity"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing BIOPEP columns: {', '.join(sorted(missing))}")
    raw["Sequence"] = raw["Sequence"].astype(str).str.strip().str.upper()
    raw["standard_sequence"] = raw["Sequence"].map(lambda value: bool(value) and set(value) <= STANDARD_AA)
    raw["length"] = raw["Sequence"].str.len()
    valid = raw[raw["standard_sequence"]].copy()
    invalid = raw[~raw["standard_sequence"]].copy()
    return valid, invalid


def merge_positive_catalog(catalog: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    record_counts = valid.groupby("Sequence").size().to_dict()
    result = catalog.copy().set_index("sequence", drop=False)
    for sequence, count in record_counts.items():
        if sequence in result.index:
            sources = set(str(result.at[sequence, "positive_evidence_sources"]).split(";"))
            if SOURCE_NAME not in sources:
                sources.add(SOURCE_NAME)
                result.at[sequence, "positive_source_count"] = int(result.at[sequence, "positive_source_count"]) + 1
            result.at[sequence, "positive_evidence_sources"] = ";".join(sorted(sources))
            result.at[sequence, "source_record_count"] = int(result.at[sequence, "source_record_count"]) + int(count)
        else:
            result.loc[sequence] = {
                "sequence": sequence,
                "length": len(sequence),
                "standard_sequence": True,
                "positive_evidence_sources": SOURCE_NAME,
                "positive_source_count": 1,
                "source_record_count": int(count),
                "has_regression_eligible_ic50": False,
                "label_status": "positive_evidence",
            }
    return result.reset_index(drop=True).sort_values("sequence").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a BIOPEP-UWM ACE activity export")
    parser.add_argument("--xlsx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    valid, invalid = read_biopep_export(args.xlsx)
    args.source_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    valid.to_csv(args.source_dir / "biopep_uwm_ace_export_clean.csv", index=False)
    invalid.to_csv(args.source_dir / "biopep_uwm_ace_export_nonstandard.csv", index=False)

    catalog_path = args.output_dir / "positive_evidence_catalog.csv"
    before = pd.read_csv(catalog_path)
    after = merge_positive_catalog(before, valid)
    after.to_csv(catalog_path, index=False)

    summary_path = args.output_dir / "source_summary.csv"
    summary = pd.read_csv(summary_path)
    summary = summary[summary["source"] != SOURCE_NAME]
    summary.loc[len(summary)] = {
        "source": SOURCE_NAME,
        "record_count": len(valid),
        "unique_sequence_count": valid["Sequence"].nunique(),
        "duplicate_record_count": len(valid) - valid["Sequence"].nunique(),
        "invalid_standard_sequence_count": len(invalid),
        "minimum_length": int(valid["length"].min()),
        "median_length": float(valid["length"].median()),
        "maximum_length": int(valid["length"].max()),
    }
    summary.to_csv(summary_path, index=False)

    new_sequences = set(after["sequence"]) - set(before["sequence"])
    report = f"""# BIOPEP-UWM ACE export import

- Source: https://biochemia.uwm.edu.pl/biopep-uwm/
- Export rows: {len(valid) + len(invalid):,}
- Standard peptide rows: {len(valid):,}
- Unique standard peptides: {valid['Sequence'].nunique():,}
- Non-standard or modified sequences retained separately: {len(invalid):,}
- New positive sequences added to the evidence catalog: {len(new_sequences):,}
- Total positive evidence catalog after import: {len(after):,}
- Length 3: {valid.loc[valid['length'] == 3, 'Sequence'].nunique():,} positive, 0 candidate-negative in this export
- Length 4: {valid.loc[valid['length'] == 4, 'Sequence'].nunique():,} positive, 0 candidate-negative in this export

This export strengthens exact positive-evidence matching. It cannot by itself
support a binary 3–18 aa classifier because every included activity record is
ACE inhibitor and no short negative class is provided.
"""
    (args.report_dir / "biopep_uwm_import_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
