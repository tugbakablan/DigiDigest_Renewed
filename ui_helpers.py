from __future__ import annotations

import math
import re


def parse_fasta_text(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = None
    parts: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(parts).upper()))
            header = line[1:].strip() or "protein"
            parts = []
        elif header is None:
            raise ValueError("A FASTA record must include a header beginning with >.")
        else:
            parts.append(line.replace(" ", ""))
    if header is not None:
        records.append((header, "".join(parts).upper()))
    if not records:
        raise ValueError("No valid FASTA record was found.")
    return records


def normalize_protein_sequence(text: str) -> str:
    sequence = re.sub(r"\s+", "", text).upper()
    if not sequence:
        raise ValueError("The amino-acid sequence cannot be empty.")
    return sequence


def split_single_fasta(text: str) -> tuple[str, str]:
    records = parse_fasta_text(text)
    if len(records) != 1:
        raise ValueError("Upload exactly one protein FASTA record at a time.")
    return records[0]


def format_ace_probability(value: object) -> str:
    """Format a classifier probability without implying a result when no model ran."""
    if value is None:
        return "N/A"
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if math.isnan(probability):
        return "N/A"
    return f"{100 * probability:.1f}%"


def format_ace_status(
    known_ace: bool,
    classifier_in_scope: bool,
    classifier_positive: bool,
) -> str:
    """Keep experimental evidence and model classification visibly distinct."""
    if known_ace:
        return "Known ACE — experimental"
    if not classifier_in_scope:
        return "ACE assay required"
    if classifier_positive:
        return "Predicted ACE candidate"
    return "Classification negative"
