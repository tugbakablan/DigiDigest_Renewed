import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui_helpers import (  # noqa: E402
    format_ace_probability,
    format_ace_status,
    normalize_protein_sequence,
    parse_fasta_text,
    split_single_fasta,
)


class AppInputTests(unittest.TestCase):
    def test_fasta_parser_contract(self):
        self.assertEqual(parse_fasta_text(">protein one\nACDE\nFG\n"), [("protein one", "ACDEFG")])

    def test_fasta_requires_header(self):
        with self.assertRaises(ValueError):
            parse_fasta_text("ACDEFG")

    def test_split_and_normalize_inputs(self):
        self.assertEqual(split_single_fasta(">Protein one\nAC DE\n"), ("Protein one", "ACDE"))
        self.assertEqual(normalize_protein_sequence("ac de\nfg"), "ACDEFG")

    def test_ace_probability_is_visible_when_classifier_runs(self):
        self.assertEqual(format_ace_probability(0.6342), "63.4%")
        self.assertEqual(format_ace_probability(None), "N/A")
        self.assertEqual(format_ace_probability(float("nan")), "N/A")

    def test_ace_status_is_separate_from_probability(self):
        self.assertEqual(format_ace_status(False, True, True), "Predicted ACE candidate")
        self.assertEqual(format_ace_status(False, True, False), "Classification negative")
        self.assertEqual(format_ace_status(False, False, False), "ACE assay required")
        self.assertEqual(format_ace_status(True, False, False), "Known ACE — experimental")


if __name__ == "__main__":
    unittest.main()
