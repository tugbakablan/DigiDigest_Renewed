import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from digest_and_rank import (  # noqa: E402
    Fragment,
    _ranking_reason,
    annotate_and_rank,
    cleavage_products,
    chymotrypsin_high_cut,
    staged_digest,
    trypsin_cut,
)


class DigestionTests(unittest.TestCase):
    def test_trypsin_blocks_cleavage_before_proline(self):
        parent = Fragment("p", "AKPQR", 1, 5, "input", "none")
        products = cleavage_products(parent, [trypsin_cut], "intestinal", "trypsin")
        self.assertEqual([item.sequence for item in products], ["AKPQR"])

    def test_combined_intestinal_enzymes_and_coordinates(self):
        parent = Fragment("p", "AAKFFR", 11, 16, "gastric", "pepsin")
        products = cleavage_products(
            parent, [trypsin_cut, chymotrypsin_high_cut], "intestinal", "combined"
        )
        self.assertEqual([(x.sequence, x.start, x.end) for x in products], [("AAK", 11, 13), ("F", 14, 14), ("F", 15, 15), ("R", 16, 16)])

    def test_staged_digest_preserves_parent_coordinates(self):
        products = staged_digest("wheat", "AAFLKQ")
        gastric = [item for item in products if item.stage == "gastric"]
        intestinal = [item for item in products if item.stage == "intestinal"]
        self.assertEqual([(x.sequence, x.start, x.end) for x in gastric], [("AAF", 1, 3), ("L", 4, 4), ("KQ", 5, 6)])
        self.assertEqual(intestinal[-1].end, 6)

    def test_invalid_residue_is_rejected(self):
        with self.assertRaises(ValueError):
            staged_digest("bad", "ABZ")

    def test_ranking_reason_distinguishes_measurement_and_model(self):
        import pandas as pd

        measured = pd.Series({"measured_median_pIC50": 4.0, "known_ace_positive_exact_match": True, "classifier_in_scope": False, "regression_in_scope": True})
        novel = pd.Series({"measured_median_pIC50": None, "known_ace_positive_exact_match": False, "classifier_in_scope": True, "regression_in_scope": True})
        self.assertEqual(_ranking_reason(measured), "exact_ACE_match_with_measured_pIC50")
        self.assertEqual(_ranking_reason(novel), "novel_candidate_ranked_by_model_predictions_and_size")

    def test_selected_region_keeps_original_coordinates(self):
        products = staged_digest("mature", "AAFL", coordinate_start=21)
        gastric = [item for item in products if item.stage == "gastric"]
        self.assertEqual([(x.sequence, x.start, x.end) for x in gastric], [("AAF", 21, 23), ("L", 24, 24)])

    def test_ic50_is_gated_by_ace_support(self):
        root = Path(__file__).resolve().parents[1]
        table = annotate_and_rank(
            staged_digest("test", "IIVFGRQLL"), root / "models", root / "output", "intestinal"
        )
        unknown_short = table[(table.sequence == "IIVF") & (table.stage == "intestinal")].iloc[0]
        known_short = table[(table.sequence == "GR") & (table.stage == "intestinal")].iloc[0]
        self.assertEqual(unknown_short.ace_assessment_status, "ACE_assay_required")
        self.assertTrue(__import__("pandas").isna(unknown_short.predicted_IC50_uM_exploratory))
        self.assertTrue(known_short.known_ace_positive_exact_match)
        self.assertTrue(__import__("pandas").isna(known_short.predicted_IC50_uM_exploratory))
        self.assertFalse(__import__("pandas").isna(known_short.measured_median_IC50_uM))


if __name__ == "__main__":
    unittest.main()
