import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from import_biopep_ace_export import merge_positive_catalog  # noqa: E402


class BiopepImportTests(unittest.TestCase):
    def test_merge_adds_new_and_updates_existing(self):
        catalog = pd.DataFrame([{
            "sequence": "VMP", "length": 3, "standard_sequence": True,
            "positive_evidence_sources": "AHTPDB", "positive_source_count": 1,
            "source_record_count": 1, "has_regression_eligible_ic50": True,
            "label_status": "positive_evidence",
        }])
        valid = pd.DataFrame({"Sequence": ["VMP", "IIVF", "IIVF"]})
        merged = merge_positive_catalog(catalog, valid).set_index("sequence")
        self.assertEqual(len(merged), 2)
        self.assertIn("BIOPEP_UWM_2026_export", merged.at["VMP", "positive_evidence_sources"])
        self.assertEqual(merged.at["IIVF", "source_record_count"], 2)


if __name__ == "__main__":
    unittest.main()
