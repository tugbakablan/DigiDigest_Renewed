import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from history_store import delete_analysis, list_analyses, load_analysis, save_analysis  # noqa: E402


class HistoryStoreTests(unittest.TestCase):
    def test_history_round_trip(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            database = Path(directory) / "history.sqlite3"
            fragments = pd.DataFrame([{"sequence": "ACD", "stage": "gastric"}])
            ranked = pd.DataFrame([{"sequence": "ACD", "priority_rank": 1}])
            analysis_id = save_analysis(database, "test protein", "ACD", 1, 3, "gastric", fragments, ranked)
            self.assertEqual(list_analyses(database).iloc[0]["protein_name"], "test protein")
            loaded = load_analysis(database, analysis_id)
            self.assertEqual(loaded["sequence"], "ACD")
            self.assertEqual(loaded["ranked"].iloc[0]["priority_rank"], 1)
            delete_analysis(database, analysis_id)
            self.assertTrue(list_analyses(database).empty)


if __name__ == "__main__":
    unittest.main()
