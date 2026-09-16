import importlib.util
from pathlib import Path
import unittest
import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "prepare_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_data", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PrepareDataTests(unittest.TestCase):
    def test_ic50_unit_conversions(self):
        self.assertEqual(MODULE.parse_ic50("1.5 mM", 500)["ic50_uM"], 1500.0)
        self.assertEqual(MODULE.parse_ic50("250 nM", 500)["ic50_uM"], 0.25)
        self.assertEqual(MODULE.parse_ic50("10 μg/mL", 500)["ic50_uM"], 20.0)
        self.assertEqual(MODULE.parse_ic50("1 mg/mL", 500)["ic50_uM"], 2000.0)

    def test_censored_values_are_not_regression_eligible(self):
        parsed = MODULE.parse_ic50(">1500 μM", 500)
        self.assertEqual(parsed["ic50_relation"], "greater_than")
        self.assertFalse(parsed["regression_eligible"])

    def test_database_absence_is_not_encoded_as_negative(self):
        self.assertTrue(MODULE.valid_standard_sequence("VPP"))
        self.assertFalse(MODULE.valid_standard_sequence("JNW"))


if __name__ == "__main__":
    unittest.main()
