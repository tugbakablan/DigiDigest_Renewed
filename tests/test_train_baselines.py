import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "train_baselines.py"
SPEC = importlib.util.spec_from_file_location("train_baselines", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TrainBaselineTests(unittest.TestCase):
    def test_feature_shape_is_stable(self):
        first = MODULE.sequence_features("VPP")
        second = MODULE.sequence_features("AHEPVK")
        self.assertEqual(first.shape, second.shape)
        self.assertTrue(np.isfinite(first).all())

    def test_bounded_levenshtein(self):
        self.assertEqual(MODULE.levenshtein_with_limit("VPP", "IPP", 1), 1)
        self.assertGreater(MODULE.levenshtein_with_limit("VPP", "AAAA", 1), 1)

    def test_similar_sequences_share_cluster(self):
        groups = MODULE.cluster_sequences(["AHEPVK", "AHEPVQ", "GGGGGG"], 0.8)
        self.assertEqual(groups["AHEPVK"], groups["AHEPVQ"])
        self.assertNotEqual(groups["AHEPVK"], groups["GGGGGG"])


if __name__ == "__main__":
    unittest.main()
