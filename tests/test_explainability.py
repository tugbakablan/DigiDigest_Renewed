import unittest
from pathlib import Path
import sys

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explainability import (  # noqa: E402
    classifier_explanation_card,
    explain_classifier,
    explain_regressor,
    reference_feature_vector,
    summarize_classifier_explanation,
)


class ExplainabilityTests(unittest.TestCase):
    def test_explanations_cover_feature_groups(self):
        root = Path(__file__).resolve().parents[1]
        classifier = joblib.load(root / "models" / "ace_classifier.joblib")
        regressor = joblib.load(root / "models" / "pic50_regressor.joblib")
        classifier_data = pd.read_csv(root / "output" / "classifier_dataset.csv")
        regression_data = pd.read_csv(root / "output" / "regression_dataset.csv")
        self.assertEqual(len(explain_classifier(classifier, "IIVFGRQLL", reference_feature_vector(classifier_data))), 5)
        self.assertEqual(len(explain_regressor(regressor, "IIVF", reference_feature_vector(regression_data))), 5)

    def test_classifier_summary_reports_probability_drivers_and_ic50_rule(self):
        rows = [
            {"Feature group": "C-terminal pattern", "Influence (percentage points)": 8.4},
            {"Feature group": "Dipeptide pattern", "Influence (percentage points)": -2.1},
        ]
        positive = summarize_classifier_explanation("ABCDE", 0.634, rows, predicted_ic50=24.4)
        self.assertIn("63.4% ACE probability", positive)
        self.assertIn("+8.4 percentage points", positive)
        self.assertIn("-2.1 percentage points", positive)
        self.assertIn("24.4 µM", positive)

        negative = summarize_classifier_explanation("ABCDE", 0.420, rows)
        self.assertIn("classification negative", negative)
        self.assertIn("No predicted IC50", negative)

    def test_classifier_card_is_compact_and_marks_borderline_results(self):
        rows = [
            {"Feature group": "N-terminal pattern", "Influence (percentage points)": 1.1},
            {"Feature group": "Amino-acid composition", "Influence (percentage points)": -6.3},
        ]
        card = classifier_explanation_card(0.505, rows, predicted_ic50=63.2)
        self.assertEqual(card["Decision"], "Predicted ACE candidate")
        self.assertEqual(card["Probability"], "50.5%")
        self.assertEqual(card["Threshold margin"], "+0.5 pp · Borderline")
        self.assertEqual(
            card["Main support"],
            "N-terminal pattern had the strongest positive effect, increasing the ACE probability by 1.1 percentage points.",
        )
        self.assertEqual(
            card["Main opposition"],
            "Amino-acid composition had the strongest negative effect, decreasing the ACE probability by 6.3 percentage points.",
        )
        self.assertEqual(card["Predicted potency"], "IC50 63.2 µM")
        self.assertIn("borderline", card["Laboratory interpretation"])
        self.assertIn("not a measured value", card["Potency interpretation"])


if __name__ == "__main__":
    unittest.main()
