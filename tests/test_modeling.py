"""Testes unitários mínimos das regras de preparação e feature engineering."""

import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modeling import add_features, load_and_prepare  # noqa: E402


class ModelingTests(unittest.TestCase):
    def test_feature_engineering(self):
        frame = pd.DataFrame(
            {
                "total sulfur dioxide": [50.0],
                "free sulfur dioxide": [10.0],
            }
        )
        result = add_features(frame)
        self.assertAlmostEqual(result.loc[0, "bound sulfur dioxide"], 40.0)
        self.assertAlmostEqual(result.loc[0, "free to total sulfur ratio"], 0.2)

    def test_target_id_and_duplicate_rules(self):
        rows = pd.DataFrame(
            {
                "fixed acidity": [7.0, 7.0, 8.0],
                "volatile acidity": [0.5, 0.5, 0.3],
                "citric acid": [0.1, 0.1, 0.4],
                "residual sugar": [2.0, 2.0, 2.2],
                "chlorides": [0.08, 0.08, 0.07],
                "free sulfur dioxide": [10.0, 10.0, 12.0],
                "total sulfur dioxide": [30.0, 30.0, 34.0],
                "density": [0.996, 0.996, 0.995],
                "pH": [3.3, 3.3, 3.2],
                "sulphates": [0.6, 0.6, 0.8],
                "alcohol": [10.0, 10.0, 11.5],
                "quality": [6, 6, 7],
                "Id": [1, 2, 3],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.csv"
            rows.to_csv(path, index=False)
            X, y, audit = load_and_prepare(path)
        self.assertNotIn("Id", X.columns)
        self.assertEqual(len(X), 2)
        self.assertEqual(audit["duplicates_removed"], 1)
        self.assertEqual(y.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()

