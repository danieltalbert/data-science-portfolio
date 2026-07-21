import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.property_age_model import EXCLUDED_FEATURES, prepare_features, run_analysis


def synthetic_data(rows: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    year = rng.integers(1940, 2020, size=rows)
    return pd.DataFrame(
        {
            "parcel": [f"P{index:04}" for index in range(rows)],
            "yrbuilt": year,
            "before1980": (year < 1980).astype(int),
            "livearea": np.maximum(500, (year - 1900) * 12 + rng.normal(0, 90, rows)),
            "stories": rng.integers(1, 4, size=rows),
            "numbaths": rng.integers(1, 5, size=rows),
            "quality_C": (year < 1985).astype(int),
            "arcstyle_ONE-STORY": rng.integers(0, 2, size=rows),
        }
    )


class PropertyAgeModelTests(unittest.TestCase):
    def test_feature_boundary_removes_identifiers_and_leakage(self) -> None:
        features, target = prepare_features(synthetic_data())
        self.assertTrue(EXCLUDED_FEATURES.isdisjoint(features.columns))
        self.assertEqual(len(features), len(target))

    def test_missing_required_column_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            prepare_features(synthetic_data().drop(columns="yrbuilt"))

    def test_analysis_is_reproducible_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_analysis(synthetic_data(), temp_dir)
            self.assertGreaterEqual(result["metrics"]["accuracy"], 0.75)
            for filename in (
                "metrics.json",
                "confusion-matrix.png",
                "feature-importance.png",
                "build-year-distribution.png",
            ):
                self.assertTrue((Path(temp_dir) / filename).is_file(), filename)


if __name__ == "__main__":
    unittest.main()
