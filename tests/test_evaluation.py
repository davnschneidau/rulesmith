"""Tests for evaluation system."""

import pytest

from rulesmith.evaluation.datasets import create_evaluation_slices, load_dataset, slice_dataset
from rulesmith.evaluation.scorers import accuracy_scorer, f1_scorer, precision_scorer, recall_scorer


class TestScorers:
    """Test evaluation scorers."""

    def test_accuracy_scorer(self):
        """Test accuracy scorer."""
        try:
            import pandas as pd

            data = pd.DataFrame({"target": [1, 0, 1, 0], "feature": [1, 2, 3, 4]})
            predictions = [1, 0, 1, 0]

            accuracy = accuracy_scorer(data, predictions, targets="target")
            assert accuracy == 1.0  # All correct

            predictions_wrong = [0, 0, 1, 0]
            accuracy = accuracy_scorer(data, predictions_wrong, targets="target")
            assert accuracy < 1.0  # Some wrong

        except ImportError:
            pytest.skip("pandas not available")


class TestDatasets:
    """Test dataset utilities."""

    def test_slice_dataset(self):
        """Test dataset slicing."""
        try:
            import pandas as pd

            data = pd.DataFrame({
                "feature": [1, 2, 3, 4],
                "category": ["A", "B", "A", "B"],
            })

            sliced = slice_dataset(data, {"category": "A"})
            assert len(sliced) == 2
            assert all(sliced["category"] == "A")

        except ImportError:
            pytest.skip("pandas not available")

