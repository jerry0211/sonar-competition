"""Sanity tests for the metric suite. Run: pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sonar_toolkit.validation import recall_at_fpr, evaluate


def test_perfect_separation():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    assert recall_at_fpr(y, score, max_fpr=0.0) == 1.0
    assert evaluate(y, score)["auc"] == 1.0


def test_zero_budget_when_overlapping():
    y = np.array([0, 1, 0, 1])
    score = np.array([0.5, 0.5, 0.5, 0.5])  # no separation
    assert recall_at_fpr(y, score, max_fpr=0.0) == 0.0


def test_single_class_auc_is_nan():
    y = np.array([1, 1, 1])
    score = np.array([0.2, 0.6, 0.9])
    assert np.isnan(evaluate(y, score)["auc"])
