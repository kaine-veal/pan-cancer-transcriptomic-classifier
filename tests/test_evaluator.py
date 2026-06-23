"""
test_evaluator.py

Unit tests for evaluator.py- covers metric bounds,
confusion matrix shape, and prediction count.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.insert(0, "src")  

from tcga_classifier.evaluator import evaluate_model


# Fixtures

@pytest.fixture
def trained_model_and_test_data(tmp_path):
    """Train a small Random Forest and return it alongside a test set."""
    np.random.seed(42)

    # X: 50 patients, 10 features- small enough to train instantly in tests
    X_train = np.random.rand(50, 10)
    X_test  = np.random.rand(20, 10)

    # y: 5 tumour types, balanced across both sets
    y_train = pd.Series(["BRCA", "COAD", "KIRC", "LUAD", "PRAD"] * 10)
    y_test  = pd.Series(["BRCA", "COAD", "KIRC", "LUAD", "PRAD"] * 4)

    # fit a small Random Forest — n_estimators=10 keeps it fast
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # evaluate_model saves plots in tmp_path
    return model, X_test, y_test, str(tmp_path)


# Tests 

def test_macro_f1_between_zero_and_one(trained_model_and_test_data):
    """Macro F1 score must be a valid metric value between 0.0 and 1.0."""
    model, X_test, y_test, output_dir = trained_model_and_test_data

    # evaluate_model returns a dict of metrics- unpack macro_f1
    metrics = evaluate_model(model, X_test, y_test, output_dir=output_dir)

    # any value outside 0–1 would indicate a calculation error
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_confusion_matrix_is_5x5(trained_model_and_test_data):
    """Confusion matrix should be 5x5 — one row and column per tumour type."""
    model, X_test, y_test, output_dir = trained_model_and_test_data
    metrics = evaluate_model(model, X_test, y_test, output_dir=output_dir)

    # .shape returns (rows, columns) — should be (5, 5) for a 5-class problem
    assert metrics["confusion_matrix"].shape == (5, 5)


def test_prediction_count_matches_test_set(trained_model_and_test_data):
    """Number of predictions should equal number of test samples."""
    model, X_test, y_test, output_dir = trained_model_and_test_data
    metrics = evaluate_model(model, X_test, y_test, output_dir=output_dir)

    # one prediction per patient- length mismatch would cause downstream errors
    assert len(metrics["y_pred"]) == len(y_test)
