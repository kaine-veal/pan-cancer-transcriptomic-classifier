"""
test_preprocessor.py

Unit tests for preprocessor.py — covers train/test splitting,
feature scaling, and PCA dimensionality reduction.
"""

import pytest
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "src")

from tcga_classifier.preprocessor import split_data, scale_features, apply_pca


# Fixtures

@pytest.fixture
def sample_data():
    """50 patients, 20 genes, 5 balanced tumour type classes."""
    np.random.seed(42)  # Fixed seed ensures the same random data is generated every run

    # X: feature matrix- rows are patients, columns are genes
    X = pd.DataFrame(
        np.random.rand(50, 20),
        columns=[f"gene_{i}" for i in range(20)]
    )

    # y: label vector- 5 tumour types repeated 10 times = 50 labels, one per patient
    y = pd.Series(["BRCA", "KIRC", "LUAD", "PRAD", "COAD"] * 10, name="Class")
    return X, y


@pytest.fixture
def scaled_data(sample_data):
    """Pre-split and scaled data ready for PCA tests."""
    X, y = sample_data

    # split first, then scale. Matches the real pipeline order: split -> scale -> PCA.
    X_train, X_test, _, _ = split_data(X, y)

    # scaler is fitted on training data only and applied to both train and test sets.
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
    return X_train_scaled, X_test_scaled


# Tests

def test_split_no_overlap(sample_data):
    """No sample should appear in both train and test sets."""
    X, y = sample_data

    # split into train and test sets, discard labels for this test, only need the feature matrices
    X_train, X_test, _, _ = split_data(X, y)

    # .index gives the row labels (patient IDs) for each set
    # .isdisjoint() returns True if the two sets have no elements in common
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_scaler_mean_near_zero(sample_data):
    """Training data mean should be approximately 0 after scaling."""
    X, y = sample_data
    X_train, X_test, _, _ = split_data(X, y)

    # scale_features fits StandardScaler on X_train, then transforms both sets
    # X_train_scaled should have mean ≈ 0 by definition of StandardScaler
    X_train_scaled, _, _ = scale_features(X_train, X_test)

    # floating point arithmetic means the mean won't be exactly 0.0- a small tolerance is needed
    assert abs(X_train_scaled.mean()) < 0.01


def test_pca_reduces_dimensions(scaled_data):
    """PCA output should have exactly n_components columns."""
    X_train_scaled, X_test_scaled = scaled_data

    # apply PCA requesting 5 components
    X_train_pca, _, _ = apply_pca(X_train_scaled, X_test_scaled, n_components=5)

    # .shape returns (rows, columns) — index [1] is the number of features (components)
    assert X_train_pca.shape[1] == 5
    