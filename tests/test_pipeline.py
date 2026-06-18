"""
test_pipeline.py

Pytest test suite for the Pan-Cancer Transcriptomic Subtyping pipeline.
Covers boundary testing and core validation across all pipeline modules.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.insert(0, "src")

from tcga_classifier.data_loader import load_data
from tcga_classifier.preprocessor import split_data, scale_features, apply_pca
from tcga_classifier.trainer import save_model, load_model
from tcga_classifier.evaluator import evaluate_model


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    """Minimal expression matrix and labels — 50 patients, 50 genes."""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.rand(50, 50),
        index=[f"sample_{i}" for i in range(50)],
        columns=[f"gene_{i}" for i in range(50)]
    )
    y = pd.Series(
        ["BRCA", "KIRC", "LUAD", "PRAD", "COAD"] * 10,
        index=X.index,
        name="Class"
    )
    return X, y


@pytest.fixture
def split_scaled(sample_data):
    """Pre-split and scaled data for downstream tests."""
    X, y = sample_data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


@pytest.fixture
def trained_model(split_scaled):
    """Fitted Random Forest for evaluator tests."""
    X_train, _, y_train, _ = split_scaled
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model


# ── data_loader.py ─────────────────────────────────────────────────────────────

def test_missing_file_raises(tmp_path):
    """Missing data file should raise FileNotFoundError immediately."""
    with pytest.raises(FileNotFoundError):
        load_data("missing.csv", "missing_labels.csv")


def test_mismatched_index_raises(tmp_path, sample_data):
    """Mismatched sample IDs between files should raise ValueError."""
    X, y = sample_data
    data_path   = tmp_path / "data.csv"
    labels_path = tmp_path / "labels.csv"

    X.to_csv(data_path)
    mismatched = y.copy()
    mismatched.index = [f"patient_{i}" for i in range(50)]
    mismatched.to_frame().to_csv(labels_path)

    with pytest.raises(ValueError):
        load_data(str(data_path), str(labels_path))


def test_y_is_one_dimensional(tmp_path, sample_data):
    """squeeze() should produce a 1D Series for sklearn compatibility."""
    X, y = sample_data
    data_path   = tmp_path / "data.csv"
    labels_path = tmp_path / "labels.csv"

    X.to_csv(data_path)
    y.to_frame().to_csv(labels_path)

    _, y_loaded = load_data(str(data_path), str(labels_path))
    assert y_loaded.ndim == 1


# ── preprocessor.py ────────────────────────────────────────────────────────────

def test_split_no_overlap(sample_data):
    """No sample should appear in both train and test sets."""
    X, y = sample_data
    X_train, X_test, _, _ = split_data(X, y)
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_scaler_mean_near_zero(split_scaled):
    """Training data mean should be approximately 0 after scaling."""
    X_train_scaled, _, _, _ = split_scaled
    assert abs(X_train_scaled.mean()) < 0.01


def test_scaler_std_near_one(split_scaled):
    """Training data std should be approximately 1 after scaling."""
    X_train_scaled, _, _, _ = split_scaled
    assert abs(X_train_scaled.std() - 1.0) < 0.01


def test_pca_reduces_dimensions(split_scaled):
    """PCA should reduce features to n_components columns."""
    X_train_scaled, X_test_scaled, _, _ = split_scaled
    X_train_pca, _, _ = apply_pca(X_train_scaled, X_test_scaled, n_components=5)
    assert X_train_pca.shape[1] == 5


def test_pca_boundary_single_component(split_scaled):
    """Boundary: n_components=1 should return a single column."""
    X_train_scaled, X_test_scaled, _, _ = split_scaled
    X_train_pca, _, _ = apply_pca(X_train_scaled, X_test_scaled, n_components=1)
    assert X_train_pca.shape[1] == 1


# ── trainer.py ─────────────────────────────────────────────────────────────────

def test_save_creates_file(tmp_path, trained_model):
    """save_model should create a .pkl file at the specified path."""
    path = tmp_path / "model.pkl"
    save_model(trained_model, str(path))
    assert path.exists()


def test_load_missing_model_raises(tmp_path):
    """Loading a missing model file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_model(str(tmp_path / "missing.pkl"))


def test_save_load_predictions_match(tmp_path, trained_model, split_scaled):
    """Reloaded model should produce identical predictions to the original."""
    _, X_test, _, _ = split_scaled
    path = tmp_path / "model.pkl"

    original_preds = trained_model.predict(X_test)
    save_model(trained_model, str(path))
    loaded_preds = load_model(str(path)).predict(X_test)

    np.testing.assert_array_equal(original_preds, loaded_preds)


# ── evaluator.py ───────────────────────────────────────────────────────────────

def test_f1_boundary_between_zero_and_one(tmp_path, trained_model, split_scaled):
    """Boundary: macro F1 must be between 0.0 and 1.0 inclusive."""
    _, X_test, _, y_test = split_scaled
    metrics = evaluate_model(trained_model, X_test, y_test, output_dir=str(tmp_path))
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_confusion_matrix_shape(tmp_path, trained_model, split_scaled):
    """Confusion matrix should be 5x5 for a 5-class problem."""
    _, X_test, _, y_test = split_scaled
    metrics = evaluate_model(trained_model, X_test, y_test, output_dir=str(tmp_path))
    assert metrics["confusion_matrix"].shape == (5, 5)


def test_predictions_length_matches_test_set(tmp_path, trained_model, split_scaled):
    """Number of predictions should equal number of test samples."""
    _, X_test, _, y_test = split_scaled
    metrics = evaluate_model(trained_model, X_test, y_test, output_dir=str(tmp_path))
    assert len(metrics["y_pred"]) == len(y_test)
