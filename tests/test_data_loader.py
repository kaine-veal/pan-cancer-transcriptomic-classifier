"""
test_data_loader.py

Unit tests for data_loader.py- covers file loading,
sample ID validation, and label dimensionality.
"""

import pytest
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, "src")  

from tcga_classifier.data_loader import load_data


# Fixtures 

@pytest.fixture
def valid_files(tmp_path):
    """Write a minimal valid data.csv and labels.csv to a temporary directory."""

    # X: small expression matrix- 10 patients (rows), 5 genes (columns)
    X = pd.DataFrame(
        np.random.rand(10, 5),
        index=[f"sample_{i}" for i in range(10)],   # row labels = sample IDs
        columns=[f"gene_{i}" for i in range(5)]      # column labels = gene names
    )

    # y: one label per patient — index must match X exactly for validation to pass
    y = pd.DataFrame(
        {"Class": ["BRCA", "KIRC", "LUAD", "PRAD", "COAD"] * 2},
        index=X.index  # same sample IDs as X- this is the valid case
    )

    # write both DataFrames to CSV files inside the temporary directory
    data_path   = tmp_path / "data.csv"
    labels_path = tmp_path / "labels.csv"

    X.to_csv(data_path)
    y.to_csv(labels_path)

    # load_data expects strings, not Path objects — convert before returning
    return str(data_path), str(labels_path)


# Tests 

def test_missing_file_raises():
    """Passing a non-existent file path should raise FileNotFoundError."""
    # no files are created here- the paths point to nothing on disk
    with pytest.raises(FileNotFoundError):
        load_data("missing_data.csv", "missing_labels.csv")


def test_mismatched_index_raises(tmp_path):
    """Mismatched sample IDs between data and labels should raise ValueError."""
    # build a small expression matrix with sample IDs as the index
    X = pd.DataFrame(
        np.random.rand(5, 3),
        index=[f"sample_{i}" for i in range(5)],
        columns=[f"gene_{i}" for i in range(3)]
    )

    # labels use a different naming scheme — patient_0 instead of sample_0
    # load_data compares X.index and y.index and should reject this mismatch
    y = pd.DataFrame(
        {"Class": ["BRCA"] * 5},
        index=[f"patient_{i}" for i in range(5)]  # deliberate mismatch
    )

    # write both to disk so load_data can read them as real files
    data_path   = tmp_path / "data.csv"
    labels_path = tmp_path / "labels.csv"

    X.to_csv(data_path)
    y.to_csv(labels_path)

    # load_data should detect the index mismatch and raise ValueError
    with pytest.raises(ValueError):
        load_data(str(data_path), str(labels_path))


def test_y_is_one_dimensional(valid_files):
    """squeeze() should produce a 1D Series for sklearn compatibility."""
    # unpack the two file paths created by the valid_files fixture
    data_path, labels_path = valid_files

    # load_data reads the single-column labels CSV and calls .squeeze()
    # squeeze() converts a one-column DataFrame into a 1D Series
    # sklearn expects labels as a 1D array- a DataFrame would cause errors downstream
    _, y = load_data(data_path, labels_path)

    # ndim == 1 confirms y is a Series (1D), not a DataFrame (2D)
    assert y.ndim == 1
    