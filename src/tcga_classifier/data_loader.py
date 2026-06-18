"""
data_loader.py

Loads and validates the two TCGA source files:
- data.csv:   gene expression matrix (801 samples x 20,531 genes)
- labels.csv: tumour type labels (801 samples x 1 label)
"""

import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def load_data(data_path: str, labels_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load gene expression data (X) and tumour type labels (y).

    index_col=0 sets sample IDs as the row index rather than a data column.
    Labels are squeezed from a single-column DataFrame into a 1D Series
    for sklearn compatibility.

    Sample ID alignment is validated before returning — mismatched files
    would cause silent label errors downstream.

    Parameters
    ----------
    data_path : str
        Path to gene expression CSV (801 x 20531)
    labels_path : str
        Path to labels CSV (801 x 1)

    Returns
    -------
    X : pd.DataFrame
        Gene expression feature matrix
    y : pd.Series
        Tumour type labels

    Raises
    ------
    FileNotFoundError
        If either file does not exist
    ValueError
        If sample IDs do not match between files
    """
    data_path   = Path(data_path)
    labels_path = Path(labels_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    logger.info(f"Loading gene expression data from: {data_path}")
    X = pd.read_csv(data_path, index_col=0)
    logger.info(f"Gene expression matrix loaded: {X.shape[0]} samples, {X.shape[1]} genes")

    logger.info(f"Loading labels from: {labels_path}")
    labels_df = pd.read_csv(labels_path, index_col=0)
    y = labels_df.squeeze()
    logger.info(f"Labels loaded: {len(y)} samples")

    if not X.index.equals(y.index):
        raise ValueError(
            "Sample IDs in data and labels files do not match. "
            "Check that both files refer to the same cohort."
        )

    logger.info("Sample IDs validated - data and labels are aligned")
    logger.info(f"Class distribution:\n{y.value_counts().to_string()}")

    return X, y


def summarise_data(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Log a summary of the loaded dataset.

    Checks for missing values, zero-heavy genes, and class distribution.
    The class imbalance (BRCA dominates at 37%) is visible immediately
    and motivates the use of macro F1 over accuracy during evaluation.

    Parameters
    ----------
    X : pd.DataFrame
        Gene expression feature matrix
    y : pd.Series
        Tumour type labels
    """
    logger.info("=== Data Summary ===")
    logger.info(f"Total samples: {X.shape[0]}")
    logger.info(f"Total features (genes): {X.shape[1]}")
    logger.info(f"Missing values: {X.isnull().sum().sum()}")
    logger.info(f"Zero values: {(X == 0).sum().sum()}")
    logger.info(f"Classes: {sorted(y.unique())}")
    logger.info(f"Class counts:\n{y.value_counts().to_string()}")