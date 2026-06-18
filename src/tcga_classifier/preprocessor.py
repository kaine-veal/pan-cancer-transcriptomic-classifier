"""
preprocessor.py

Prepares raw gene expression data for modelling in three steps:
1. Train/test splitting
2. Feature scaling (StandardScaler)
3. Dimensionality reduction (PCA)

The scaler and PCA are fitted on training data only and saved alongside
the model so new patient data can be transformed identically at inference.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Split data into 80% training and 20% test sets.

    Stratified splitting ensures class proportions are preserved in both
    sets. Without stratification, COAD (only 78 samples total) could end
    up severely underrepresented in the test set, making evaluation of
    that class unreliable.

    random_state fixes the split for reproducibility — the same 640/161
    patient split is produced on every run.

    Parameters
    ----------
    X : pd.DataFrame
        Gene expression feature matrix
    y : pd.Series
        Tumour type labels
    test_size : float
        Proportion reserved for testing (default 0.2)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
    """
    logger.info(f"Splitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    logger.info(f"Training class distribution:\n{y_train.value_counts().to_string()}")

    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> tuple:
    """
    Scale gene expression features using StandardScaler.

    Gene expression values vary enormously across genes — one gene may
    range 0-10, another 0-10,000. Without scaling, PCA would be dominated
    by high-value genes regardless of biological relevance.

    StandardScaler transforms each gene to mean=0, std=1 across all
    training samples, putting all 20,531 genes on equal footing.

    The scaler is fitted on training data only. Fitting on the full dataset
    would expose test set statistics to the model during training (data leakage).
    The same training statistics are then applied to the test set.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training gene expression matrix
    X_test : pd.DataFrame
        Test gene expression matrix

    Returns
    -------
    X_train_scaled : np.ndarray
    X_test_scaled : np.ndarray
    scaler : fitted StandardScaler
    """
    logger.info("Fitting StandardScaler on training data")

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    logger.info("Feature scaling complete")
    logger.info(f"Training data mean after scaling: {X_train_scaled.mean():.4f} (should be ~0)")
    logger.info(f"Training data std after scaling: {X_train_scaled.std():.4f} (should be ~1)")

    return X_train_scaled, X_test_scaled, scaler


def apply_pca(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    n_components: int = 160,
    random_state: int = 42
) -> tuple:
    """
    Reduce dimensionality from 20,531 genes to 160 principal components.

    With 20,531 features and only 801 samples the dataset is highly
    susceptible to overfitting and the curse of dimensionality — models
    memorise training patients rather than learning generalisable biology.
    PCA addresses this by finding the directions of maximum variation
    across patients and discarding the rest.

    n_components=160 was determined empirically using a scree plot
    (notebooks/eda.ipynb). Values tested:

        100 components → 73.6% variance (too low)
        160 components → 80.0% variance (selected)
        200 components → 83.2% variance
        500 components → 96.9% variance (too high, captures noise)

    80% variance retains the dominant biological signals while discarding
    noise. The scree plot confirmed the cumulative variance curve crosses
    the 80% threshold at exactly 160 components.

    PCA is fitted on training data only — same data leakage rule as the
    scaler. The fitted PCA object is returned for use at inference.

    Parameters
    ----------
    X_train_scaled : np.ndarray
        Scaled training features
    X_test_scaled : np.ndarray
        Scaled test features
    n_components : int
        Number of principal components to retain (default 160)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    X_train_pca : np.ndarray
        PCA-transformed training features (640, 160)
    X_test_pca : np.ndarray
        PCA-transformed test features (161, 160)
    pca : fitted PCA object
    """
    logger.info(f"Fitting PCA with {n_components} components on training data")

    pca          = PCA(n_components=n_components, random_state=random_state)
    X_train_pca  = pca.fit_transform(X_train_scaled)
    X_test_pca   = pca.transform(X_test_scaled)

    variance_explained = pca.explained_variance_ratio_.sum() * 100
    logger.info(f"PCA complete: {n_components} components retain {variance_explained:.1f}% of variance")
    logger.info(f"Training shape after PCA: {X_train_pca.shape}")
    logger.info(f"Test shape after PCA: {X_test_pca.shape}")

    return X_train_pca, X_test_pca, pca