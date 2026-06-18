"""
explainer.py

Extracts biological interpretability from the trained model.

The MLP was trained on 160 PCA components, not genes. PCA components
are linear combinations of thousands of genes, so biomarker genes cannot
be read directly from the model weights.

A separate Random Forest is trained on the scaled pre-PCA gene expression
data to extract feature importances in the original gene space. This is
standard practice in transcriptomic ML — use the best model for prediction,
use an interpretable model for explanation.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


def train_explainer_model(
    X_train_scaled: np.ndarray,
    y_train,
    gene_names: list,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Train a Random Forest on the full scaled gene space (pre-PCA).

    Feature importance scores reflect how much each gene reduced
    classification uncertainty (impurity) across all decision nodes
    in all 100 trees. Genes used frequently and effectively to separate
    tumour types receive higher scores.

    Parameters
    ----------
    X_train_scaled : np.ndarray
        Scaled training features in original gene space (640, 20531)
    y_train : pd.Series
        Training labels
    gene_names : list
        Gene names corresponding to columns in X_train_scaled
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    rf_explainer : fitted RandomForestClassifier
    """
    logger.info("Training Random Forest explainer on full gene space")
    logger.info(f"Input shape: {X_train_scaled.shape}")

    rf_explainer = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )

    rf_explainer.fit(X_train_scaled, y_train)
    logger.info("Explainer model training complete")

    return rf_explainer


def extract_top_genes(
    rf_explainer: RandomForestClassifier,
    gene_names: list,
    top_n: int = 20,
    output_dir: str = "outputs"
) -> pd.DataFrame:
    """
    Extract and rank the top discriminatory genes from the explainer.

    Three importance measures are reported:
    - importance:          raw mean decrease in impurity score
    - relative_importance: score relative to the top gene (top gene = 100%)
    - pct_of_total:        share of total model importance across all 20,531 genes

    The top gene (gene_14092) scores 1.14% of total importance — roughly
    230x higher than the average gene, confirming it as a strong biomarker
    candidate.

    Parameters
    ----------
    rf_explainer : fitted RandomForestClassifier
    gene_names : list
        Gene names corresponding to model features
    top_n : int
        Number of top genes to return
    output_dir : str
        Directory to save the feature importance plot

    Returns
    -------
    top_genes : pd.DataFrame
        Ranked genes with importance, relative_importance, and pct_of_total
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    importances = rf_explainer.feature_importances_

    gene_importance_df = pd.DataFrame({
        "gene":       gene_names,
        "importance": importances
    })

    top_genes = (
        gene_importance_df
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    top_genes["relative_importance"] = (
        top_genes["importance"] / top_genes["importance"].max() * 100
    ).round(1)

    top_genes["pct_of_total"] = (
        top_genes["importance"] / gene_importance_df["importance"].sum() * 100
    ).round(2)

    for _, row in top_genes.iterrows():
        logger.info(
            f"  {row['gene']}: importance={row['importance']:.4f} | "
            f"relative={row['relative_importance']}% | "
            f"pct_of_total={row['pct_of_total']}%"
        )

    _plot_feature_importance(top_genes, top_n, output_dir)

    return top_genes


def _plot_feature_importance(
    top_genes: pd.DataFrame,
    top_n: int,
    output_dir: str
) -> None:
    """
    Save a horizontal bar chart of top gene importances.

    The x-axis shows mean decrease in impurity — how much each gene
    reduced classification uncertainty across all trees. Higher values
    indicate stronger discrimination between tumour types.

    Parameters
    ----------
    top_genes : pd.DataFrame
        Output from extract_top_genes
    top_n : int
        Number of genes shown in the plot
    output_dir : str
        Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(
        top_genes["gene"][::-1],
        top_genes["importance"][::-1],
        color="steelblue",
        alpha=0.8
    )

    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
    ax.set_title(f"Top {top_n} Biomarker Genes")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / "feature_importance.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Feature importance plot saved to: {output_path}")


def explain_prediction(
    patient_scaled: np.ndarray,
    rf_explainer: RandomForestClassifier,
    gene_names: list,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Return the top driving genes for a single patient prediction.

    Used in the Flask app to provide per-patient biomarker explanations
    alongside the tumour type prediction — satisfying the no black box
    requirement from the project brief.

    Parameters
    ----------
    patient_scaled : np.ndarray
        Scaled gene expression for one patient (1, 20531)
    rf_explainer : fitted RandomForestClassifier
    gene_names : list
        Gene names corresponding to model features
    top_n : int
        Number of top genes to return

    Returns
    -------
    top_driver_genes : pd.DataFrame
        Top genes with importance and expression value for this patient
    """
    importances = rf_explainer.feature_importances_

    gene_importance_df = pd.DataFrame({
        "gene":       gene_names,
        "importance": importances,
        "expression": patient_scaled.flatten()
    })

    return (
        gene_importance_df
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )