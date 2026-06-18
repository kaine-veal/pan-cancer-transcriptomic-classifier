"""
evaluator.py

Evaluates the tuned model on the held-back 20% test set.

This is the first and only time the test set is used — it represents
genuinely unseen data, giving an honest estimate of real-world performance.

Produces:
- Classification report (precision, recall, F1 per class)
- Macro and micro F1 scores
- Confusion matrix saved as a two-panel heatmap (raw counts + normalised)
- Warning-level log entries for every misclassified sample
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score

logger = logging.getLogger(__name__)

CLASS_NAMES = ["BRCA", "COAD", "KIRC", "LUAD", "PRAD"]


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test,
    output_dir: str = "outputs"
) -> dict:
    """
    Generate predictions on the test set and compute evaluation metrics.

    y_pred gives the hard class prediction per patient.
    y_prob gives the full probability distribution across all five classes,
    used in the Flask app to display confidence and flag uncertain predictions.

    Macro F1 is the primary metric — it weights all five tumour types equally
    regardless of class size, which matters given BRCA dominates at 37%.

    Results on this dataset:
        Macro F1:       0.9577
        Micro F1:       0.9627
        Accuracy:       96.3%
        Misclassified:  6/161 — all involving LUAD

    Parameters
    ----------
    model : fitted sklearn estimator
        Tuned MLP from trainer.py
    X_test : np.ndarray
        PCA-transformed test features (161, 160)
    y_test : pd.Series
        True tumour type labels for the test set
    output_dir : str
        Directory to save the confusion matrix plot

    Returns
    -------
    metrics : dict
        y_pred, y_prob, macro_f1, micro_f1, confusion_matrix,
        classification_report
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Evaluating model on held-back test set")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    logger.info(f"Classification Report:\n{report}")

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    micro_f1 = f1_score(y_test, y_pred, average="micro")
    logger.info(f"Macro F1 Score: {macro_f1:.4f}")
    logger.info(f"Micro F1 Score: {micro_f1:.4f}")

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_NAMES)
    logger.info(f"Confusion Matrix:\n{cm}")

    _plot_confusion_matrix(cm, output_dir)

    return {
        "y_pred":                y_pred,
        "y_prob":                y_prob,
        "macro_f1":              macro_f1,
        "micro_f1":              micro_f1,
        "confusion_matrix":      cm,
        "classification_report": report
    }


def _plot_confusion_matrix(cm: np.ndarray, output_dir: str) -> None:
    """
    Save the confusion matrix as a two-panel heatmap.

    Left panel shows raw counts — how many patients were correctly or
    incorrectly classified. Right panel shows normalised proportions,
    dividing each row by its total so performance is comparable across
    classes of different sizes (e.g. COAD with 16 test samples vs
    BRCA with 60).

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix from sklearn
    output_dir : str
        Directory to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0]
    )
    axes[0].set_title("Confusion Matrix - Raw Counts")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    cm_normalised = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_normalised,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[1]
    )
    axes[1].set_title("Confusion Matrix - Normalised")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    output_path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to: {output_path}")


def log_misclassifications(
    y_test,
    y_pred: np.ndarray,
    X_test_index
) -> None:
    """
    Log each misclassified sample at WARNING level for clinical review.

    In a clinical context a misclassification is a significant event.
    WARNING level entries stand out in the log and can be filtered
    independently of INFO entries during audit.

    Parameters
    ----------
    y_test : pd.Series
        True labels
    y_pred : np.ndarray
        Predicted labels
    X_test_index : pd.Index
        Sample IDs from the test set
    """
    misclassified = y_test.values != y_pred

    if misclassified.sum() == 0:
        logger.info("No misclassifications on test set")
        return

    logger.info(f"Misclassified samples: {misclassified.sum()}")

    for idx, (true, pred) in enumerate(zip(y_test.values, y_pred)):
        if true != pred:
            sample_id = X_test_index[idx]
            logger.warning(
                f"MISCLASSIFIED: {sample_id} | True: {true} | Predicted: {pred}"
            )