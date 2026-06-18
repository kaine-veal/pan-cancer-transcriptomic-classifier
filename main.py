"""
Entry point for the Pan-Cancer Transcriptomic Subtyping pipeline.

Orchestrates the full training pipeline in sequence:
    Phase 1 - Data loading and validation
    Phase 2 - Preprocessing (scaling and PCA)
    Phase 3 - Model training, comparison and tuning
    Phase 4 - Evaluation on held-back test set
    Phase 5 - Explainability and biomarker extraction
"""


import joblib
import logging
from pathlib import Path
from src.tcga_classifier.data_loader import load_data, summarise_data
from src.tcga_classifier.preprocessor import split_data, scale_features, apply_pca
from src.tcga_classifier.trainer import (
    train_baseline_models,
    compare_models,
    tune_model,
    save_model
)
from src.tcga_classifier.evaluator import evaluate_model, log_misclassifications
from src.tcga_classifier.explainer import train_explainer_model, extract_top_genes

logger = logging.getLogger(__name__)

# Logging setup
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler()
    ]
)

def main():

    logger.info("=" * 60)
    logger.info("Pan-Cancer Transcriptomic Subtyping Pipeline")
    logger.info("=" * 60)

    # Phase 1 - Data Loading
    logger.info("Phase 1: Data Loading")
    X, y = load_data("data/data.csv", "data/labels.csv")
    summarise_data(X, y)

    # Phase 2 - Preprocessing 
    logger.info("Phase 2: Preprocessing")
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    X_train_pca, X_test_pca, pca = apply_pca(
        X_train_scaled, X_test_scaled, n_components=160
    )

    # Phase 3 - Training 
    logger.info("Phase 3: Model Training")
    models = train_baseline_models(X_train_pca, y_train)
    best_model_name = compare_models(models, X_train_pca, y_train)
    tuned_model = tune_model(best_model_name, X_train_pca, y_train)
    save_model(tuned_model)

    # Phase 4 - Evaluation 
    logger.info("Phase 4: Evaluation")
    metrics = evaluate_model(tuned_model, X_test_pca, y_test)
    log_misclassifications(y_test, metrics["y_pred"], X_test.index)

    # Phase 5 - Explainability 
    logger.info("Phase 5: Explainability")
    gene_names = list(X_train.columns)
    rf_explainer = train_explainer_model(X_train_scaled, y_train, gene_names)
    top_genes = extract_top_genes(rf_explainer, gene_names, top_n=20)

    # Save all pipeline objects for Flask 
    logger.info("Saving pipeline objects for Flask deployment")
    Path("models").mkdir(exist_ok=True)
    joblib.dump(scaler,       "models/scaler.pkl")
    joblib.dump(pca,          "models/pca.pkl")
    joblib.dump(rf_explainer, "models/explainer.pkl")
    logger.info("scaler.pkl saved")
    logger.info("pca.pkl saved")
    logger.info("explainer.pkl saved")

    # Save test set for Flask testing 
    X_test.to_csv("data/test_samples.csv")
    y_test.to_frame().to_csv("data/test_labels.csv")
    logger.info("Test samples saved to data/test_samples.csv")
    logger.info("Test labels saved to data/test_labels.csv")

    logger.info("=" * 60)
    logger.info("Pipeline complete")
    logger.info(f"Macro F1 on test set: {metrics['macro_f1']:.4f}")
    logger.info(f"Micro F1 on test set: {metrics['micro_f1']:.4f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
    