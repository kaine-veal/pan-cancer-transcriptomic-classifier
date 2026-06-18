"""
app.py

Flask web application for Pan-Cancer Transcriptomic Subtyping.

Endpoints:
- GET  /         : Home page with CSV upload form
- POST /predict  : Receives CSV, returns tumour type predictions
- GET  /health   : Health check confirming app and models are loaded

Usage:
    python src/tcga_classifier/app.py
"""

import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, render_template, jsonify

# ── Logging ────────────────────────────────────────────────────────────────────

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── App and model loading ──────────────────────────────────────────────────────

# template_folder navigates up two levels from src/tcga_classifier/ to project root
app = Flask(__name__, template_folder="../../templates")

MODEL_DIR = Path("models")

try:
    model        = joblib.load(MODEL_DIR / "model.pkl")
    scaler       = joblib.load(MODEL_DIR / "scaler.pkl")
    pca          = joblib.load(MODEL_DIR / "pca.pkl")
    rf_explainer = joblib.load(MODEL_DIR / "explainer.pkl")
    logger.info("All models loaded successfully")
except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}")
    raise

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Serve the upload form."""
    logger.info("Home page requested")
    return render_template("home.html")


@app.route("/health")
def health():
    """Health check — confirms app is running and all models are loaded."""
    return jsonify({
        "status": "healthy",
        "models_loaded": ["model", "scaler", "pca", "rf_explainer"]
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive an uploaded CSV and return tumour type predictions.

    The uploaded file is validated, scaled using the training scaler,
    reduced to 160 PCA components, and passed to the tuned MLP for
    classification. The Random Forest explainer then identifies the top
    10 driving genes per patient.

    All predictions and confidence scores are logged to app.log.
    Predictions below 70% confidence are flagged for clinical review.
    """
    logger.info("Prediction request received")

    # Validate upload
    if "file" not in request.files:
        logger.warning("No file in request")
        return render_template("error.html", error="No file uploaded.")

    file = request.files["file"]

    if file.filename == "":
        return render_template("error.html", error="No file selected.")

    if not file.filename.endswith(".csv"):
        return render_template("error.html", error="File must be a .csv")

    # Load CSV
    try:
        df = pd.read_csv(file, index_col=0)
        logger.info(f"Uploaded file shape: {df.shape}")
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return render_template("error.html", error=f"Could not read CSV: {e}")

    # Validate dimensions
    if df.shape[1] != 20531:
        return render_template(
            "error.html",
            error=(
                f"Expected 20531 gene columns, got {df.shape[1]}. "
                f"Please upload a correctly formatted RNA-Seq profile."
            )
        )

    # Preprocessing — apply training scaler and PCA
    try:
        X_scaled = scaler.transform(df.values)
        X_pca    = pca.transform(X_scaled)
        logger.info(f"Preprocessing complete. Shape after PCA: {X_pca.shape}")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return render_template("error.html", error=f"Preprocessing error: {e}")

    # Predict
    try:
        predictions   = model.predict(X_pca)
        probabilities = model.predict_proba(X_pca)
        class_labels  = model.classes_
        logger.info(f"Predictions: {predictions}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return render_template("error.html", error=f"Prediction error: {e}")

    # Build per-patient results
    patients         = []
    gene_names       = list(df.columns)
    importances      = rf_explainer.feature_importances_
    total_importance = importances.sum()

    for i, sample_id in enumerate(df.index):

        patient_scaled = X_scaled[i].reshape(1, -1)

        prob_dict = {
            label: round(float(prob) * 100, 1)
            for label, prob in zip(class_labels, probabilities[i])
        }

        sorted_probs = sorted(
            [{"label": k, "value": v} for k, v in prob_dict.items()],
            key=lambda x: x["value"],
            reverse=True
        )

        confidence = prob_dict[predictions[i]]

        gene_importance_df = pd.DataFrame({
            "gene":       gene_names,
            "importance": importances,
            "expression": patient_scaled.flatten()
        })

        top_genes_df = (
            gene_importance_df
            .sort_values("importance", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )

        top_genes_df["relative_importance"] = (
            top_genes_df["importance"] / top_genes_df["importance"].max() * 100
        ).round(1)

        top_genes_df["pct_of_total"] = (
            top_genes_df["importance"] / total_importance * 100
        ).round(2)

        top_genes_df["expression"] = top_genes_df["expression"].round(3)

        patients.append({
            "sample_id":      sample_id,
            "prediction":     predictions[i],
            "confidence":     confidence,
            "low_confidence": confidence < 70,
            "probabilities":  sorted_probs,
            "top_genes":      top_genes_df.to_dict("records")
        })

        logger.info(
            f"Sample {sample_id}: predicted {predictions[i]} "
            f"with {confidence}% confidence"
        )

    return render_template("results.html", patients=patients)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    """Start the Flask development server."""
    logger.info("Starting Pan-Cancer Classifier Flask app")
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False
    )


if __name__ == "__main__":
    main()