"""
trainer.py

Trains and compares two classification models on PCA-reduced data:
1. Random Forest — ensemble of 100 decision trees
2. MLP           — neural network with two hidden layers (100, 50)

Both models are evaluated at baseline using 5-fold cross-validation.
The better model is tuned via GridSearchCV and saved to disk for
use by the Flask deployment.
"""

import logging
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)


def train_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> dict:
    """
    Train both models with default parameters for initial comparison.

    Random Forest builds 100 independent decision trees in parallel
    (n_jobs=-1), each trained on a random subset of patients and features,
    with the final prediction determined by majority vote.

    MLP uses a funnel architecture — 160 PCA inputs → 100 neurons →
    50 neurons → 5 output classes. Each output neuron represents one
    tumour type, with softmax converting raw scores to probabilities.
    max_iter=500 caps training iterations to prevent infinite loops.

    random_state fixes both the Random Forest subsampling and MLP weight
    initialisation for reproducibility.

    Parameters
    ----------
    X_train : np.ndarray
        PCA-transformed training features (640, 160)
    y_train : np.ndarray
        Training labels
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    models : dict
        Fitted model objects keyed by name
    """
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=random_state
        )
    }

    for name, model in models.items():
        logger.info(f"Training baseline {name}...")
        model.fit(X_train, y_train)
        logger.info(f"{name} baseline training complete")

    return models


def compare_models(
    models: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5
) -> str:
    """
    Compare models using stratified k-fold cross-validation.

    The training set (640 patients) is split into 5 folds of ~128 patients.
    Each fold takes a turn as the validation set while the remaining four
    train the model. This produces 5 F1 scores per model, averaged for
    the final result. Every patient is evaluated exactly once.

    Macro F1 is used rather than accuracy — it weights all five tumour
    types equally regardless of class size. A model predicting only BRCA
    would score 37% accuracy but near-zero macro F1, correctly exposing it.

    Cross-validation operates entirely within X_train. The held-back test
    set (161 patients) is not involved at any point.

    Results on this dataset:
        RandomForest — Mean Macro F1: 0.9633 (+/- 0.0169)
        MLP          — Mean Macro F1: 0.9724 (+/- 0.0114)

    MLP wins on both mean score and consistency (lower std across folds).

    Parameters
    ----------
    models : dict
        Fitted baseline models from train_baseline_models
    X_train : np.ndarray
        PCA-transformed training features
    y_train : np.ndarray
        Training labels
    cv_folds : int
        Number of cross-validation folds (default 5)

    Returns
    -------
    best_model_name : str
        Name of the winning model, passed to tune_model
    """
    logger.info(f"Comparing models using {cv_folds}-fold cross-validation")

    cv      = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        scores     = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1
        )
        mean_score = scores.mean()
        std_score  = scores.std()
        results[name] = mean_score

        logger.info(f"{name} - Mean Macro F1: {mean_score:.4f} (+/- {std_score:.4f})")

    best_model_name = max(results, key=results.get)
    logger.info(f"Best model: {best_model_name} with Macro F1: {results[best_model_name]:.4f}")

    return best_model_name


def tune_model(
    best_model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42
):
    """
    Tune the winning model using GridSearchCV.

    GridSearchCV exhaustively evaluates every hyperparameter combination
    in the grid using k-fold cross-validation, returning the combination
    with the highest mean macro F1.

    For MLP: 3 x 3 x 2 = 18 combinations x 5 folds = 90 total fits.
    The Random Forest grid is defined but unused when MLP wins — the
    if/else selects only the relevant model and grid.

    MLP hyperparameters tuned:
    - hidden_layer_sizes: network architecture options
    - alpha:              regularisation strength (prevents overfitting)
    - learning_rate:      whether step size is fixed or adaptive

    Best parameters found:
        hidden_layer_sizes: (100, 50)
        alpha:              0.0001
        learning_rate:      constant
        Best CV Macro F1:   0.9724

    The default parameters were already optimal — GridSearchCV confirmed
    rather than improved performance.

    Parameters
    ----------
    best_model_name : str
        Name of model to tune, from compare_models
    X_train : np.ndarray
        PCA-transformed training features
    y_train : np.ndarray
        Training labels
    cv_folds : int
        Number of cross-validation folds
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    best_estimator : fitted sklearn estimator
        Tuned model trained on the full training set
    """
    param_grids = {
        "RandomForest": {
            "n_estimators":     [100, 200, 300],
            "max_depth":        [None, 10, 20],
            "min_samples_split": [2, 5],
        },
        "MLP": {
            "hidden_layer_sizes": [(100,), (100, 50), (200, 100)],
            "alpha":              [0.0001, 0.001, 0.01],
            "learning_rate":      ["constant", "adaptive"],
        }
    }

    if best_model_name == "RandomForest":
        model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    else:
        model = MLPClassifier(max_iter=500, random_state=random_state)

    param_grid = param_grids[best_model_name]
    cv         = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    logger.info(f"Starting GridSearchCV for {best_model_name}")
    logger.info(f"Parameter grid: {param_grid}")

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation Macro F1: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def save_model(model, model_path: str = "models/model.pkl") -> None:
    """
    Save the tuned model to disk using joblib.

    joblib is preferred over pickle for sklearn models as it handles
    large numpy arrays more efficiently. The saved file contains all
    learned weights and parameters — no retraining needed at inference.

    Parameters
    ----------
    model : fitted sklearn estimator
    model_path : str
        Destination path for the saved model file
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to: {model_path}")


def load_model(model_path: str = "models/model.pkl"):
    """
    Load a saved model from disk.

    Used in app.py at startup — models are loaded once when Flask
    initialises rather than on every prediction request.

    Parameters
    ----------
    model_path : str
        Path to the saved model file

    Returns
    -------
    model : fitted sklearn estimator

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at the given path
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"No model found at: {model_path}")

    model = joblib.load(model_path)
    logger.info(f"Model loaded from: {model_path}")
    return model