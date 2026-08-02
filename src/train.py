"""
train.py
--------
Reusable model training pipeline.

Takes the engineered data (produced by features.py, saved at
data/processed/featured_data.csv) and produces a trained, tuned model
saved to disk. Same logic as notebooks/04_model_training.ipynb,
reorganized into functions so the whole pipeline can be run as one
script (`python train.py`) instead of clicking through notebook cells.
"""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from preprocessing import TARGET_COLUMN

THIS_DIR = Path(__file__).resolve().parent
PROCESSED_DATA_PATH = THIS_DIR.parent / "data" / "processed" / "featured_data.csv"
MODELS_DIR = THIS_DIR.parent / "models"

RANDOM_STATE = 42


def load_training_data(filepath: Path = PROCESSED_DATA_PATH):
    """Load the engineered dataset and split into features (X) and target (y)."""
    df = pd.read_csv(filepath)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def split_data(X, y, test_size=0.2):
    """Stratified train/test split -- keeps the ~20% churn rate consistent
    in both sets (see notebook 04 for why this matters with imbalanced data)."""
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


def scale_features(X_train, X_test):
    """Fit the scaler on train only, then apply to both -- fitting on test
    data too would leak information about the test set into training."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def build_candidate_models(y_train):
    """Same 3-4 candidates from Step 5, with class_weight/scale_pos_weight
    set to account for the ~80/20 class imbalance."""
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE
        ),
    }
    if XGBOOST_AVAILABLE:
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, scale_pos_weight=neg / pos,
            random_state=RANDOM_STATE, eval_metric="logloss",
        )
    return models


def select_best_candidate(models, X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test):
    """Trains every candidate and ranks by ROC-AUC on the held-out test set.
    Returns the winning model's name plus the comparison table.

    Note: full metric reporting (precision/recall/confusion matrix/
    classification report) belongs in evaluate.py, not here -- train.py's
    job is picking a model, not writing the evaluation report."""
    scores = []
    for name, model in models.items():
        uses_scaled = name == "Logistic Regression"
        X_tr = X_train_scaled if uses_scaled else X_train
        X_te = X_test_scaled if uses_scaled else X_test

        model.fit(X_tr, y_train)
        y_proba = model.predict_proba(X_te)[:, 1]
        scores.append({"Model": name, "ROC-AUC": roc_auc_score(y_test, y_proba)})

    scores_df = pd.DataFrame(scores).sort_values("ROC-AUC", ascending=False)
    best_name = scores_df.iloc[0]["Model"]
    return best_name, scores_df


def tune_random_forest(X_train, y_train, param_grid=None, cv=3):
    """Hyperparameter tuning, matching Step 5's example. Wired up for
    Random Forest specifically, since it's been the strongest candidate
    so far -- if your best candidate turns out different, add an
    equivalent tuning function for that model type rather than skipping
    tuning altogether."""
    param_grid = param_grid or {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 5],
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def save_artifacts(model, scaler, models_dir: Path = MODELS_DIR):
    """Serialize the final model + scaler so predict.py can load them
    later without retraining."""
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "churn_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")


def run_training_pipeline():
    """Runs the full pipeline end to end: load -> split -> scale ->
    compare candidates -> tune the winner -> save. This is the one
    function a CI job or Makefile target would call."""
    X, y = load_training_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    candidates = build_candidate_models(y_train)
    best_name, scores_df = select_best_candidate(
        candidates, X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test
    )
    print("Model comparison (ROC-AUC):")
    print(scores_df.to_string(index=False))
    print(f"\nBest candidate: {best_name}")

    tuned_model, best_params, best_cv_score = tune_random_forest(X_train, y_train)
    print(f"\nTuned Random Forest best params: {best_params}")
    print(f"Best cross-validated ROC-AUC: {best_cv_score:.3f}")

    save_artifacts(tuned_model, scaler)
    print(f"\nSaved tuned model + scaler to {MODELS_DIR}")

    return tuned_model, scaler


if __name__ == "__main__":
    run_training_pipeline()