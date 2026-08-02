"""
evaluate.py
-----------
Reusable model evaluation functions.

Loads the model + scaler saved by train.py, recreates the exact same
test split (same random_state, so it's the identical held-out data
train.py never trained on), and produces the full evaluation report:
accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and a text
classification report. This is what turns into reports/Model_Report,
and what a CI pipeline could re-run after every training job to catch
a regression before it ships.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from train import load_training_data, split_data, MODELS_DIR

THIS_DIR = Path(__file__).resolve().parent
REPORTS_DIR = THIS_DIR.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def load_model_and_scaler(models_dir: Path = MODELS_DIR):
    """Load the artifacts train.py saved -- no retraining needed."""
    model = joblib.load(models_dir / "churn_model.pkl")
    scaler = joblib.load(models_dir / "scaler.pkl")
    return model, scaler


def get_test_predictions(model, scaler, X_test):
    """Scale the input only if the model actually needs it. Tree-based
    models (Decision Tree/Random Forest/XGBoost) use raw features,
    matching exactly how train.py trained them."""
    if isinstance(model, LogisticRegression):
        X_input = scaler.transform(X_test)
    else:
        X_input = X_test
    y_pred = model.predict(X_input)
    y_proba = model.predict_proba(X_input)[:, 1]
    return y_pred, y_proba


def compute_metrics(y_test, y_pred, y_proba) -> dict:
    """Same five metrics from Step 5. Accuracy alone would be misleading
    here since churn is imbalanced (~20% of customers)."""
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }


def plot_confusion_matrix(y_test, y_pred, save_path: Path = None):
    """Draws and optionally saves the confusion matrix as a PNG."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
    plt.close()


def save_text_report(metrics: dict, classification_text: str, save_path: Path):
    """Writes a plain-text report -- the same numbers printed to the
    console, kept as a file so it can be attached to a PR or shared
    without anyone having to re-run the code."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write("Model Evaluation Report\n")
        f.write("=======================\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("\nClassification Report\n")
        f.write("----------------------\n")
        f.write(classification_text)


def run_evaluation():
    """Full evaluation pipeline: load model -> recreate test split ->
    predict -> compute metrics -> save confusion matrix + text report."""
    model, scaler = load_model_and_scaler()

    X, y = load_training_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    y_pred, y_proba = get_test_predictions(model, scaler, X_test)
    metrics = compute_metrics(y_test, y_pred, y_proba)

    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    report_text = classification_report(y_test, y_pred, target_names=["Stayed", "Churned"])
    print("\nClassification report:")
    print(report_text)

    plot_confusion_matrix(y_test, y_pred, FIGURES_DIR / "confusion_matrix_final.png")
    save_text_report(metrics, report_text, REPORTS_DIR / "Model_Report.txt")

    print(f"Saved confusion matrix to {FIGURES_DIR / 'confusion_matrix_final.png'}")
    print(f"Saved text report to {REPORTS_DIR / 'Model_Report.txt'}")

    return metrics


if __name__ == "__main__":
    run_evaluation()