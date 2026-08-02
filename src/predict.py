"""
predict.py
----------
Reusable prediction pipeline for scoring a single new customer.

Takes a new customer's raw details (the same fields collected on a bank
application/CRM form), runs them through the exact same feature
engineering used in training, aligns the resulting columns to match
what the model actually saw during training, and returns a churn
probability. This is what app/streamlit_app.py will call.
"""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression

from features import engineer_features
from train import MODELS_DIR, PROCESSED_DATA_PATH
from preprocessing import TARGET_COLUMN

THIS_DIR = Path(__file__).resolve().parent


def load_model_and_scaler(models_dir: Path = MODELS_DIR):
    """Load the artifacts train.py saved -- no retraining needed."""
    model = joblib.load(models_dir / "churn_model.pkl")
    scaler = joblib.load(models_dir / "scaler.pkl")
    return model, scaler


def get_training_columns(processed_path: Path = PROCESSED_DATA_PATH) -> list:
    """The exact set of columns the model was trained on, in order.
    `nrows=0` reads only the header -- no need to load 10,000 rows just
    to get column names. Every new customer gets reindexed to match
    this list before prediction (see align_columns below)."""
    columns = pd.read_csv(processed_path, nrows=0).columns.tolist()
    return [c for c in columns if c != TARGET_COLUMN]


def align_columns(df: pd.DataFrame, training_columns: list) -> pd.DataFrame:
    """Fixes the get_dummies gotcha flagged back in features.py: a
    single new customer only generates dummy columns for the categories
    THEY happen to have. e.g. one customer from France produces no
    'Geography_Germany' column at all -- there's nothing to one-hot
    encode into it from a single row.

    reindex() adds back any training column missing from this customer,
    filled with 0 (correctly meaning "this customer is NOT Germany"),
    and drops any unexpected extra columns -- guaranteeing the model
    always receives the exact same shape and column order it was
    trained on."""
    return df.reindex(columns=training_columns, fill_value=0)


def build_customer_dataframe(customer: dict) -> pd.DataFrame:
    """Wraps one customer's raw field values into a one-row DataFrame,
    since engineer_features() expects a DataFrame, not a plain dict."""
    return pd.DataFrame([customer])


def predict_churn(customer: dict) -> dict:
    """End-to-end: a raw customer dict -> a churn probability.

    Expected keys in `customer` (same fields as the raw CSV, minus the
    ID columns, Complain, and Exited):
        CreditScore, Geography, Gender, Age, Tenure, Balance,
        NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary,
        "Satisfaction Score", "Card Type", "Point Earned"
    """
    model, scaler = load_model_and_scaler()
    training_columns = get_training_columns()

    raw_df = build_customer_dataframe(customer)
    featured_df = engineer_features(raw_df)
    aligned_df = align_columns(featured_df, training_columns)

    # Same rule as evaluate.py: only Logistic Regression needs scaled input.
    if isinstance(model, LogisticRegression):
        model_input = scaler.transform(aligned_df)
    else:
        model_input = aligned_df

    churn_probability = model.predict_proba(model_input)[0, 1]
    churn_prediction = int(model.predict(model_input)[0])

    return {
        "churn_probability": round(float(churn_probability), 4),
        "churn_prediction": churn_prediction,  # 1 = likely to churn, 0 = likely to stay
    }


if __name__ == "__main__":
    # Manual smoke test: run one sample customer through the whole pipeline.
    sample_customer = {
        "CreditScore": 619,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 2,
        "Balance": 0.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 101348.88,
        "Satisfaction Score": 2,
        "Card Type": "DIAMOND",
        "Point Earned": 464,
    }

    result = predict_churn(sample_customer)
    print("Sample customer:", sample_customer)
    print("\nPrediction:", result)