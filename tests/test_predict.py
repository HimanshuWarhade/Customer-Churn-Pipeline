"""
test_predict.py
----------------
Tests for the single-customer prediction pipeline in src/predict.py.

These load the model/scaler already saved by train.py, so run
`python train.py` (from inside src/) at least once before running this
test file.

Run with: pytest tests/test_predict.py -v
"""

import pandas as pd
import pytest

from predict import predict_churn, align_columns, get_training_columns


SAMPLE_CUSTOMER = {
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


def test_predict_churn_returns_expected_keys():
    result = predict_churn(SAMPLE_CUSTOMER)
    assert "churn_probability" in result
    assert "churn_prediction" in result


def test_predict_churn_probability_is_a_valid_probability():
    result = predict_churn(SAMPLE_CUSTOMER)
    assert 0.0 <= result["churn_probability"] <= 1.0


def test_predict_churn_prediction_is_binary():
    result = predict_churn(SAMPLE_CUSTOMER)
    assert result["churn_prediction"] in (0, 1)


def test_predict_churn_handles_every_geography():
    """Regression test for the get_dummies gotcha from features.py: each
    of these used to risk a missing-column crash before align_columns()
    existed. If someone removes that fix later, this test should fail."""
    for country in ["France", "Germany", "Spain"]:
        customer = {**SAMPLE_CUSTOMER, "Geography": country}
        result = predict_churn(customer)
        assert 0.0 <= result["churn_probability"] <= 1.0


def test_align_columns_fills_missing_columns_with_zero():
    """Directly tests the fix, independent of the model: simulate what a
    single one-hot-encoded customer row looks like when it's missing
    dummy columns the model was trained on."""
    training_columns = ["Age", "Geography_Germany", "Geography_Spain"]
    incomplete_df = pd.DataFrame([{"Age": 42}])  # missing both Geography columns

    aligned = align_columns(incomplete_df, training_columns)

    assert list(aligned.columns) == training_columns
    assert aligned["Geography_Germany"].iloc[0] == 0
    assert aligned["Geography_Spain"].iloc[0] == 0


def test_get_training_columns_excludes_target():
    columns = get_training_columns()
    assert "Exited" not in columns
    assert "CreditScore" in columns
