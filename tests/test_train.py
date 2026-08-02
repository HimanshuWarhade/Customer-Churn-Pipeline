"""
test_train.py
--------------
Tests for the training pipeline in src/train.py.

We deliberately don't re-run the full GridSearchCV hyperparameter
search here -- that would make the test suite slow every time it runs.
Instead we test the individual pieces: does the data load correctly,
does the split stay stratified, does scaling work as expected, do the
candidate models get built, and does model selection return a sane
result on a fast subset of the data.

Run with: pytest tests/test_train.py -v
(from the project root, with your venv active)
"""

import pytest

from train import (
    load_training_data,
    split_data,
    scale_features,
    build_candidate_models,
    select_best_candidate,
)


@pytest.fixture(scope="module")
def data():
    """Load once and reuse across every test in this file -- reloading
    10,000 rows from disk for each individual test would be wasteful."""
    return load_training_data()


def test_load_training_data_returns_expected_shape(data):
    X, y = data
    assert X.shape[0] == 10000
    assert X.shape[0] == y.shape[0]
    assert set(y.unique()) == {0, 1}


def test_split_is_stratified(data):
    """stratify=y in split_data() is specifically what should keep the
    churn rate consistent between train and test -- this test would
    fail if someone "simplified" split_data() and dropped stratify."""
    X, y = data
    X_train, X_test, y_train, y_test = split_data(X, y)

    train_churn_rate = y_train.mean()
    test_churn_rate = y_test.mean()
    assert abs(train_churn_rate - test_churn_rate) < 0.02


def test_split_sizes_match_test_size(data):
    X, y = data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    assert len(X_test) == pytest.approx(len(X) * 0.2, abs=1)
    assert len(X_train) + len(X_test) == len(X)


def test_scale_features_produces_standardized_train_data(data):
    """After StandardScaler, training data should have ~0 mean and ~1
    standard deviation per column -- that's the literal definition of
    what it does. A bug here (e.g. fitting the scaler on the wrong
    data) would show up as these numbers being clearly off."""
    X, y = data
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    assert X_train_scaled.mean() == pytest.approx(0, abs=1e-6)
    assert X_train_scaled.std() == pytest.approx(1, abs=1e-1)


def test_build_candidate_models_includes_expected_models(data):
    X, y = data
    models = build_candidate_models(y)

    assert "Logistic Regression" in models
    assert "Decision Tree" in models
    assert "Random Forest" in models
    # XGBoost is intentionally optional (see try/except in train.py) --
    # we don't assert its presence, since a passing test shouldn't
    # depend on what happens to be pip installed.


def test_select_best_candidate_returns_valid_model_name(data):
    """Uses a smaller subset purely to keep this test fast -- we're
    checking the selection logic works correctly, not re-benchmarking
    model performance (that's what the notebooks are for)."""
    X, y = data
    X_small, y_small = X.iloc[:1000], y.iloc[:1000]
    X_train, X_test, y_train, y_test = split_data(X_small, y_small)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    models = build_candidate_models(y_train)
    best_name, scores_df = select_best_candidate(
        models, X_train, X_train_scaled, X_test, X_test_scaled, y_train, y_test
    )

    assert best_name in models
    assert len(scores_df) == len(models)
    assert scores_df["ROC-AUC"].between(0, 1).all()
