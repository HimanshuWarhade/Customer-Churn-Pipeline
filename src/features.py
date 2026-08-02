"""
features.py
-----------
Reusable feature engineering functions.

Same logic proved out in notebooks/03_feature_engineering.ipynb, moved
here so train.py and predict.py can both build features the exact same
way. This matters most for predict.py later: a brand-new customer has to
go through identical transformations as the training data did, or the
model will receive differently-shaped input and either error out or
silently produce wrong predictions.
"""

from pathlib import Path
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROCESSED_DATA_PATH = THIS_DIR.parent / "data" / "processed" / "featured_data.csv"

CATEGORICAL_COLUMNS = ["Geography", "Gender", "Card Type", "AgeGroup"]


def add_zero_balance_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flags the customer segment EDA found sitting at exactly 0 balance."""
    df = df.copy()
    df["HasZeroBalance"] = (df["Balance"] == 0).astype(int)
    return df


def add_balance_salary_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Balance relative to income, not in isolation. +1 avoids a
    divide-by-zero error (salaries in this data are never actually 0)."""
    df = df.copy()
    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    return df


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Buckets age into groups, since churn rose clearly with age in EDA."""
    df = df.copy()
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[18, 30, 45, 60, 100],
        labels=["18-30", "31-45", "46-60", "60+"],
    )
    return df


def encode_categoricals(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """One-hot encodes nominal (unordered) categorical columns.
    drop_first=True avoids redundant columns (see notebook 03 for why)."""
    columns = columns or CATEGORICAL_COLUMNS
    columns = [c for c in columns if c in df.columns]
    return pd.get_dummies(df, columns=columns, drop_first=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Runs every feature engineering step, in the same order as
    notebooks/03_feature_engineering.ipynb. This is the one function
    both train.py and predict.py should call -- never duplicate these
    steps by hand elsewhere, or training and prediction can silently
    drift apart."""
    df = add_zero_balance_flag(df)
    df = add_balance_salary_ratio(df)
    df = add_age_group(df)
    df = encode_categoricals(df)
    return df


def save_processed_data(df: pd.DataFrame, filepath: Path = PROCESSED_DATA_PATH) -> None:
    """Saves the engineered dataset so train.py can just load a CSV
    instead of re-running feature engineering every time."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


if __name__ == "__main__":
    # Quick manual check: `python features.py` from inside src/
    from preprocessing import load_raw_data, drop_non_feature_columns

    raw_df = load_raw_data()
    cleaned_df = drop_non_feature_columns(raw_df)
    featured_df = engineer_features(cleaned_df)

    print("Shape after feature engineering:", featured_df.shape)
    print("\nColumns:", list(featured_df.columns))

    save_processed_data(featured_df)
    print(f"\nSaved to {PROCESSED_DATA_PATH}")