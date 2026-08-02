"""
preprocessing.py
-----------------
Reusable data loading + validation functions.

This is the same logic we already proved works in
notebooks/01_data_understanding.ipynb — moved here so `train.py`,
`predict.py`, and future scripts can import and reuse it instead of
copy-pasting notebook cells. Notebooks stay exploratory; this file is
the "real" version.
"""

from pathlib import Path
import pandas as pd

# Path to the raw data, built from this file's own location so it works
# regardless of which folder someone runs a script from.
THIS_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = THIS_DIR.parent / "data" / "raw" / "Customer-Churn-Records.csv"

TARGET_COLUMN = "Exited"
ID_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
LEAKAGE_COLUMNS = ["Complain"]  # decision made & documented in notebook 03


def load_raw_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV."""
    return pd.read_csv(filepath)


def validate_data(df: pd.DataFrame) -> dict:
    """Run the same integrity checks from Step 2. Returns a report dict
    instead of just printing, so calling code (or a test) can act on it,
    e.g. raise an error if something unexpected shows up."""
    report = {
        "missing_values": df.isnull().sum().to_dict(),
        "full_row_duplicates": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(df["CustomerId"].duplicated().sum()),
        "target_distribution": df[TARGET_COLUMN].value_counts(normalize=True).to_dict(),
    }
    return report


def drop_non_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove identifier columns and the documented leakage column.

    Note: `Complain` is dropped here deliberately, not because it isn't
    predictive (it's the single strongest signal in the data) but because
    it's ~99.9% aligned with the target and unlikely to be known at the
    time a real prediction is needed. See notebooks/03_feature_engineering.ipynb
    for the full reasoning.
    """
    columns_to_drop = [c for c in ID_COLUMNS + LEAKAGE_COLUMNS if c in df.columns]
    return df.drop(columns=columns_to_drop)


if __name__ == "__main__":
    # Quick manual check when running this file directly:
    # `python preprocessing.py` from inside src/
    raw_df = load_raw_data()
    print("Loaded shape:", raw_df.shape)

    report = validate_data(raw_df)
    print("\nValidation report:")
    for key, value in report.items():
        print(f"  {key}: {value}")

    cleaned_df = drop_non_feature_columns(raw_df)
    print("\nShape after dropping ID + leakage columns:", cleaned_df.shape)