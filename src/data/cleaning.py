import pandas as pd

TARGET_COL = "Churn"

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Industry-grade data cleaning:
    - Fix data types
    - Handle missing values
    - Standardize target variable
    """

    df = df.copy()

    # -----------------------------
    # Fix TotalCharges (known issue)
    # -----------------------------
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    # -----------------------------
    # Drop rows with missing target
    # -----------------------------
    if TARGET_COL not in df.columns:
        raise ValueError("Target column 'Churn' not found")

    df = df.dropna(subset=[TARGET_COL])

    # -----------------------------
    # Standardize target labels
    # -----------------------------
    df[TARGET_COL] = (
        df[TARGET_COL]
        .astype(str)
        .str.strip()
        .map({"Yes": 1, "No": 0})
    )

    # Final safety check
    if df[TARGET_COL].isna().any():
        raise ValueError("NaN values remain in target after cleaning")

    # -----------------------------
    # Handle remaining NaNs
    # -----------------------------
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    categorical_cols = df.select_dtypes(include=["object"]).columns
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    return df
