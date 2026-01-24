from config.settings import TARGET_COL


def validate_data(df):
    if df.empty:
        raise ValueError("Dataset is empty")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing")
