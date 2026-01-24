import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            path,
            encoding="utf-8",
            on_bad_lines="skip"
        )
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")
