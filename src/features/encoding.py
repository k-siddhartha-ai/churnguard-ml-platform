import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from config.settings import TARGET_COL


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = [
        c for c in df.select_dtypes(include=["object"]).columns
        if c != TARGET_COL
    ]

    if not categorical_cols:
        return df

    encoder = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore"
    )

    encoded = encoder.fit_transform(df[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index
    )

    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)

    if TARGET_COL not in df.columns:
        raise RuntimeError("CRITICAL: Target column lost during encoding")

    return df
