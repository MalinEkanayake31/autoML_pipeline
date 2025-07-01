import pandas as pd
import re

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        re.sub(r'\W+', '_', col.strip().lower())
        for col in df.columns
    ]
    return df

def handle_missing_values(df: pd.DataFrame, strategy="drop") -> pd.DataFrame:
    if strategy == "drop":
        return df.dropna()
    elif strategy == "mean":
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == "mode":
        return df.fillna(df.mode().iloc[0])
    else:
        raise ValueError("Unknown missing value strategy")

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

def clean_data(df: pd.DataFrame, missing_strategy="drop") -> pd.DataFrame:
    df = normalize_column_names(df)
    df = handle_missing_values(df, strategy=missing_strategy)
    df = remove_duplicates(df)
    return df

def save_cleaned_data(df: pd.DataFrame, path: str = "outputs/cleaned_data.csv") -> None:
    df.to_csv(path, index=False)
