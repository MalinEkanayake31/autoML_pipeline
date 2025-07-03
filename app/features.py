import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from typing import Tuple, Literal


def encode_categoricals(df: pd.DataFrame, encoding: Literal["label", "onehot"] = "onehot") -> pd.DataFrame:
    df_copy = df.copy()
    cat_cols = df_copy.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if encoding == "label":
        for col in cat_cols:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col].astype(str))
    elif encoding == "onehot":
        df_copy = pd.get_dummies(df_copy, columns=cat_cols, drop_first=True)
    return df_copy


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    datetime_cols = df_copy.select_dtypes(include=["datetime64", "datetime"]).columns.tolist()

    for col in datetime_cols:
        df_copy[col + "_year"] = df_copy[col].dt.year
        df_copy[col + "_month"] = df_copy[col].dt.month
        df_copy[col + "_day"] = df_copy[col].dt.day
        df_copy = df_copy.drop(columns=[col])
    return df_copy


def scale_numericals(
        df: pd.DataFrame, scaler_type: Literal["standard", "minmax"] = "standard"
) -> Tuple[pd.DataFrame, object]:
    df_copy = df.copy()
    num_cols = df_copy.select_dtypes(include=["number"]).columns.tolist()

    if len(num_cols) > 0:
        scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        df_copy[num_cols] = scaler.fit_transform(df_copy[num_cols])
        return df_copy, scaler
    else:
        return df_copy, None


def remove_high_correlation(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    # Only work with numerical columns
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return df

    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find columns to drop
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop from original dataframe
    return df.drop(columns=to_drop)


def feature_engineering_pipeline(
        df: pd.DataFrame,
        encoding: str = "onehot",
        scale: str = "standard",
        drop_corr: bool = False
) -> pd.DataFrame:
    print(f"Starting feature engineering with shape: {df.shape}")

    # Extract datetime features
    df = extract_datetime_features(df)
    print(f"After datetime extraction: {df.shape}")

    # Encode categorical variables
    df = encode_categoricals(df, encoding=encoding)
    print(f"After categorical encoding: {df.shape}")

    # Scale numerical features
    df, scaler = scale_numericals(df, scaler_type=scale)
    print(f"After scaling: {df.shape}")

    # Remove highly correlated features
    if drop_corr:
        df = remove_high_correlation(df)
        print(f"After correlation removal: {df.shape}")

    return df