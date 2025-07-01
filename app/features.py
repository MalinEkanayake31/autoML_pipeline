import pandas as pd
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
        df_copy.drop(columns=[col], inplace=True)
    return df_copy


def scale_numericals(
        df: pd.DataFrame, scaler_type: Literal["standard", "minmax"] = "standard"
) -> Tuple[pd.DataFrame, object]:
    df_copy = df.copy()
    num_cols = df_copy.select_dtypes(include=["number"]).columns.tolist()
    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
    df_copy[num_cols] = scaler.fit_transform(df_copy[num_cols])
    return df_copy, scaler


def remove_high_correlation(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        pd.np.triu(pd.np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)


def feature_engineering_pipeline(
        df: pd.DataFrame,
        encoding: str = "onehot",
        scale: str = "standard",
        drop_corr: bool = False
) -> pd.DataFrame:
    df = extract_datetime_features(df)
    df = encode_categoricals(df, encoding=encoding)
    df, _ = scale_numericals(df, scaler_type=scale)
    if drop_corr:
        df = remove_high_correlation(df)
    return df
