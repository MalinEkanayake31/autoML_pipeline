import pandas as pd
from sklearn.feature_selection import VarianceThreshold, RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from typing import List, Literal

def variance_filter(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    selector = VarianceThreshold(threshold)
    selected = selector.fit_transform(df)
    kept_features = df.columns[selector.get_support()].tolist()
    return pd.DataFrame(selected, columns=kept_features)

def rfe_selection(df: pd.DataFrame, target: pd.Series, num_features: int = 10) -> pd.DataFrame:
    estimator = LogisticRegression(max_iter=1000)
    selector = RFE(estimator, n_features_to_select=num_features)
    selector = selector.fit(df, target)
    selected_columns = df.columns[selector.support_]
    return df[selected_columns]

def mutual_info_selection(df: pd.DataFrame, target: pd.Series, top_k: int = 10) -> pd.DataFrame:
    mi = mutual_info_classif(df, target)
    selected = pd.Series(mi, index=df.columns).sort_values(ascending=False).head(top_k).index
    return df[selected]

def feature_selection_pipeline(
    df: pd.DataFrame,
    target: pd.Series,
    method: Literal["variance", "rfe", "mutual_info"] = "variance",
    k: int = 10
) -> pd.DataFrame:
    if method == "variance":
        return variance_filter(df)
    elif method == "rfe":
        return rfe_selection(df, target, num_features=k)
    elif method == "mutual_info":
        return mutual_info_selection(df, target, top_k=k)
    else:
        raise ValueError("Unsupported feature selection method")
