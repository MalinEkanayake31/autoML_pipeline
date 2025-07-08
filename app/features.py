import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Literal, Optional
import warnings

warnings.filterwarnings('ignore')


def safe_encode_categoricals(df: pd.DataFrame, encoding: Literal["label", "onehot"] = "onehot",
                             max_cardinality: int = 50) -> pd.DataFrame:
    """
    Safely encode categorical variables with cardinality limits
    """
    df_copy = df.copy()
    cat_cols = df_copy.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    print(f"Found {len(cat_cols)} categorical columns: {cat_cols}")

    if not cat_cols:
        return df_copy

    for col in cat_cols:
        # Handle NaN values first
        if df_copy[col].isna().any():
            df_copy[col] = df_copy[col].fillna("MISSING")

        # Convert to string to handle mixed types
        df_copy[col] = df_copy[col].astype(str)

        # Check cardinality
        cardinality = df_copy[col].nunique()
        print(f"  {col}: {cardinality} unique values")

        if cardinality > max_cardinality:
            print(f"  Warning: {col} has high cardinality ({cardinality}), using label encoding")
            # Use label encoding for high cardinality
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col])
        else:
            if encoding == "label":
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col])
            elif encoding == "onehot":
                # Use pandas get_dummies for better handling
                dummies = pd.get_dummies(df_copy[col], prefix=col, dummy_na=False)
                df_copy = pd.concat([df_copy, dummies], axis=1)
                df_copy = df_copy.drop(columns=[col])

    return df_copy


def safe_extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely extract datetime features with better detection
    """
    df_copy = df.copy()
    datetime_cols = []

    # Check for datetime columns
    for col in df_copy.columns:
        if df_copy[col].dtype == 'datetime64[ns]':
            datetime_cols.append(col)
        elif df_copy[col].dtype == 'object':
            # Try to convert object columns to datetime
            try:
                test_series = pd.to_datetime(df_copy[col].dropna().head(100), errors='coerce')
                if test_series.notna().sum() > 50:  # If more than 50% are valid dates
                    df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                    datetime_cols.append(col)
            except:
                continue

    print(f"Found {len(datetime_cols)} datetime columns: {datetime_cols}")

    for col in datetime_cols:
        try:
            # Extract features
            df_copy[f"{col}_year"] = df_copy[col].dt.year
            df_copy[f"{col}_month"] = df_copy[col].dt.month
            df_copy[f"{col}_day"] = df_copy[col].dt.day
            df_copy[f"{col}_dayofweek"] = df_copy[col].dt.dayofweek
            df_copy[f"{col}_quarter"] = df_copy[col].dt.quarter

            # Handle missing values in extracted features
            for new_col in [f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dayofweek", f"{col}_quarter"]:
                if df_copy[new_col].isna().any():
                    df_copy[new_col] = df_copy[new_col].fillna(df_copy[new_col].median())

            # Drop original column
            df_copy = df_copy.drop(columns=[col])
            print(f"  Extracted features from {col}")
        except Exception as e:
            print(f"  Warning: Could not extract features from {col}: {e}")

    return df_copy


def safe_scale_numericals(df: pd.DataFrame, scaler_type: Literal["standard", "minmax"] = "standard",
                          exclude_columns: Optional[list] = None) -> Tuple[pd.DataFrame, object]:
    """
    Safely scale numerical features with proper handling of edge cases
    """
    df_copy = df.copy()
    exclude_columns = exclude_columns or []

    # Get numerical columns
    num_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [col for col in num_cols if col not in exclude_columns]

    if not num_cols:
        print("No numerical columns to scale")
        return df_copy, None

    print(f"Scaling {len(num_cols)} numerical columns")

    # Handle infinite values
    for col in num_cols:
        if np.isinf(df_copy[col]).any():
            print(f"  Warning: Infinite values found in {col}, replacing with NaN")
            df_copy[col] = df_copy[col].replace([np.inf, -np.inf], np.nan)

    # Impute missing values first
    imputer = SimpleImputer(strategy='median')
    df_copy[num_cols] = imputer.fit_transform(df_copy[num_cols])

    # Remove constant columns before scaling
    constant_cols = []
    for col in num_cols:
        if df_copy[col].nunique() <= 1:
            constant_cols.append(col)

    if constant_cols:
        print(f"  Removing {len(constant_cols)} constant columns: {constant_cols}")
        df_copy = df_copy.drop(columns=constant_cols)
        num_cols = [col for col in num_cols if col not in constant_cols]

    if not num_cols:
        print("No numerical columns left after removing constants")
        return df_copy, None

    # Scale remaining columns
    try:
        scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        df_copy[num_cols] = scaler.fit_transform(df_copy[num_cols])
        print(f"  Successfully scaled {len(num_cols)} columns")
        return df_copy, scaler
    except Exception as e:
        print(f"  Warning: Scaling failed: {e}")
        return df_copy, None


def safe_remove_high_correlation(df: pd.DataFrame, threshold: float = 0.95,
                                 exclude_columns: Optional[list] = None) -> pd.DataFrame:
    """
    Safely remove highly correlated features with better handling
    """
    exclude_columns = exclude_columns or []

    # Get only numerical columns for correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_columns]

    if len(numeric_cols) <= 1:
        print("Not enough numerical columns for correlation analysis")
        return df

    try:
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr().abs()

        # Find highly correlated pairs
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find columns to drop
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

        if to_drop:
            print(f"Removing {len(to_drop)} highly correlated columns: {to_drop}")
            df = df.drop(columns=to_drop)
        else:
            print("No highly correlated columns found")

        return df
    except Exception as e:
        print(f"Warning: Correlation analysis failed: {e}")
        return df


def robust_feature_engineering_pipeline(
        df: pd.DataFrame,
        target_column: list[str],
        encoding: str = "onehot",
        scale: str = "standard",
        drop_corr: bool = True,
        correlation_threshold: float = 0.95,
        max_cardinality: int = 50
) -> pd.DataFrame:
    """
    Robust feature engineering pipeline that handles any dataset
    """
    print(f"Starting robust feature engineering with shape: {df.shape}")

    # Ensure all target columns exist
    for col in target_column:
        if col not in df.columns:
            raise ValueError(f"Target column '{col}' not found in dataset")

    # Separate features and targets
    X = df.drop(columns=target_column).copy()
    y = df[target_column].copy()

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Step 1: Extract datetime features
    print("\n1. Extracting datetime features...")
    X = safe_extract_datetime_features(X)
    print(f"After datetime extraction: {X.shape}")

    # Step 2: Encode categorical variables
    print("\n2. Encoding categorical variables...")
    X = safe_encode_categoricals(X, encoding=encoding, max_cardinality=max_cardinality)
    print(f"After categorical encoding: {X.shape}")

    # Step 3: Scale numerical features
    print("\n3. Scaling numerical features...")
    if scale != "none":
        X, scaler = safe_scale_numericals(X, scaler_type=scale)
        print(f"After scaling: {X.shape}")

    # Step 4: Remove highly correlated features
    if drop_corr:
        print("\n4. Removing highly correlated features...")
        X = safe_remove_high_correlation(X, threshold=correlation_threshold)
        print(f"After correlation removal: {X.shape}")

    # Step 5: Handle any remaining NaN values
    print("\n5. Final cleanup...")
    if X.isna().any().any():
        print("Warning: NaN values found, filling with 0")
        X = X.fillna(0)

    # Step 6: Ensure no infinite values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(X[col]).any():
            print(f"Warning: Infinite values in {col}, replacing with column median")
            X[col] = X[col].replace([np.inf, -np.inf], X[col].median())

    # Combine with targets
    result_df = X.copy()
    for col in y.columns:
        result_df[col] = y[col]

    print(f"\nFinal shape: {result_df.shape}")
    print(f"Final columns: {result_df.columns.tolist()}")

    return result_df