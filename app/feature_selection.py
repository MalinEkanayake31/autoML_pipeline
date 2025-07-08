import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, f_classif, f_regression
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import List, Literal, Optional
import warnings

warnings.filterwarnings('ignore')


def safe_variance_filter(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """
    Safely apply variance filtering with fallback
    """
    try:
        # Only apply to numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            print("No numerical columns for variance filtering")
            return df

        # Check if we have any variance at all
        variances = df[numeric_cols].var()
        if variances.max() == 0:
            print("Warning: All numerical columns have zero variance")
            return df

        # Apply variance threshold
        selector = VarianceThreshold(threshold=threshold)

        # Fit only on numerical columns
        numeric_data = df[numeric_cols]
        selected_numeric = selector.fit_transform(numeric_data)

        # Get selected column names
        selected_cols = np.array(numeric_cols)[selector.get_support()].tolist()

        if not selected_cols:
            print(f"Warning: No columns passed variance threshold {threshold}, keeping all")
            return df

        # Create result dataframe
        result_df = pd.DataFrame(selected_numeric, columns=selected_cols, index=df.index)

        # Add back non-numerical columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        for col in non_numeric_cols:
            result_df[col] = df[col]

        print(f"Variance filtering: {len(numeric_cols)} -> {len(selected_cols)} columns")
        return result_df

    except Exception as e:
        print(f"Warning: Variance filtering failed: {e}")
        return df


def safe_rfe_selection(df: pd.DataFrame, target: pd.Series, num_features: int = 10,
                       task_type: str = "classification") -> pd.DataFrame:
    """
    Safely apply RFE with fallback to simpler methods
    """
    try:
        # Only use numerical columns for RFE
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            print("No numerical columns for RFE")
            return df

        # Limit features to available columns
        num_features = min(num_features, len(numeric_cols))

        # Check target for issues
        if target.isna().any():
            print("Warning: Target has NaN values, dropping corresponding rows")
            valid_mask = ~target.isna()
            df_clean = df[valid_mask]
            target_clean = target[valid_mask]
        else:
            df_clean = df
            target_clean = target

        # Use simpler estimator for RFE
        if task_type == "classification":
            # Check if target is binary or multiclass
            n_classes = len(target_clean.unique())
            if n_classes == 2:
                estimator = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
            else:
                estimator = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
        else:
            estimator = LinearRegression()

        # Apply RFE only to numerical columns
        numeric_data = df_clean[numeric_cols]

        selector = RFE(estimator, n_features_to_select=num_features, step=1)
        selected_data = selector.fit_transform(numeric_data, target_clean)

        # Get selected columns
        selected_cols = np.array(numeric_cols)[selector.support_].tolist()

        # Create result dataframe
        result_df = pd.DataFrame(selected_data, columns=selected_cols, index=df_clean.index)

        # Add back non-numerical columns
        non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()
        for col in non_numeric_cols:
            result_df[col] = df_clean[col]

        print(f"RFE selection: {len(numeric_cols)} -> {len(selected_cols)} columns")
        return result_df

    except Exception as e:
        print(f"Warning: RFE selection failed: {e}")
        return safe_univariate_selection(df, target, num_features, task_type)


def safe_univariate_selection(df: pd.DataFrame, target: pd.Series, top_k: int = 10,
                              task_type: str = "classification") -> pd.DataFrame:
    """
    Safely apply univariate feature selection (fallback method)
    """
    try:
        # Only use numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            print("No numerical columns for univariate selection")
            return df

        # Limit features to available columns
        top_k = min(top_k, len(numeric_cols))

        # Handle missing values in target
        if target.isna().any():
            print("Warning: Target has NaN values, dropping corresponding rows")
            valid_mask = ~target.isna()
            df_clean = df[valid_mask]
            target_clean = target[valid_mask]
        else:
            df_clean = df
            target_clean = target

        # Choose scoring function
        if task_type == "classification":
            # Check if target is numeric for classification
            if target_clean.dtype in ['int64', 'float64']:
                # Convert to categorical if it looks like classes
                if len(target_clean.unique()) < 20:
                    target_clean = target_clean.astype('category')
            score_func = f_classif
        else:
            score_func = f_regression

        # Apply univariate selection
        numeric_data = df_clean[numeric_cols]
        selector = SelectKBest(score_func=score_func, k=top_k)
        selected_data = selector.fit_transform(numeric_data, target_clean)

        # Get selected columns
        selected_cols = np.array(numeric_cols)[selector.get_support()].tolist()

        # Create result dataframe
        result_df = pd.DataFrame(selected_data, columns=selected_cols, index=df_clean.index)

        # Add back non-numerical columns
        non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()
        for col in non_numeric_cols:
            result_df[col] = df_clean[col]

        print(f"Univariate selection: {len(numeric_cols)} -> {len(selected_cols)} columns")
        return result_df

    except Exception as e:
        print(f"Warning: Univariate selection failed: {e}")
        print("Returning original dataframe")
        return df


def safe_mutual_info_selection(df: pd.DataFrame, target: pd.Series, top_k: int = 10,
                               task_type: str = "classification") -> pd.DataFrame:
    """
    Safely apply mutual information selection with fallback
    """
    try:
        # Only use numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            print("No numerical columns for mutual info selection")
            return df

        # Limit features to available columns
        top_k = min(top_k, len(numeric_cols))

        # Handle missing values in target
        if target.isna().any():
            print("Warning: Target has NaN values, dropping corresponding rows")
            valid_mask = ~target.isna()
            df_clean = df[valid_mask]
            target_clean = target[valid_mask]
        else:
            df_clean = df
            target_clean = target

        # Apply mutual information
        numeric_data = df_clean[numeric_cols]

        if task_type == "classification":
            mi_scores = mutual_info_classif(numeric_data, target_clean, random_state=42)
        else:
            mi_scores = mutual_info_regression(numeric_data, target_clean, random_state=42)

        # Get top k features
        mi_series = pd.Series(mi_scores, index=numeric_cols)
        selected_cols = mi_series.nlargest(top_k).index.tolist()

        # Create result dataframe
        result_df = df_clean[selected_cols].copy()

        # Add back non-numerical columns
        non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()
        for col in non_numeric_cols:
            result_df[col] = df_clean[col]

        print(f"Mutual info selection: {len(numeric_cols)} -> {len(selected_cols)} columns")
        return result_df

    except Exception as e:
        print(f"Warning: Mutual info selection failed: {e}")
        return safe_univariate_selection(df, target, top_k, task_type)


def robust_feature_selection_pipeline(
        df: pd.DataFrame,
        target_column: list[str],
        method: Literal["variance", "rfe", "mutual_info", "univariate", "auto"] = "auto",
        k: int = 10,
        task_type: str = "classification",
        variance_threshold: float = 0.01
) -> pd.DataFrame:
    """
    Robust feature selection pipeline with multiple fallback methods
    """
    print(f"Starting robust feature selection with shape: {df.shape}")

    # Validate inputs
    for col in target_column:
        if col not in df.columns:
            raise ValueError(f"Target column '{col}' not found in dataset")

    # Separate features and targets
    X = df.drop(columns=target_column).copy()
    y = df[target_column].copy()

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Auto-select method based on data characteristics
    if method == "auto":
        n_features = X.select_dtypes(include=[np.number]).shape[1]
        n_samples = len(X)

        if n_features < 5:
            method = "variance"
        elif n_samples < 1000:
            method = "mutual_info"
        elif n_features > 1000:
            method = "variance"
        else:
            method = "rfe"

        print(f"Auto-selected method: {method}")

    # Ensure k is reasonable
    max_features = X.select_dtypes(include=[np.number]).shape[1]
    if max_features == 0:
        print("Warning: No numerical features found, returning original dataframe")
        return df

    k = min(k, max_features)
    k = max(k, 1)  # At least 1 feature

    print(f"Selecting {k} features using {method} method")

    # Apply selected method with fallbacks (use first target for selection)
    main_target = y.columns[0]
    try:
        if method == "variance":
            X_selected = safe_variance_filter(X, threshold=variance_threshold)
        elif method == "rfe":
            X_selected = safe_rfe_selection(X, y[main_target], num_features=k, task_type=task_type)
        elif method == "mutual_info":
            X_selected = safe_mutual_info_selection(X, y[main_target], top_k=k, task_type=task_type)
        elif method == "univariate":
            X_selected = safe_univariate_selection(X, y[main_target], top_k=k, task_type=task_type)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Ensure we have some features
        if X_selected.shape[1] == 0:
            print("Warning: No features selected, using univariate selection as fallback")
            X_selected = safe_univariate_selection(X, y[main_target], top_k=k, task_type=task_type)

        # Combine with targets
        result_df = X_selected.copy()
        for col in y.columns:
            result_df[col] = y[col]

        # Handle target alignment
        if len(result_df) != len(y):
            # Align indices
            result_df = result_df.reindex(y.index)

        print(f"Final shape: {result_df.shape}")
        print(f"Selected features: {X_selected.columns.tolist()}")

        return result_df

    except Exception as e:
        print(f"Error in feature selection: {e}")
        print("Returning original dataframe")
        return df