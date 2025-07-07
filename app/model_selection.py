import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
                             mean_squared_error, r2_score, mean_absolute_error)
from sklearn.preprocessing import LabelEncoder
from typing import Literal, Dict, Any
import warnings

warnings.filterwarnings('ignore')


def safe_encode_target(y: pd.Series, task_type: str) -> tuple[pd.Series, Any]:
    """
    Safely encode target variable if needed
    """
    encoder = None

    if task_type == "classification":
        # Check if target is already numeric
        if y.dtype == 'object' or y.dtype.name == 'category':
            encoder = LabelEncoder()
            y_encoded = pd.Series(encoder.fit_transform(y), index=y.index)
            print(f"Target encoded: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")
            return y_encoded, encoder

    return y, encoder


def evaluate_classification(model, X_test, y_test, y_test_original=None) -> Dict:
    """
    Comprehensive classification evaluation with error handling
    """
    try:
        y_pred = model.predict(X_test)

        # Basic metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

        # ROC AUC (only for models with predict_proba)
        if hasattr(model, "predict_proba"):
            try:
                y_pred_proba = model.predict_proba(X_test)
                n_classes = len(np.unique(y_test))

                if n_classes == 2:
                    metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except Exception as e:
                print(f"Warning: Could not compute ROC AUC: {e}")
                metrics["roc_auc"] = None
        else:
            metrics["roc_auc"] = None

        return metrics
    except Exception as e:
        print(f"Error in classification evaluation: {e}")
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0, "roc_auc": None}


def evaluate_regression(model, X_test, y_test) -> Dict:
    """
    Comprehensive regression evaluation with error handling
    """
    try:
        y_pred = model.predict(X_test)

        # Handle any infinite or NaN predictions
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            print("Warning: Model produced NaN or infinite predictions")
            y_pred = np.nan_to_num(y_pred, nan=y_test.mean(), posinf=y_test.max(), neginf=y_test.min())

        return {
            "rmse": mean_squared_error(y_test, y_pred, squared=False),
            "r2": r2_score(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred)
        }
    except Exception as e:
        print(f"Error in regression evaluation: {e}")
        return {"rmse": float('inf'), "r2": -float('inf'), "mse": float('inf'), "mae": float('inf')}


def get_models_for_task(task_type: str, n_samples: int, n_features: int) -> Dict:
    """
    Get appropriate models based on task type and data characteristics
    """
    models = {}

    if task_type == "classification":
        models["LogisticRegression"] = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='liblinear' if n_samples < 10000 else 'lbfgs'
        )

        models["RandomForestClassifier"] = RandomForestClassifier(
            n_estimators=100 if n_samples > 1000 else 50,
            max_depth=10 if n_features > 50 else None,
            random_state=42,
            n_jobs=-1
        )

        models["DecisionTreeClassifier"] = DecisionTreeClassifier(
            max_depth=10 if n_features > 20 else None,
            random_state=42
        )

        # Add more models for larger datasets
        if n_samples > 100:
            models["GradientBoostingClassifier"] = GradientBoostingClassifier(
                n_estimators=100 if n_samples > 1000 else 50,
                max_depth=3,
                random_state=42
            )

            models["GaussianNB"] = GaussianNB()

            # Add SVM for smaller datasets
            if n_samples < 10000 and n_features < 1000:
                models["SVC"] = SVC(probability=True, random_state=42)

            # Add KNN for smaller datasets
            if n_samples < 5000:
                models["KNeighborsClassifier"] = KNeighborsClassifier(
                    n_neighbors=min(5, n_samples // 10)
                )

    else:  # regression
        models["Ridge"] = Ridge(random_state=42)
        models["Lasso"] = Lasso(random_state=42, max_iter=1000)

        models["RandomForestRegressor"] = RandomForestRegressor(
            n_estimators=100 if n_samples > 1000 else 50,
            max_depth=10 if n_features > 50 else None,
            random_state=42,
            n_jobs=-1
        )

        models["DecisionTreeRegressor"] = DecisionTreeRegressor(
            max_depth=10 if n_features > 20 else None,
            random_state=42
        )

        # Add more models for larger datasets
        if n_samples > 100:
            models["GradientBoostingRegressor"] = GradientBoostingRegressor(
                n_estimators=100 if n_samples > 1000 else 50,
                max_depth=3,
                random_state=42
            )

            # Add SVR for smaller datasets
            if n_samples < 10000 and n_features < 1000:
                models["SVR"] = SVR()

            # Add KNN for smaller datasets
            if n_samples < 5000:
                models["KNeighborsRegressor"] = KNeighborsRegressor(
                    n_neighbors=min(5, n_samples // 10)
                )

    return models


def safe_train_model(model, X_train, y_train, X_test, y_test, model_name: str, task_type: str) -> tuple[Dict, bool]:
    """
    Safely train a model with error handling
    """
    try:
        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        if task_type == "classification":
            metrics = evaluate_classification(model, X_test, y_test)
            score = metrics["f1_score"]
        else:
            metrics = evaluate_regression(model, X_test, y_test)
            score = metrics["r2"]

        print(f"✅ {model_name} - Score: {score:.4f}")
        return metrics, True

    except Exception as e:
        print(f"❌ {model_name} failed: {str(e)}")
        return {}, False


def model_selection_pipeline(
        df: pd.DataFrame,
        target_column: str,
        task_type: Literal["classification", "regression"],
        test_size: float = 0.2,
        random_state: int = 42
) -> Dict:
    """
    Robust model selection pipeline that works with any dataset
    """
    # Ensure models directory exists
    Path("models").mkdir(exist_ok=True)

    # Validate that target column exists
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in DataFrame. Available columns: {df.columns.tolist()}")

    print(f"Starting model selection for {task_type} task...")
    print(f"Dataset shape: {df.shape}")

    # Handle any remaining NaN values
    if df.isna().any().any():
        print("Warning: NaN values found in dataset, handling them...")
        # Fill NaN in target with mode/median
        if df[target_column].isna().any():
            if task_type == "classification":
                df[target_column] = df[target_column].fillna(df[target_column].mode().iloc[0])
            else:
                df[target_column] = df[target_column].fillna(df[target_column].median())

        # Fill NaN in features
        for col in df.columns:
            if col != target_column and df[col].isna().any():
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "MISSING")

    # Split features and target
    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target type: {y.dtype}")

    # Handle infinite values in features
    for col in X.select_dtypes(include=[np.number]).columns:
        if np.isinf(X[col]).any():
            print(f"Warning: Infinite values in {col}, replacing with median")
            X[col] = X[col].replace([np.inf, -np.inf], X[col].median())

    # Encode target if necessary
    y_encoded, target_encoder = safe_encode_target(y, task_type)
    y = y_encoded

    # Check for minimum samples per class in classification
    if task_type == "classification":
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        print(f"Class distribution: {class_counts.to_dict()}")

        if min_class_count < 2:
            print("Warning: Some classes have very few samples")
            # Remove classes with only 1 sample
            valid_classes = class_counts[class_counts >= 2].index
            mask = y.isin(valid_classes)
            X = X[mask]
            y = y[mask]
            print(f"Filtered to {len(valid_classes)} classes with sufficient samples")

    # Split into train and test
    try:
        if task_type == "classification" and len(y.unique()) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
    except Exception as e:
        print(f"Warning: Stratified split failed: {e}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Get appropriate models
    models = get_models_for_task(task_type, len(X_train), X_train.shape[1])
    print(f"Training {len(models)} models: {list(models.keys())}")

    # Initialize best model tracking
    best_score = float("-inf") if task_type == "classification" else float("-inf")
    best_model = None
    best_metrics = {}
    best_model_name = ""
    successful_models = 0

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")

        metrics, success = safe_train_model(model, X_train, y_train, X_test, y_test, name, task_type)

        if success:
            successful_models += 1

            # Determine if this is the best model
            if task_type == "classification":
                score = metrics["f1_score"]
                is_better = score > best_score
            else:
                score = metrics["r2"]
                is_better = score > best_score

            # Always set best_model_name to the first successful model if none is set
            if best_model_name == "":
                best_score = score
                best_model = model
                best_metrics = metrics
                best_model_name = name
                print(f"🏆 New best model: {name} (Score: {score:.4f})")
            elif is_better:
                best_score = score
                best_model = model
                best_metrics = metrics
                best_model_name = name
                print(f"🏆 New best model: {name} (Score: {score:.4f})")

    if successful_models == 0:
        raise ValueError("No models could be trained successfully")

    print(f"\n✅ Successfully trained {successful_models}/{len(models)} models")
    print(f"🏆 Best model: {best_model_name} (Score: {best_score:.4f})")

    # Save the best model
    try:
        model_data = {
            'model': best_model,
            'target_encoder': target_encoder,
            'task_type': task_type,
            'target_column': target_column,
            'feature_columns': X.columns.tolist(),
            'model_name': best_model_name
        }

        joblib.dump(model_data, "models/best_model.pkl")
        print("📁 Best model saved successfully")
    except Exception as e:
        print(f"Warning: Could not save model: {e}")

    return {
        "model": best_model_name,
        "metrics": best_metrics,
        "best_score": best_score,
        "successful_models": successful_models,
        "total_models": len(models)
    }