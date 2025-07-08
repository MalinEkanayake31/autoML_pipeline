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
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier

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


def evaluate_classification(y_true, y_pred) -> Dict:
    """
    Comprehensive classification evaluation with error handling
    """
    try:
        # Basic metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        # ROC AUC (only for models with predict_proba)
        if hasattr(y_true, "predict_proba"): # Assuming y_true is a classifier model or similar
            try:
                y_pred_proba = y_true.predict_proba(y_pred) # Assuming y_pred is the model's prediction
                n_classes = len(np.unique(y_true))

                if n_classes == 2:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            except Exception as e:
                print(f"Warning: Could not compute ROC AUC: {e}")
                metrics["roc_auc"] = None
        else:
            metrics["roc_auc"] = None

        return metrics
    except Exception as e:
        print(f"Error in classification evaluation: {e}")
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0, "roc_auc": None}


def evaluate_regression(y_true, y_pred) -> Dict:
    """
    Comprehensive regression evaluation with error handling
    """
    try:
        # Handle any infinite or NaN predictions
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            print("Warning: Model produced NaN or infinite predictions")
            y_pred = np.nan_to_num(y_pred, nan=y_true.mean(), posinf=y_true.max(), neginf=y_true.min())

        return {
            "rmse": mean_squared_error(y_true, y_pred, squared=False),
            "r2": r2_score(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred)
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
            metrics = evaluate_classification(y_test, model.predict(X_test))
            score = metrics["f1_score"]
        else:
            metrics = evaluate_regression(y_test, model.predict(X_test))
            score = metrics["r2"]

        print(f"✅ {model_name} - Score: {score:.4f}")
        return metrics, True

    except Exception as e:
        print(f"❌ {model_name} failed: {str(e)}")
        return {}, False


def model_selection_pipeline(
        df: pd.DataFrame,
        target_column: list[str],
        task_type: Literal["classification", "regression"],
        test_size: float = 0.2,
        random_state: int = 42
) -> Dict:
    """
    Robust model selection pipeline that works with any dataset, now supporting multi-target
    """
    Path("models").mkdir(exist_ok=True)
    for col in target_column:
        if col not in df.columns:
            raise ValueError(
                f"Target column '{col}' not found in DataFrame. Available columns: {df.columns.tolist()}")
    print(f"Starting model selection for {task_type} task...")
    print(f"Dataset shape: {df.shape}")
    # Handle any remaining NaN values
    if df.isna().any().any():
        print("Warning: NaN values found in dataset, handling them...")
        for col in target_column:
            if df[col].isna().any():
                if task_type == "classification":
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
                else:
                    df[col] = df[col].fillna(df[col].median())
        for col in df.columns:
            if col not in target_column and df[col].isna().any():
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "MISSING")
    X = df.drop(columns=target_column).copy()
    y = df[target_column].copy()
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target type: {y.dtypes}")
    # Handle infinite values in features
    for col in X.select_dtypes(include=[np.number]).columns:
        if np.isinf(X[col]).any():
            print(f"Warning: Infinite values in {col}, replacing with median")
            X[col] = X[col].replace([np.inf, -np.inf], X[col].median())
    # Encode target if necessary (only for single target classification)
    if task_type == "classification" and len(target_column) == 1:
        y_encoded, target_encoder = safe_encode_target(y[target_column[0]], task_type)
        y[target_column[0]] = y_encoded
    else:
        target_encoder = None
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    models = get_models_for_task(task_type, len(X_train), X_train.shape[1])
    print(f"Training {len(models)} models: {list(models.keys())}")
    best_score = float("-inf")
    best_model = None
    best_metrics = {}
    best_model_name = ""
    successful_models = 0
    for name, model in models.items():
        print(f"\nTraining {name}...")
        # Wrap with multi-output if needed
        if len(target_column) > 1:
            if task_type == "classification":
                model = MultiOutputClassifier(model)
            else:
                model = MultiOutputRegressor(model)
        try:
            model.fit(X_train, y_train)
            if len(target_column) == 1:
                # Single target: use standard metrics
                if task_type == "classification":
                    y_pred = model.predict(X_test)
                    metrics = evaluate_classification(y_test[target_column[0]], y_pred)
                    score = metrics["f1_score"]
                else:
                    y_pred = model.predict(X_test)
                    metrics = evaluate_regression(y_test[target_column[0]], y_pred)
                    score = metrics["r2"]
            else:
                # Multi-target: average metrics across targets
                y_pred = model.predict(X_test)
                metrics = {}
                scores = []
                for i, col in enumerate(target_column):
                    if task_type == "classification":
                        m = evaluate_classification(y_test[col], y_pred[:, i])
                        score = m["f1_score"]
                    else:
                        m = evaluate_regression(y_test[col], y_pred[:, i])
                        score = m["r2"]
                    metrics[col] = m
                    scores.append(score)
                score = np.mean(scores)
            successful_models += 1
            if best_model_name == "" or score > best_score:
                best_score = score
                best_model = model
                best_metrics = metrics
                best_model_name = name
                print(f"🏆 New best model: {name} (Score: {score:.4f})")
        except Exception as e:
            print(f"❌ {name} failed: {str(e)}")
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