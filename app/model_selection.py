import pandas as pd
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
from typing import Literal, Dict


def evaluate_classification(model, X_test, y_test) -> Dict:
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }

    # Only add ROC AUC if model has predict_proba
    if hasattr(model, "predict_proba"):
        try:
            y_pred_proba = model.predict_proba(X_test)
            # Handle binary vs multiclass
            if len(set(y_test)) == 2:
                metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        except Exception as e:
            print(f"Warning: Could not compute ROC AUC: {e}")
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def evaluate_regression(model, X_test, y_test) -> Dict:
    y_pred = model.predict(X_test)
    return {
        "rmse": mean_squared_error(y_test, y_pred, squared=False),
        "r2": r2_score(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred)
    }


def model_selection_pipeline(
        df: pd.DataFrame,
        target_column: str,
        task_type: Literal["classification", "regression"]
) -> Dict:
    # Ensure models directory exists
    Path("models").mkdir(exist_ok=True)

    # Validate that target column exists
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in DataFrame. Available columns: {df.columns.tolist()}")

    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    print(f"DEBUG: X shape: {X.shape}")
    print(f"DEBUG: y shape: {y.shape}")
    print(f"DEBUG: X columns: {X.columns.tolist()}")

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                        stratify=y if task_type == "classification" else None)

    # Define models based on task type
    if task_type == "classification":
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForestClassifier": RandomForestClassifier(random_state=42)
        }
    else:
        models = {
            "Ridge": Ridge(random_state=42),
            "RandomForestRegressor": RandomForestRegressor(random_state=42)
        }

    # Initialize best model tracking
    best_score = float("-inf") if task_type == "classification" else float("inf")
    best_model = None
    best_metrics = {}
    best_model_name = ""

    # Evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train, y_train)

            if task_type == "classification":
                metrics = evaluate_classification(model, X_test, y_test)
                score = metrics["f1_score"]
                is_better = score > best_score
            else:
                metrics = evaluate_regression(model, X_test, y_test)
                score = metrics["r2"]  # Use R² for regression (higher is better)
                is_better = score > best_score

            print(f"{name} - Score: {score:.4f}")

            if is_better:
                best_score = score
                best_model = model
                best_metrics = metrics
                best_model_name = name

        except Exception as e:
            print(f"Error training {name}: {e}")
            continue

    if best_model is None:
        raise ValueError("No models could be trained successfully")

    # Save the best model
    try:
        joblib.dump(best_model, "models/best_model.pkl")
        print(f"Best model ({best_model_name}) saved successfully")
    except Exception as e:
        print(f"Warning: Could not save model: {e}")

    return {
        "model": best_model_name,
        "metrics": best_metrics,
        "best_score": best_score
    }