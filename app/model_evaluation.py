import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_classification(y_true, y_pred, y_proba=None, output_dir=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
    elif y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    else:
        metrics["roc_auc"] = None

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    if output_dir:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()

    return metrics


def evaluate_regression(y_true, y_pred, output_dir=None):
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),  # Compute RMSE manually for compatibility
        "r2": r2_score(y_true, y_pred),
    }
    if output_dir:
        plt.figure(figsize=(6, 5))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "regression_scatter.png"))
        plt.close()

        # Residuals plot
        residuals = y_true - y_pred
        plt.figure(figsize=(6, 5))
        sns.histplot(residuals, bins=30, kde=True)
        plt.title("Residuals Distribution")
        plt.xlabel("Residuals")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "residuals.png"))
        plt.close()

    return metrics


def evaluate_model_on_test(test_df, target_column, model_path, task_type, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    model = joblib.load(model_path)
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    if task_type == "classification":
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        metrics = evaluate_classification(y_test, y_pred, y_proba, output_dir)
    else:
        y_pred = model.predict(X_test)
        metrics = evaluate_regression(y_test, y_pred, output_dir)

    # Save metrics
    metrics_path = os.path.join(output_dir, "test_metrics.json")
    pd.Series(metrics).to_json(metrics_path)
    print(f"Test metrics saved to {metrics_path}")
    return metrics
