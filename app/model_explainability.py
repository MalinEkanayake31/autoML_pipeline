import os
import shap
import matplotlib.pyplot as plt
import numpy as np


def explain_with_shap(model, X, output_dir, model_type="auto", max_display=20):
    """
    Compute and save SHAP global feature importance plot.
    For multi-output models, call this function for each sub-model/target separately.
    Optionally, save per-instance explanations for the first few samples.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Auto-detect explainer type if not specified
    if model_type == "auto":
        if hasattr(model, "predict_proba") or hasattr(model, "feature_importances_"):
            model_type = "tree"
        else:
            model_type = "linear"
    # Use TreeExplainer for tree-based models, LinearExplainer for linear
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    # Global summary plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"))
    plt.close()
    # Optionally, save per-instance force plot for first sample
    try:
        shap.initjs()
        force_plot = shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0], matplotlib=True, show=False)
        plt.savefig(os.path.join(output_dir, "shap_force_first.png"))
        plt.close()
    except Exception:
        pass
    # Save SHAP values as .npy (optional)
    np.save(os.path.join(output_dir, "shap_values.npy"), shap_values.values)
    return shap_values

def plot_feature_importance(model, feature_names, output_dir, max_display=20):
    """
    Plot and save feature importances for tree-based models.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:max_display]
        plt.figure(figsize=(8, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importance.png"))
        plt.close()
        return importances
    return None
