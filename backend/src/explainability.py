"""
NeuroPredictAI - Explainable AI (XAI) & Interpretability Engine
Extracts global feature importances and computes patient-level local attribution using SHAP or linear coefficients.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import shap

def get_global_feature_importance(model_bundle: dict) -> Dict[str, Any]:
    """Computes global feature importance from trained bundle."""
    model = model_bundle["model"]
    feature_names = model_bundle.get("feature_names_out", [])
    
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # For logistic regression, take mean absolute weight across classes
        coef = np.abs(model.coef_)
        importances = np.mean(coef, axis=0) if coef.ndim > 1 else coef
    else:
        # Fallback uniform
        importances = np.ones(len(feature_names))
        
    total = np.sum(importances)
    norm_importances = (importances / total * 100.0) if total > 0 else np.zeros_like(importances)
    
    items = []
    for name, weight in zip(feature_names, norm_importances):
        items.append({
            "feature": name,
            "importance": round(float(weight), 2)
        })
    items = sorted(items, key=lambda x: x["importance"], reverse=True)
    
    return {
        "model_name": model_bundle.get("model_name"),
        "features": items
    }

def explain_prediction(model_bundle: dict, patient_df_transformed: np.ndarray, 
                       feature_names: List[str]) -> Dict[str, Any]:
    """
    Computes local feature contribution for a specific prediction using SHAP or linear attributions.
    """
    model = model_bundle["model"]
    model_type = model_bundle.get("model_type", "")
    
    contributions = []
    
    try:
        # Use TreeExplainer for Tree models (RF, XGBoost)
        if "forest" in model_type or "xgb" in model_type:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(patient_df_transformed)
            
            # If multiclass list, take first class or absolute max
            if isinstance(shap_values, list):
                vals = np.mean([np.abs(sv[0]) for sv in shap_values], axis=0)
            elif shap_values.ndim == 3: # (samples, features, classes)
                vals = np.mean(np.abs(shap_values[0]), axis=-1)
            else:
                vals = shap_values[0]
                
            for name, val in zip(feature_names, vals):
                contributions.append({
                    "feature": name,
                    "contribution": round(float(val), 4),
                    "impact": "Increases Risk" if val > 0 else "Decreases Risk"
                })
        elif hasattr(model, "coef_"):
            # Linear Explainer fallback via coefficients * standardized value
            coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            vals = patient_df_transformed[0] * coef
            for name, val in zip(feature_names, vals):
                contributions.append({
                    "feature": name,
                    "contribution": round(float(val), 4),
                    "impact": "Increases Risk" if val > 0 else "Decreases Risk"
                })
        else:
            raise ValueError("Model does not support direct feature attribution.")
    except Exception as e:
        # Graceful fallback: return top global feature weights as explanation proxy
        global_imp = get_global_feature_importance(model_bundle)
        for item in global_imp["features"][:10]:
            contributions.append({
                "feature": item["feature"],
                "contribution": round(float(item["importance"]) / 100.0, 4),
                "impact": "Key Predictive Factor"
            })
            
    # Sort by absolute magnitude of contribution
    contributions = sorted(contributions, key=lambda x: abs(x["contribution"]), reverse=True)
    
    return {
        "method": "SHAP / Feature Attribution",
        "contributions": contributions[:10],
        "disclaimer": "Feature contributions are mathematical model attributions and do not imply medical causality."
    }
