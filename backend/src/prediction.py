"""
NeuroPredictAI - Inference and Prediction Module
Generates predictions for structured patient records using trained models,
computes confidence scores/probabilities, generates SHAP explanations, and records history.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from .explainability import explain_prediction
from .database import save_prediction, DEFAULT_DB_PATH

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BACKEND_DIR, "models")

def load_trained_model(model_name_or_type: str, models_dir: str = MODELS_DIR) -> Dict[str, Any]:
    """Loads a saved model bundle by name or key."""
    norm_name = model_name_or_type.lower().replace(" ", "_")
    candidates = [
        f"{norm_name}_pipeline.joblib",
        f"{norm_name}.joblib",
        f"random_forest_pipeline.joblib",
        f"xgboost_pipeline.joblib",
        f"logistic_regression_pipeline.joblib"
    ]
    
    # Check exact match first
    for cand in candidates:
        path = os.path.join(models_dir, cand)
        if os.path.exists(path):
            return joblib.load(path)
            
    # Check any file in models_dir
    if os.path.exists(models_dir):
        files = [f for f in os.listdir(models_dir) if f.endswith(".joblib") and "pipeline" in f]
        if files:
            return joblib.load(os.path.join(models_dir, files[0]))
            
    raise FileNotFoundError(f"No trained model artifact found for '{model_name_or_type}'. Please train a model first.")

def predict_patient(patient_data: dict, model_name: str = "Random Forest",
                    patient_ref: Optional[str] = None,
                    models_dir: str = MODELS_DIR,
                    db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Executes inference for a single patient:
    1. Loads trained pipeline
    2. Constructs feature row matching required schema
    3. Transforms features through trained preprocessing pipeline
    4. Predicts class and class probabilities
    5. Computes SHAP explanation
    6. Stores prediction in SQLite database
    7. Returns comprehensive response with disclaimer
    """
    bundle = load_trained_model(model_name, models_dir=models_dir)
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    classes = bundle.get("classes", [])
    int_to_target = bundle.get("int_to_target", {})
    expected_features = bundle.get("features", [])
    
    # Build single-row DataFrame
    row_dict = {}
    for feat in expected_features:
        # Check direct or case-insensitive key
        val = patient_data.get(feat)
        if val is None:
            # Try lower-case match
            for k, v in patient_data.items():
                if k.lower() == feat.lower():
                    val = v
                    break
        row_dict[feat] = [val]
        
    df_single = pd.DataFrame(row_dict)
    
    # Feature engineering and transformation
    from .feature_engineering import engineer_features
    df_engineered = engineer_features(df_single)
    
    # Preprocessor transform
    X_transformed = preprocessor.transform(df_engineered)
    
    # Inference
    is_xgb = "xgb" in bundle.get("model_type", "")
    if is_xgb:
        pred_encoded = model.predict(X_transformed)[0]
        predicted_class = int_to_target.get(pred_encoded, str(pred_encoded))
    else:
        predicted_class = model.predict(X_transformed)[0]
        
    # Probabilities
    probability = None
    class_probabilities = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_transformed)[0]
        probability = float(np.max(probs))
        class_probabilities = {}
        for i, p in enumerate(probs):
            class_label = int_to_target.get(i, str(classes[i])) if (is_xgb and int_to_target) else str(classes[i]) if i < len(classes) else f"Class_{i}"
            class_probabilities[class_label] = round(float(p), 4)
            
    # SHAP Explanation
    explanation = explain_prediction(
        model_bundle=bundle,
        patient_df_transformed=X_transformed,
        feature_names=bundle.get("feature_names_out", [])
    )
    
    # Save to SQLite history
    pred_id = save_prediction(
        model_name=bundle.get("model_name", model_name),
        predicted_class=str(predicted_class),
        probability=probability,
        features=patient_data,
        class_probabilities=class_probabilities,
        explanation=explanation,
        patient_ref=patient_ref,
        validation_status="Valid",
        db_path=db_path
    )
    
    return {
        "prediction_id": pred_id,
        "model_name": bundle.get("model_name", model_name),
        "predicted_class": str(predicted_class),
        "probability": round(probability, 4) if probability is not None else None,
        "class_probabilities": class_probabilities,
        "explanation": explanation,
        "validation_status": "Valid",
        "timestamp": pd.Timestamp.now().isoformat(),
        "disclaimer": "This result is a preliminary research prediction and is not a medical diagnosis."
    }
