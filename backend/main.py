"""
NeuroPredictAI - FastAPI Main Application Server
Exposes high-performance REST APIs for dataset upload, statistical analytics,
model training, model evaluation benchmarking, patient prediction, and SHAP explainability.
"""

import os
import shutil
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys

# Ensure backend root is on sys.path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Internal core backend modules
from src.database import (
    init_db,
    log_dataset_upload,
    get_active_dataset,
    get_trained_models,
    get_prediction_history,
    DEFAULT_DB_PATH
)
from src.data_loader import (
    load_dataset,
    get_dataset_summary,
    get_dataset_preview,
    get_dataset_analytics
)
from src.train import (
    train_model_pipeline,
    SUPPORTED_MODELS,
    MODELS_DIR
)
from src.prediction import (
    predict_patient,
    load_trained_model
)
from src.explainability import (
    get_global_feature_importance
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, "sample")

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Initialize database schema
init_db()

# Seed default dataset if none exists in database
active_ds = get_active_dataset()
if not active_ds:
    sample_csv = os.path.join(SAMPLE_DATA_DIR, "adni_oasis_dataset.csv")
    if os.path.exists(sample_csv):
        df_sample = load_dataset(sample_csv)
        log_dataset_upload(
            filename="adni_oasis_dataset.csv",
            filepath=sample_csv,
            row_count=len(df_sample),
            column_count=len(df_sample.columns),
            columns=list(df_sample.columns),
            target_column="Alzheimer_Risk_Class" if "Alzheimer_Risk_Class" in df_sample.columns else None
        )

app = FastAPI(
    title="NeuroPredict AI API",
    description="Research Prototype API for Alzheimer's Pattern Identification and Patient Risk Analytics",
    version="2.0.0"
)

# CORS configuration allowing local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory training tracker
training_state = {
    "is_training": False,
    "current_model": None,
    "last_trained_model": None,
    "last_status": "Idle",
    "last_error": None,
    "last_completed_at": None
}

# --- Pydantic Schemas ---

class PredictRequest(BaseModel):
    model_name: str = Field(default="Random Forest", description="Model architecture to use")
    patient_ref: Optional[str] = Field(default=None, description="Optional patient reference ID")
    features: Dict[str, Any] = Field(description="Dynamic key-value map of patient feature values")

class TrainRequest(BaseModel):
    model_type: str = Field(default="random_forest", description="Algorithm: logistic_regression, random_forest, or xgboost")
    target_column: Optional[str] = Field(default=None, description="Target classification column")

# --- Helper functions ---

def get_current_dataset_path() -> str:
    active = get_active_dataset()
    if active and os.path.exists(active["filepath"]):
        return active["filepath"]
    
    sample_csv = os.path.join(SAMPLE_DATA_DIR, "adni_oasis_dataset.csv")
    if os.path.exists(sample_csv):
        return sample_csv
        
    raise HTTPException(status_code=404, detail="No active dataset found. Please upload a dataset.")

# --- Endpoints ---

@app.get("/")
def get_app_status():
    """GET / — application status"""
    return {
        "application": "NeuroPredict AI",
        "team": "Team UNFAZED (DEX26341)",
        "event": "Dexathon 2026",
        "status": "online",
        "disclaimer": "NeuroPredict AI provides preliminary research predictions and does not replace professional medical evaluation."
    }

@app.get("/api/health")
def get_health():
    """GET /api/health — backend health"""
    active = get_active_dataset()
    models = get_trained_models()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected",
        "active_dataset": active["filename"] if active else None,
        "trained_models_count": len(models)
    }

@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """POST /api/dataset/upload — upload CSV dataset"""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are accepted.")
    
    file_path = os.path.join(RAW_DATA_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        df = load_dataset(file_path)
        
        # Detect target column automatically
        target_col = None
        candidates = ["Alzheimer_Risk_Class", "group", "Group", "dx", "Diagnosis", "target", "cdr"]
        for c in df.columns:
            if c in candidates:
                target_col = c
                break
                
        dataset_id = log_dataset_upload(
            filename=file.filename,
            filepath=file_path,
            row_count=len(df),
            column_count=len(df.columns),
            columns=list(df.columns),
            target_column=target_col
        )
        
        summary = get_dataset_summary(df, filename=file.filename)
        
        return {
            "status": "success",
            "message": f"Successfully uploaded and indexed '{file.filename}'.",
            "dataset_id": dataset_id,
            "summary": summary
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process dataset: {str(e)}")

@app.get("/api/dataset/summary")
def get_summary():
    """GET /api/dataset/summary — dataset summary"""
    path = get_current_dataset_path()
    df = load_dataset(path)
    active = get_active_dataset()
    filename = active["filename"] if active else os.path.basename(path)
    return get_dataset_summary(df, filename=filename)

@app.get("/api/dataset/preview")
def get_preview(n_rows: int = Query(default=10, ge=1, le=100)):
    """GET /api/dataset/preview — dataset preview"""
    path = get_current_dataset_path()
    df = load_dataset(path)
    return get_dataset_preview(df, n_rows=n_rows)

@app.get("/api/dataset/columns")
def get_columns():
    """GET /api/dataset/columns — available columns and features for forms"""
    path = get_current_dataset_path()
    df = load_dataset(path)
    summary = get_dataset_summary(df)
    
    # Identify non-identifier features that can be collected via patient form
    input_columns = [
        col for col in summary["columns"]
        if not col["is_identifier"] and not col["is_target_candidate"]
    ]
    
    return {
        "all_columns": list(df.columns),
        "target_candidates": summary["target_candidates"],
        "input_features": input_columns
    }

@app.get("/api/dataset/analytics")
def get_analytics():
    """GET /api/dataset/analytics — dataset statistics, histograms, correlation matrix"""
    path = get_current_dataset_path()
    df = load_dataset(path)
    active = get_active_dataset()
    target_col = active["target_column"] if active else None
    return get_dataset_analytics(df, target_column=target_col)

# Background worker for model training
def run_training_task(model_type: str, target_col: str, filepath: str, filename: str):
    global training_state
    try:
        df = load_dataset(filepath)
        result = train_model_pipeline(
            df=df,
            target_column=target_col,
            model_type=model_type,
            dataset_filename=filename
        )
        training_state["is_training"] = False
        training_state["last_trained_model"] = result["model_name"]
        training_state["last_status"] = "Completed"
        training_state["last_error"] = None
        training_state["last_completed_at"] = datetime.now().isoformat()
    except Exception as e:
        training_state["is_training"] = False
        training_state["last_status"] = "Failed"
        training_state["last_error"] = str(e)

@app.post("/api/model/train")
def train_model(payload: TrainRequest, background_tasks: BackgroundTasks):
    """POST /api/model/train — train selected model"""
    global training_state
    if training_state["is_training"]:
        raise HTTPException(status_code=409, detail="A model training task is already in progress.")
        
    path = get_current_dataset_path()
    active = get_active_dataset()
    filename = active["filename"] if active else os.path.basename(path)
    
    df = load_dataset(path)
    
    # Target column determination
    target_col = payload.target_column or (active["target_column"] if active else None)
    if not target_col or target_col not in df.columns:
        candidates = ["Alzheimer_Risk_Class", "group", "Group", "dx", "Diagnosis", "target", "cdr"]
        for c in df.columns:
            if c in candidates:
                target_col = c
                break
                
    if not target_col or target_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Could not determine valid target column. Available columns: {list(df.columns)}"
        )
        
    training_state["is_training"] = True
    training_state["current_model"] = payload.model_type
    training_state["last_status"] = "Training"
    training_state["last_error"] = None
    
    # Execute training synchronously if dataset is relatively small, or via background task
    # For robust deterministic results and instant API return under 3-5 seconds, run directly
    try:
        result = train_model_pipeline(
            df=df,
            target_column=target_col,
            model_type=payload.model_type,
            dataset_filename=filename
        )
        training_state["is_training"] = False
        training_state["last_trained_model"] = result["model_name"]
        training_state["last_status"] = "Completed"
        training_state["last_completed_at"] = datetime.now().isoformat()
        
        return {
            "status": "success",
            "message": f"Successfully trained {result['model_name']}.",
            "model_name": result["model_name"],
            "model_type": result["model_type"],
            "metrics": result["metrics"]
        }
    except Exception as e:
        training_state["is_training"] = False
        training_state["last_status"] = "Failed"
        training_state["last_error"] = str(e)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/api/model/status")
def get_training_status():
    """GET /api/model/status — training status"""
    return {
        "status": training_state["last_status"],
        "is_training": training_state["is_training"],
        "current_model": training_state["current_model"],
        "last_trained_model": training_state["last_trained_model"],
        "last_error": training_state["last_error"],
        "last_completed_at": training_state["last_completed_at"]
    }

@app.get("/api/model/metrics")
def get_model_metrics(model_name: Optional[str] = Query(default=None)):
    """GET /api/model/metrics — actual metrics for model"""
    models = get_trained_models()
    if not models:
        raise HTTPException(status_code=404, detail="Model has not been trained yet.")
        
    if model_name:
        for m in models:
            if m["model_name"].lower() == model_name.lower() or m["model_type"].lower() == model_name.lower():
                return m
        raise HTTPException(status_code=404, detail=f"No trained model found matching '{model_name}'.")
        
    return models[0]

@app.get("/api/model/compare")
def compare_models():
    """GET /api/model/compare — compare all trained models"""
    models = get_trained_models()
    if not models:
        return {
            "status": "empty",
            "message": "No models have been trained yet.",
            "models": []
        }
        
    comparison = []
    for m in models:
        metrics = m["metrics"]
        comparison.append({
            "model_name": m["model_name"],
            "model_type": m["model_type"],
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1_score"),
            "roc_auc": metrics.get("roc_auc"),
            "sensitivity": metrics.get("sensitivity"),
            "specificity": metrics.get("specificity"),
            "trained_at": m["training_timestamp"]
        })
    return {
        "models": comparison,
        "count": len(comparison)
    }

@app.post("/api/predict")
def generate_prediction(payload: PredictRequest):
    """POST /api/predict — generate patient prediction with SHAP explanation"""
    try:
        result = predict_patient(
            patient_data=payload.features,
            model_name=payload.model_name,
            patient_ref=payload.patient_ref
        )
        return result
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/predict/history")
def get_history(limit: int = Query(default=50, ge=1, le=200)):
    """GET /api/predict/history — prediction history"""
    return get_prediction_history(limit=limit)

@app.get("/api/model/explain")
def get_explanation(model_name: str = Query(default="Random Forest")):
    """GET /api/model/explain — SHAP and feature attribution insights"""
    try:
        bundle = load_trained_model(model_name)
        importance = get_global_feature_importance(bundle)
        return {
            "model_name": bundle.get("model_name", model_name),
            "method": "SHAP Global Feature Importance",
            "feature_importance": importance["features"],
            "disclaimer": "Feature importance reflects statistical attributions within the model and does not prove clinical causality."
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Explainability unavailable: {str(e)}")

@app.get("/api/model/feature-importance")
def get_feature_importance(model_name: str = Query(default="Random Forest")):
    """GET /api/model/feature-importance — feature importance"""
    try:
        bundle = load_trained_model(model_name)
        return get_global_feature_importance(bundle)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Feature importance unavailable: {str(e)}")
