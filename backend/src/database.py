"""
NeuroPredictAI - Database Management Module
Handles SQLite schema, connection management, dataset metadata, and prediction history.
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(BACKEND_DIR, "data", "neuropredict.sqlite")

def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Returns a SQLite connection with dict-like row access."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: str = DEFAULT_DB_PATH):
    """Initializes tables for datasets, models, prediction history, and audit logs."""
    conn = get_connection(db_path)
    cur = conn.cursor()

    # 1. Datasets registry table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        filepath TEXT NOT NULL,
        row_count INTEGER NOT NULL,
        column_count INTEGER NOT NULL,
        columns_json TEXT NOT NULL,
        target_column TEXT,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_active INTEGER DEFAULT 1
    );
    """)

    # 2. Model information and performance benchmark table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS models_registry (
        model_id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        model_type TEXT NOT NULL,
        model_path TEXT NOT NULL,
        dataset_filename TEXT,
        metrics_json TEXT NOT NULL,
        training_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_active INTEGER DEFAULT 1
    );
    """)

    # 3. Patient prediction history table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS prediction_history (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_ref TEXT,
        model_name TEXT NOT NULL,
        predicted_class TEXT NOT NULL,
        probability REAL,
        class_probabilities_json TEXT,
        features_json TEXT NOT NULL,
        explanation_json TEXT,
        validation_status TEXT DEFAULT 'Valid',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    conn.close()

def log_dataset_upload(filename: str, filepath: str, row_count: int, column_count: int, 
                       columns: list, target_column: Optional[str] = None, 
                       db_path: str = DEFAULT_DB_PATH) -> int:
    """Records an uploaded dataset and marks prior datasets inactive."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("UPDATE datasets SET is_active = 0 WHERE is_active = 1")
    cur.execute("""
        INSERT INTO datasets (filename, filepath, row_count, column_count, columns_json, target_column, is_active)
        VALUES (?, ?, ?, ?, ?, ?, 1)
    """, (filename, filepath, row_count, column_count, json.dumps(columns), target_column))
    dataset_id = cur.lastrowid
    conn.commit()
    conn.close()
    return dataset_id

def get_active_dataset(db_path: str = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieves metadata of current active dataset."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM datasets WHERE is_active = 1 ORDER BY dataset_id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    if row:
        return {
            "dataset_id": row["dataset_id"],
            "filename": row["filename"],
            "filepath": row["filepath"],
            "row_count": row["row_count"],
            "column_count": row["column_count"],
            "columns": json.loads(row["columns_json"]),
            "target_column": row["target_column"],
            "uploaded_at": row["uploaded_at"]
        }
    return None

def log_trained_model(model_name: str, model_type: str, model_path: str,
                      dataset_filename: str, metrics: dict, db_path: str = DEFAULT_DB_PATH) -> int:
    """Records or updates trained model in the registry."""
    init_db(db_path)
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO models_registry (model_name, model_type, model_path, dataset_filename, metrics_json, is_active)
        VALUES (?, ?, ?, ?, ?, 1)
    """, (model_name, model_type, model_path, dataset_filename, json.dumps(metrics)))
    model_id = cur.lastrowid
    conn.commit()
    conn.close()
    return model_id

def get_trained_models(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Fetches all registered trained models and their latest metrics."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM models_registry WHERE is_active = 1 ORDER BY training_timestamp DESC")
    rows = cur.fetchall()
    conn.close()
    
    seen_names = set()
    models = []
    for row in rows:
        name = row["model_name"]
        if name not in seen_names:
            seen_names.add(name)
            models.append({
                "model_id": row["model_id"],
                "model_name": row["model_name"],
                "model_type": row["model_type"],
                "model_path": row["model_path"],
                "dataset_filename": row["dataset_filename"],
                "metrics": json.loads(row["metrics_json"]),
                "training_timestamp": row["training_timestamp"]
            })
    return models

def save_prediction(model_name: str, predicted_class: str, probability: Optional[float],
                    features: dict, class_probabilities: Optional[dict] = None,
                    explanation: Optional[dict] = None, patient_ref: Optional[str] = None,
                    validation_status: str = "Valid", db_path: str = DEFAULT_DB_PATH) -> int:
    """Stores a single prediction record with full audit parameters."""
    init_db(db_path)
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO prediction_history (
            patient_ref, model_name, predicted_class, probability,
            class_probabilities_json, features_json, explanation_json, validation_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_ref,
        model_name,
        str(predicted_class),
        float(probability) if probability is not None else None,
        json.dumps(class_probabilities) if class_probabilities else None,
        json.dumps(features),
        json.dumps(explanation) if explanation else None,
        validation_status
    ))
    pred_id = cur.lastrowid
    conn.commit()
    conn.close()
    return pred_id

def get_prediction_history(limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves recent predictions."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT prediction_id, patient_ref, model_name, predicted_class, probability,
               class_probabilities_json, features_json, explanation_json, validation_status, created_at
        FROM prediction_history
        ORDER BY prediction_id DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()

    history = []
    for r in rows:
        history.append({
            "prediction_id": r["prediction_id"],
            "patient_ref": r["patient_ref"],
            "model_name": r["model_name"],
            "predicted_class": r["predicted_class"],
            "probability": r["probability"],
            "class_probabilities": json.loads(r["class_probabilities_json"]) if r["class_probabilities_json"] else None,
            "features": json.loads(r["features_json"]) if r["features_json"] else {},
            "explanation": json.loads(r["explanation_json"]) if r["explanation_json"] else None,
            "validation_status": r["validation_status"],
            "created_at": r["created_at"]
        })
    return history
