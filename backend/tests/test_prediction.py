import pytest
import os
import sys
import tempfile
import pandas as pd

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.train import train_model_pipeline
from src.prediction import predict_patient

def test_training_and_prediction():
    # Synthetic small dataset
    df = pd.DataFrame({
        "Subject_ID": [f"PT_{i}" for i in range(30)],
        "Age": [60 + i % 25 for i in range(30)],
        "Gender": ["M" if i % 2 == 0 else "F" for i in range(30)],
        "MMSE": [20 + (i % 10) for i in range(30)],
        "nWBV": [0.65 + (i % 15) * 0.01 for i in range(30)],
        "Alzheimer_Risk_Class": [0 if i < 15 else 1 for i in range(30)]
    })
    
    with tempfile.TemporaryDirectory() as temp_dir:
        models_dir = os.path.join(temp_dir, "models")
        db_path = os.path.join(temp_dir, "test.sqlite")
        
        train_result = train_model_pipeline(
            df=df,
            target_column="Alzheimer_Risk_Class",
            model_type="random_forest",
            dataset_filename="synthetic.csv",
            models_dir=models_dir,
            db_path=db_path
        )
        
        assert train_result["status"] == "success"
        assert "metrics" in train_result
        assert "accuracy" in train_result["metrics"]
        
        # Test inference
        patient_sample = {
            "Age": 74,
            "Gender": "F",
            "MMSE": 22.0,
            "nWBV": 0.69
        }
        
        pred = predict_patient(
            patient_data=patient_sample,
            model_name="random_forest",
            patient_ref="PATIENT_SYNTHETIC",
            models_dir=models_dir,
            db_path=db_path
        )
        
        assert "predicted_class" in pred
        assert "probability" in pred
        assert "explanation" in pred
        assert pred["validation_status"] == "Valid"
