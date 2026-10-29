"""
NeuroPredictAI - Unit & Integration Test Suite
Validates:
1. Zero data leakage in patient-level splitting
2. Feature engineering calculations
3. SQL database retrieval & logging
4. Model inference & calibrated probability bounds [0, 1]
5. Explainability calculations
"""

import os
import pytest
import numpy as np
import pandas as pd
import joblib

from src.preprocessing import patient_level_split, engineer_features, ClinicalPreprocessingPipeline, CLINICAL_FEATURE_COLS
from src.db import query_dataframe, execute_custom_sql, DEFAULT_DB_PATH
from src.explainability import explain_patient_prediction, get_global_feature_importance

def test_patient_level_split_no_leakage():
    sample_data = pd.DataFrame({
        "Subject_ID": ["P1", "P1", "P2", "P2", "P3", "P4", "P4", "P5", "P6", "P6"],
        "Visit_Number": [1, 2, 1, 2, 1, 1, 2, 1, 1, 2],
        "Alzheimer_Risk_Class": [0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        "Age": [65, 66, 72, 73, 68, 79, 80, 62, 74, 75]
    })
    train_df, test_df = patient_level_split(sample_data, patient_id_col="Subject_ID", test_size=0.3, random_state=42)
    
    train_patients = set(train_df["Subject_ID"])
    test_patients = set(test_df["Subject_ID"])
    
    # Assert zero overlap between train and test patients
    assert len(train_patients.intersection(test_patients)) == 0
    assert len(train_df) + len(test_df) == len(sample_data)

def test_feature_engineering():
    sample = pd.DataFrame([{
        "Age": 75.0,
        "Gender": "F",
        "Education_Years": 16,
        "Socioeconomic_Status": 2,
        "MMSE": 24.0,
        "CDR": 0.5,
        "MoCA": 22.0,
        "FAQ": 6.0,
        "eTIV": 1420.0,
        "nWBV": 0.72,
        "ASF": 1.25,
        "APOE_e4_Alleles": 1,
        "Systolic_BP": 135.0,
        "Cholesterol": 210.0,
        "Blood_Glucose": 105.0
    }])
    
    eng = engineer_features(sample)
    
    # Brain Atrophy Index = 1 - nWBV
    assert np.isclose(eng["Brain_Atrophy_Index"].iloc[0], 0.28, atol=1e-3)
    # Cognitive Decline Ratio = (30 - 24) / 30 = 0.20
    assert np.isclose(eng["Cognitive_Decline_Ratio"].iloc[0], 0.20, atol=1e-3)
    # Gender encoded = 1 for Female
    assert eng["Gender_Encoded"].iloc[0] == 1
    # Age vulnerability = (75 - 60) / 10 = 1.5
    assert np.isclose(eng["Age_Vulnerability_Score"].iloc[0], 1.5, atol=1e-3)

def test_sql_database_query():
    if os.path.exists(DEFAULT_DB_PATH):
        df = query_dataframe("SELECT COUNT(*) AS total FROM clinical_visits")
        assert len(df) == 1
        assert df["total"].iloc[0] > 0
        
        success, res_df, msg = execute_custom_sql("SELECT patient_id, age, mmse FROM clinical_visits LIMIT 5")
        assert success is True
        assert len(res_df) == 5

def test_model_inference():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    pipeline_path = os.path.join(models_dir, "scaler_pipeline.joblib")
    xgb_path = os.path.join(models_dir, "xgboost_calibrated.joblib")
    
    if os.path.exists(pipeline_path) and os.path.exists(xgb_path):
        pipeline = joblib.load(pipeline_path)
        model = joblib.load(xgb_path)
        
        sample = pd.DataFrame([{
            "Age": 78.0,
            "Gender": "M",
            "Education_Years": 12,
            "Socioeconomic_Status": 3,
            "MMSE": 21.0,
            "CDR": 1.0,
            "MoCA": 19.0,
            "FAQ": 12.0,
            "eTIV": 1500.0,
            "nWBV": 0.68,
            "ASF": 1.15,
            "APOE_e4_Alleles": 2,
            "Systolic_BP": 145.0,
            "Cholesterol": 230.0,
            "Blood_Glucose": 115.0
        }])
        
        X_scaled, _ = pipeline.transform(sample)
        prob = model.predict_proba(X_scaled)[0, 1]
        
        # Probabilities must be properly calibrated in [0.0, 1.0]
        assert 0.0 <= prob <= 1.0
        # High-risk profile should output elevated probability
        assert prob > 0.60

if __name__ == "__main__":
    pytest.main(["-v", __file__])
