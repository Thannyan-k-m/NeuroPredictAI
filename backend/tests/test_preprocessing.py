import pytest
import pandas as pd
import numpy as np
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from src.preprocessing import DatasetPreprocessor, split_dataset
from src.feature_engineering import engineer_features

def test_feature_engineering():
    sample_df = pd.DataFrame({
        "Gender": ["Female", "Male"],
        "nWBV": [0.72, 0.78],
        "MMSE": [24.0, 29.0],
        "Systolic_BP": [135.0, 120.0],
        "Cholesterol": [210.0, 190.0],
        "Blood_Glucose": [110.0, 95.0],
        "Age": [75, 65]
    })
    
    engineered = engineer_features(sample_df)
    assert "Gender_Encoded" in engineered.columns
    assert engineered["Gender_Encoded"].tolist() == [1, 0]
    assert "Brain_Atrophy_Index" in engineered.columns
    assert "Cognitive_Decline_Ratio" in engineered.columns
    assert "Cardiovascular_Risk_Index" in engineered.columns
    assert "Age_Vulnerability_Score" in engineered.columns

def test_preprocessing_pipeline():
    sample_df = pd.DataFrame({
        "Subject_ID": ["SUBJ_01", "SUBJ_01", "SUBJ_02", "SUBJ_02", "SUBJ_03"],
        "Age": [72, 73, 68, 69, 81],
        "Gender": ["M", "M", "F", "F", "F"],
        "MMSE": [28.0, 27.0, 24.0, 22.0, 30.0],
        "nWBV": [0.75, 0.74, 0.71, 0.70, 0.79],
        "Alzheimer_Risk_Class": [0, 0, 1, 1, 0]
    })
    
    preprocessor = DatasetPreprocessor(target_column="Alzheimer_Risk_Class")
    X, y, groups = preprocessor.prepare_data(sample_df)
    
    assert "Subject_ID" not in X.columns
    assert "Alzheimer_Risk_Class" not in X.columns
    assert groups is not None
    
    X_train, X_test, y_train, y_test = split_dataset(X, y, groups=groups, test_size=0.33, random_state=42)
    assert len(X_train) > 0
    assert len(X_test) > 0
    
    transformed_train = preprocessor.fit_transform(X_train, y_train)
    transformed_test = preprocessor.transform(X_test)
    
    assert transformed_train.shape[1] == transformed_test.shape[1]
    assert len(preprocessor.feature_names_out) == transformed_train.shape[1]
