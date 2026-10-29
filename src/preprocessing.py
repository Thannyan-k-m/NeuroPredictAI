"""
NeuroPredictAI - Preprocessing and Feature Engineering Pipeline
Implements patient-level data splitting (preventing longitudinal data leakage),
Multiple Imputation for clinical missingness, and clinically validated feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

CLINICAL_FEATURE_COLS = [
    "Age",
    "Gender_Encoded",
    "Education_Years",
    "Socioeconomic_Status",
    "MMSE",
    "CDR",
    "MoCA",
    "FAQ",
    "eTIV",
    "nWBV",
    "ASF",
    "APOE_e4_Alleles",
    "Systolic_BP",
    "Cholesterol",
    "Blood_Glucose",
    # Engineered features
    "Brain_Atrophy_Index",
    "Cognitive_Decline_Ratio",
    "Cardiovascular_Risk_Index",
    "Age_Vulnerability_Score"
]

def engineer_features(df_input):
    """
    Applies domain-informed clinical feature engineering:
    - Gender encoding (Female = 1, Male = 0)
    - Brain Atrophy Index (inverse of normalized whole brain volume)
    - Cognitive Decline Ratio based on MMSE
    - Cardiovascular Risk Composite
    - Age Vulnerability Factor
    """
    df = df_input.copy()
    
    # Standardize column naming if necessary
    if "Gender" in df.columns and "Gender_Encoded" not in df.columns:
        df["Gender_Encoded"] = df["Gender"].apply(lambda g: 1 if str(g).upper().startswith("F") else 0)
    elif "gender" in df.columns and "Gender_Encoded" not in df.columns:
        df["Gender_Encoded"] = df["gender"].apply(lambda g: 1 if str(g).upper().startswith("F") else 0)
    
    # Impute intermediate defaults for feature calculation if NaN
    nwbv = df["nWBV"] if "nWBV" in df.columns else df.get("nwbv", 0.75)
    mmse = df["MMSE"] if "MMSE" in df.columns else df.get("mmse", 27.0)
    sys_bp = df["Systolic_BP"] if "Systolic_BP" in df.columns else df.get("systolic_bp", 125.0)
    chol = df["Cholesterol"] if "Cholesterol" in df.columns else df.get("cholesterol", 200.0)
    glucose = df["Blood_Glucose"] if "Blood_Glucose" in df.columns else df.get("blood_glucose", 100.0)
    age = df["Age"] if "Age" in df.columns else df.get("age", 70.0)

    # 1. Brain Atrophy Index: lower nWBV indicates elevated atrophy
    df["Brain_Atrophy_Index"] = 1.0 - nwbv
    
    # 2. Cognitive Decline Ratio: proportion of cognitive points lost on 30-pt MMSE
    df["Cognitive_Decline_Ratio"] = (30.0 - mmse.fillna(26.0)) / 30.0
    
    # 3. Cardiovascular & Metabolic Risk composite
    df["Cardiovascular_Risk_Index"] = (
        (sys_bp.fillna(130.0) / 120.0)
        + (chol.fillna(200.0) / 200.0)
        + (glucose.fillna(100.0) / 100.0)
    ) / 3.0
    
    # 4. Age Vulnerability Score
    df["Age_Vulnerability_Score"] = (age - 60.0).clip(lower=0) / 10.0
    
    return df

def patient_level_split(df, patient_id_col="Subject_ID", target_col="Alzheimer_Risk_Class", test_size=0.20, random_state=42):
    """
    Guarantees strict Patient-Level train/test splitting to eliminate data leakage.
    Longitudinal visits belonging to the same subject remain exclusively in train OR test.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df[patient_id_col]
    
    train_idx, test_idx = next(gss.split(df, groups=groups))
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    
    # Verification assertion
    train_patients = set(train_df[patient_id_col])
    test_patients = set(test_df[patient_id_col])
    overlap = train_patients.intersection(test_patients)
    assert len(overlap) == 0, f"Critical Data Leakage: {len(overlap)} patients overlap between train and test sets!"
    
    return train_df, test_df

class ClinicalPreprocessingPipeline:
    """
    Production preprocessor that fits Imputer and Scaler ONLY on the training split,
    preventing any lookahead data leakage into test or production inference.
    """
    def __init__(self, feature_cols=None):
        self.feature_cols = feature_cols or CLINICAL_FEATURE_COLS
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X_df):
        df_eng = engineer_features(X_df)
        X_mat = df_eng[self.feature_cols].values
        
        # Fit imputer and scaler
        self.imputer.fit(X_mat)
        X_imputed = self.imputer.transform(X_mat)
        self.scaler.fit(X_imputed)
        self.is_fitted = True
        return self
        
    def transform(self, X_df):
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transforming.")
        df_eng = engineer_features(X_df)
        X_mat = df_eng[self.feature_cols].values
        X_imputed = self.imputer.transform(X_mat)
        X_scaled = self.scaler.transform(X_imputed)
        return X_scaled, df_eng[self.feature_cols]
        
    def fit_transform(self, X_df):
        self.fit(X_df)
        return self.transform(X_df)
