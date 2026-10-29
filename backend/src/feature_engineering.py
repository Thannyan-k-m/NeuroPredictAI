"""
NeuroPredictAI - Feature Engineering Module
Applies clinically validated feature engineering transformations on tabular patient records.
"""

import pandas as pd
import numpy as np
from typing import Tuple

def engineer_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Applies domain-informed feature engineering:
    - Gender encoding (Female = 1, Male = 0)
    - Brain Atrophy Index (derived from normalized whole brain volume nWBV)
    - Cognitive Decline Ratio (proportion of points lost on 30-pt MMSE)
    - Cardiovascular Risk Index (composite blood pressure, cholesterol, glucose)
    - Age Vulnerability Factor
    """
    df = df_input.copy()

    # Column name normalization helpers
    cols_lower = {str(c).lower(): c for c in df.columns}

    # 1. Gender encoding
    if "Gender_Encoded" not in df.columns:
        gender_col = cols_lower.get("gender") or cols_lower.get("sex")
        if gender_col:
            df["Gender_Encoded"] = df[gender_col].apply(
                lambda g: 1 if str(g).strip().upper().startswith("F") or str(g) == "1" else 0
            )

    # 2. Extract or impute series
    nwbv_col = cols_lower.get("nwbv")
    if nwbv_col:
        df["Brain_Atrophy_Index"] = 1.0 - pd.to_numeric(df[nwbv_col], errors="coerce").fillna(0.75)

    mmse_col = cols_lower.get("mmse")
    if mmse_col:
        mmse_val = pd.to_numeric(df[mmse_col], errors="coerce").fillna(27.0)
        df["Cognitive_Decline_Ratio"] = (30.0 - mmse_val).clip(lower=0) / 30.0

    sys_bp_col = cols_lower.get("systolic_bp") or cols_lower.get("systolicbp") or cols_lower.get("bp")
    chol_col = cols_lower.get("cholesterol")
    glu_col = cols_lower.get("blood_glucose") or cols_lower.get("glucose")

    if sys_bp_col or chol_col or glu_col:
        bp_val = pd.to_numeric(df[sys_bp_col], errors="coerce").fillna(125.0) if sys_bp_col else 125.0
        chol_val = pd.to_numeric(df[chol_col], errors="coerce").fillna(200.0) if chol_col else 200.0
        glu_val = pd.to_numeric(df[glu_col], errors="coerce").fillna(100.0) if glu_col else 100.0
        
        df["Cardiovascular_Risk_Index"] = (
            (bp_val / 120.0) + (chol_val / 200.0) + (glu_val / 100.0)
        ) / 3.0

    age_col = cols_lower.get("age")
    if age_col:
        age_val = pd.to_numeric(df[age_col], errors="coerce").fillna(70.0)
        df["Age_Vulnerability_Score"] = (age_val - 60.0).clip(lower=0) / 10.0

    return df
