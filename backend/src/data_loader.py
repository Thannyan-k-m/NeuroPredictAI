"""
NeuroPredictAI - Data Loading and Quality Analysis Module
Handles dataset ingestion, quality checks, missing value profiling, and summary analytics.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

IDENTIFIER_PATTERNS = ["id", "subject", "patient", "visit_id", "mri_id", "subject_id", "patient_id"]
TARGET_CANDIDATE_PATTERNS = ["alzheimer_risk_class", "group", "dx", "diagnosis", "target", "cdr", "cdr_glob", "label", "class"]

def load_dataset(filepath: str) -> pd.DataFrame:
    """Reads a CSV file safely with encoding fallbacks."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    try:
        df = pd.read_csv(filepath)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin1")
    return df

def identify_identifier_columns(df: pd.DataFrame) -> List[str]:
    """Identifies columns that behave like unique patient/record identifiers."""
    id_cols = []
    for col in df.columns:
        col_lower = str(col).lower()
        if any(pattern == col_lower or col_lower.endswith("_" + pattern) or col_lower.startswith(pattern + "_") for pattern in IDENTIFIER_PATTERNS):
            id_cols.append(col)
        elif df[col].nunique() == len(df) and df[col].dtype == object:
            id_cols.append(col)
    return id_cols

def detect_potential_targets(df: pd.DataFrame) -> List[str]:
    """Finds target columns suitable for classification."""
    candidates = []
    for col in df.columns:
        col_lower = str(col).lower()
        if any(cand == col_lower or col_lower.endswith("_" + cand) for cand in TARGET_CANDIDATE_PATTERNS):
            candidates.append(col)
    
    # Fallback to discrete categorical columns with <= 10 unique classes
    if not candidates:
        for col in df.columns:
            if 2 <= df[col].nunique() <= 10:
                candidates.append(col)
    return candidates

def get_dataset_summary(df: pd.DataFrame, filename: str = "dataset.csv") -> Dict[str, Any]:
    """Computes comprehensive dataset quality and statistical overview."""
    total_records = len(df)
    total_columns = len(df.columns)
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    missing_counts = df.isnull().sum()
    missing_dict = {col: int(missing_counts[col]) for col in df.columns if missing_counts[col] > 0}
    total_missing_cells = int(missing_counts.sum())
    
    duplicate_count = int(df.duplicated().sum())
    id_cols = identify_identifier_columns(df)
    target_candidates = detect_potential_targets(df)
    
    column_details = []
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        col_missing = int(df[col].isnull().sum())
        missing_pct = round((col_missing / total_records) * 100, 2) if total_records > 0 else 0
        n_unique = int(df[col].nunique())
        
        detail = {
            "name": col,
            "type": "numerical" if col in num_cols else "categorical",
            "dtype": dtype_str,
            "missing_count": col_missing,
            "missing_percentage": missing_pct,
            "unique_values": n_unique,
            "is_identifier": col in id_cols,
            "is_target_candidate": col in target_candidates
        }
        
        if col in num_cols:
            valid_series = df[col].dropna()
            if not valid_series.empty:
                detail["mean"] = round(float(valid_series.mean()), 3)
                detail["std"] = round(float(valid_series.std()), 3)
                detail["min"] = round(float(valid_series.min()), 3)
                detail["max"] = round(float(valid_series.max()), 3)
                detail["median"] = round(float(valid_series.median()), 3)
        column_details.append(detail)
        
    return {
        "filename": filename,
        "total_records": total_records,
        "total_columns": total_columns,
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "identifier_columns": id_cols,
        "target_candidates": target_candidates,
        "missing_summary": {
            "total_missing_cells": total_missing_cells,
            "columns_with_missing": missing_dict
        },
        "duplicate_records": duplicate_count,
        "columns": column_details
    }

def get_dataset_preview(df: pd.DataFrame, n_rows: int = 10) -> Dict[str, Any]:
    """Generates preview rows serialized safely for JSON."""
    preview_df = df.head(n_rows).replace({np.nan: None})
    return {
        "columns": list(df.columns),
        "data": preview_df.to_dict(orient="records"),
        "total_records": len(df)
    }

def get_dataset_analytics(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
    """Generates real distributions, class counts, and correlation matrix for visualization."""
    total_records = len(df)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Target distribution
    target_dist = {}
    if target_column and target_column in df.columns:
        counts = df[target_column].value_counts(dropna=False).to_dict()
        target_dist = {str(k) if not pd.isna(k) else "Missing": int(v) for k, v in counts.items()}
    elif "Alzheimer_Risk_Class" in df.columns:
        counts = df["Alzheimer_Risk_Class"].value_counts().to_dict()
        target_dist = {str(k): int(v) for k, v in counts.items()}
    elif "Group" in df.columns:
        counts = df["Group"].value_counts().to_dict()
        target_dist = {str(k): int(v) for k, v in counts.items()}
        
    # Feature distributions (histograms for up to 6 key numeric features)
    histograms = {}
    key_features = [c for c in ["Age", "MMSE", "CDR", "eTIV", "nWBV", "ASF", "MoCA", "FAQ"] if c in df.columns]
    if not key_features:
        key_features = num_cols[:6]
        
    for feat in key_features:
        series = df[feat].dropna()
        if not series.empty and series.nunique() > 1:
            counts, bin_edges = np.histogram(series, bins=10)
            labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]
            histograms[feat] = {
                "labels": labels,
                "counts": [int(c) for c in counts],
                "mean": round(float(series.mean()), 2),
                "std": round(float(series.std()), 2)
            }
            
    # Correlation Matrix
    corr_matrix = {}
    corr_features = [c for c in num_cols if c not in identify_identifier_columns(df)][:10]
    if len(corr_features) >= 2:
        corr_df = df[corr_features].corr().replace({np.nan: 0.0})
        corr_matrix = {
            "features": corr_features,
            "values": [[round(float(val), 2) for val in row] for row in corr_df.values]
        }
        
    return {
        "total_records": total_records,
        "target_distribution": target_dist,
        "feature_histograms": histograms,
        "correlation_matrix": corr_matrix
    }
