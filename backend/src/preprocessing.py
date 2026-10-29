"""
NeuroPredictAI - Preprocessing Pipeline Module
Implements missing-value detection, numerical imputation, categorical encoding,
and scikit-learn ColumnTransformer / Pipeline integration without data leakage.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from .feature_engineering import engineer_features
from .data_loader import identify_identifier_columns

class DatasetPreprocessor:
    """
    Robust Preprocessing Engine capable of handling diverse tabular clinical datasets:
    1. Feature engineering (if compatible clinical columns exist)
    2. Automatic numerical / categorical feature segregation
    3. Missing value imputation (median for numeric, most_frequent for categorical)
    4. Scaling (StandardScaler for numeric) and Encoding (OneHotEncoder for categorical)
    5. Patient/subject grouping to prevent data leakage across longitudinal visits
    """

    def __init__(self, target_column: str, exclude_columns: Optional[List[str]] = None):
        self.target_column = target_column
        self.exclude_columns = exclude_columns or []
        self.pipeline: Optional[ColumnTransformer] = None
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.feature_names_out: List[str] = []
        self.classes_: Optional[np.ndarray] = None
        self.is_fitted = False

    def prepare_data(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Prepares raw DataFrame: applies feature engineering, isolates features, target, and groups."""
        df = df_raw.copy()
        
        # Apply domain feature engineering
        df = engineer_features(df)
        
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset columns.")
            
        y = df[self.target_column]
        
        # Check for patient/subject group column
        group_col = None
        for col in df.columns:
            if col.lower() in ["subject_id", "patient_id", "subject", "patient"]:
                group_col = col
                break
                
        groups = df[group_col] if group_col and group_col != self.target_column else None
        
        # Exclude identifiers, target, and specified exclusions
        auto_ids = identify_identifier_columns(df)
        drop_cols = set(auto_ids + self.exclude_columns + [self.target_column])
        if group_col:
            drop_cols.add(group_col)
            
        feature_cols = [c for c in df.columns if c not in drop_cols]
        X = df[feature_cols]
        
        return X, y, groups

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fits ColumnTransformer on training features only (preventing leakage)."""
        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        transformers = []
        
        if self.numerical_cols:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            transformers.append(("num", num_pipe, self.numerical_cols))
            
        if self.categorical_cols:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            transformers.append(("cat", cat_pipe, self.categorical_cols))
            
        self.pipeline = ColumnTransformer(transformers=transformers, remainder="drop")
        self.pipeline.fit(X)
        
        # Reconstruct output feature names
        out_features = []
        if self.numerical_cols:
            out_features.extend(self.numerical_cols)
        if self.categorical_cols:
            cat_encoder = self.pipeline.named_transformers_["cat"].named_steps["encoder"]
            cat_encoded_names = list(cat_encoder.get_feature_names_out(self.categorical_cols))
            out_features.extend(cat_encoded_names)
            
        self.feature_names_out = out_features
        if y is not None:
            self.classes_ = np.unique(y.dropna())
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms features using fitted pipeline."""
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Preprocessor must be fitted before transforming.")
        return self.pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fits and transforms in a single pass."""
        self.fit(X, y)
        return self.transform(X)

def split_dataset(X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None, 
                  test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into train and test sets, using group-aware split if groups are present."""
    if groups is not None and groups.nunique() > 1:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    else:
        # Check if stratification is feasible
        stratify = y if (y.nunique() > 1 and y.value_counts().min() >= 2) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    return X_train, X_test, y_train, y_test
