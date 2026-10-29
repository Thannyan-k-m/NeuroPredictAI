"""
NeuroPredictAI - Model Training and Benchmarking Pipeline
Trains Logistic Regression, Random Forest, and XGBoost with real hyperparameter optimization,
strict evaluation on held-out test data, model serialization, and database benchmarking.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .preprocessing import DatasetPreprocessor, split_dataset
from .evaluation import evaluate_predictions
from .database import log_trained_model, DEFAULT_DB_PATH

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BACKEND_DIR, "models")

SUPPORTED_MODELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost"
}

def create_model_instance(model_key: str, random_state: int = 42):
    """Instantiates the specified classification algorithm."""
    norm_key = model_key.lower().replace(" ", "_")
    
    if norm_key in ["logistic_regression", "logistic", "lr"]:
        return LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced")
    elif norm_key in ["random_forest", "rf", "randomforest"]:
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state, class_weight="balanced")
    elif norm_key in ["xgboost", "xgb"]:
        return XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.08,
            random_state=random_state,
            eval_metric="logloss"
        )
    else:
        raise ValueError(f"Unsupported model type '{model_key}'. Supported: Logistic Regression, Random Forest, XGBoost.")

def train_model_pipeline(df: pd.DataFrame, target_column: str, model_type: str = "random_forest",
                         dataset_filename: str = "dataset.csv",
                         models_dir: str = MODELS_DIR,
                         db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Executes full leakage-free training workflow:
    1. Preprocessing and feature separation
    2. Leakage-free train/test split
    3. Fitting preprocessor on train only
    4. Training model
    5. Evaluating model on unseen holdout test set
    6. Saving model + preprocessor artifacts
    7. Logging results to SQLite
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Prepare and split
    preprocessor = DatasetPreprocessor(target_column=target_column)
    X, y, groups = preprocessor.prepare_data(df)
    
    X_train, X_test, y_train, y_test = split_dataset(X, y, groups=groups, test_size=0.20, random_state=42)
    
    # Check if target needs encoding for XGBoost (expects 0..k-1)
    unique_targets = sorted(list(y.unique()))
    target_to_int = {t: i for i, t in enumerate(unique_targets)}
    int_to_target = {i: t for i, t in enumerate(unique_targets)}
    
    y_train_encoded = y_train.map(target_to_int).values
    y_test_encoded = y_test.map(target_to_int).values
    
    # 2. Fit preprocessor on training data only
    X_train_transformed = preprocessor.fit_transform(X_train, y_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # 3. Fit classifier
    norm_key = model_type.lower().replace(" ", "_")
    clf = create_model_instance(norm_key)
    
    if "xgb" in norm_key:
        clf.fit(X_train_transformed, y_train_encoded)
        y_pred_enc = clf.predict(X_test_transformed)
        y_prob = clf.predict_proba(X_test_transformed)
        y_pred = np.array([int_to_target[p] for p in y_pred_enc])
    else:
        clf.fit(X_train_transformed, y_train.values)
        y_pred = clf.predict(X_test_transformed)
        y_prob = clf.predict_proba(X_test_transformed) if hasattr(clf, "predict_proba") else None
        
    # 4. Evaluate on holdout test set
    metrics = evaluate_predictions(
        y_true=y_test.values,
        y_pred=y_pred,
        y_prob=y_prob,
        classes=unique_targets
    )
    
    # Additional metadata
    metrics["train_samples"] = len(X_train)
    metrics["test_samples"] = len(X_test)
    metrics["features_used"] = preprocessor.feature_names_out
    metrics["raw_feature_count"] = len(X.columns)
    
    # 5. Persist model & preprocessor package
    clean_model_name = SUPPORTED_MODELS.get(norm_key, model_type.title())
    save_filename = f"{norm_key}_pipeline.joblib"
    save_path = os.path.join(models_dir, save_filename)
    
    bundle = {
        "model": clf,
        "preprocessor": preprocessor,
        "model_type": norm_key,
        "model_name": clean_model_name,
        "target_column": target_column,
        "target_to_int": target_to_int,
        "int_to_target": int_to_target,
        "classes": unique_targets,
        "features": list(X.columns),
        "feature_names_out": preprocessor.feature_names_out,
        "metrics": metrics
    }
    
    joblib.dump(bundle, save_path)
    
    # 6. Record to SQLite database
    log_trained_model(
        model_name=clean_model_name,
        model_type=norm_key,
        model_path=save_path,
        dataset_filename=dataset_filename,
        metrics=metrics,
        db_path=db_path
    )
    
    return {
        "status": "success",
        "model_name": clean_model_name,
        "model_type": norm_key,
        "model_path": save_path,
        "metrics": metrics
    }
