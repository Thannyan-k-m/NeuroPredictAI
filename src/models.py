"""
NeuroPredictAI - Machine Learning Architecture
Implements:
1. Logistic Regression (Baseline)
2. Random Forest (Ensemble)
3. XGBoost (High Performance Gradient Boosting)
Includes probability calibration, Stratified K-Fold validation, and clinical metric evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold
import joblib
import os
import json

def get_base_models(random_state=42):
    """Instantiates the three required architectures according to Dexathon 2026 specs."""
    models = {
        "Logistic Regression (Baseline)": LogisticRegression(
            C=0.8,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        ),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=150,
            max_depth=7,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state
        ),
        "XGBoost (High Performance)": XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=random_state
        )
    }
    return models

def evaluate_predictions(y_true, y_pred, y_prob):
    """Calculates comprehensive clinical diagnostic metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)
        }
    }
    return metrics

def run_cross_validation(X_train, y_train, n_splits=5, random_state=42):
    """
    Performs 5-Fold Stratified Cross-Validation on the training cohort
    to evaluate generalization and prevent overfitting.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    base_models = get_base_models(random_state)
    cv_results = {}
    
    for name, model in base_models.items():
        fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]
            
            fold_metrics["accuracy"].append(accuracy_score(y_val, y_pred))
            fold_metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))
            fold_metrics["recall"].append(recall_score(y_val, y_pred, zero_division=0))
            fold_metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))
            fold_metrics["roc_auc"].append(roc_auc_score(y_val, y_prob))
            
        cv_results[name] = {
            "accuracy_mean": round(float(np.mean(fold_metrics["accuracy"])), 4),
            "accuracy_std": round(float(np.std(fold_metrics["accuracy"])), 4),
            "precision_mean": round(float(np.mean(fold_metrics["precision"])), 4),
            "recall_mean": round(float(np.mean(fold_metrics["recall"])), 4),
            "f1_mean": round(float(np.mean(fold_metrics["f1"])), 4),
            "roc_auc_mean": round(float(np.mean(fold_metrics["roc_auc"])), 4),
            "roc_auc_std": round(float(np.std(fold_metrics["roc_auc"])), 4),
        }
        
    return cv_results

def train_and_calibrate_models(X_train, y_train, X_test, y_test, output_dir):
    """
    Trains all models on training cohort, calibrates probabilities using CalibratedClassifierCV,
    evaluates on the unseen patient test cohort, and exports models & curves.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_models = get_base_models()
    trained_models = {}
    evaluation_reports = {}
    curves_data = {}
    
    for name, base_model in base_models.items():
        # Wrap with CalibratedClassifierCV for clinical probability calibration
        calibrated_model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=3)
        calibrated_model.fit(X_train, y_train)
        
        y_pred = calibrated_model.predict(X_test)
        y_prob = calibrated_model.predict_proba(X_test)[:, 1]
        
        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        evaluation_reports[name] = metrics
        trained_models[name] = calibrated_model
        
        # ROC and PR curves
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        curves_data[name] = {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "precision": [float(x) for x in precision],
            "recall": [float(x) for x in recall],
        }
        
        # Save individual model
        clean_name = name.split()[0].lower()
        model_path = os.path.join(output_dir, f"{clean_name}_calibrated.joblib")
        joblib.dump(calibrated_model, model_path)
    
    # Save metrics summary
    metrics_path = os.path.join(output_dir, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump({"models": evaluation_reports, "curves": curves_data}, f, indent=2)
        
    return trained_models, evaluation_reports, curves_data
