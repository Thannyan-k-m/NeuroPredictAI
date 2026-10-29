"""
NeuroPredictAI - Model Evaluation Module
Calculates comprehensive classification performance metrics:
Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC, Sensitivity, Specificity.
Handles both binary and multiclass scenarios with JSON-compatible serialization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                         y_prob: Optional[np.ndarray] = None, 
                         classes: Optional[List[Any]] = None) -> Dict[str, Any]:
    """
    Computes all standard clinical model evaluation metrics:
    - Accuracy
    - Precision (macro and weighted)
    - Recall / Sensitivity (macro and weighted)
    - F1-Score (macro and weighted)
    - Specificity
    - Confusion Matrix (labels, matrix grid)
    - ROC-AUC (where probabilities are available)
    - Classification breakdown per class
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    unique_labels = sorted(list(set(y_true).union(set(y_pred)))) if classes is None else list(classes)
    num_classes = len(unique_labels)
    is_binary = num_classes == 2
    
    # 1. Base Metrics
    acc = float(accuracy_score(y_true, y_pred))
    
    avg_mode = "binary" if is_binary else "macro"
    prec = float(precision_score(y_true, y_pred, average=avg_mode, zero_division=0))
    rec = float(recall_score(y_true, y_pred, average=avg_mode, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average=avg_mode, zero_division=0))
    
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    cm_list = cm.tolist()
    
    # 3. Sensitivity & Specificity
    sensitivity = rec # Equivalent to recall for positive class
    specificity = None
    if is_binary and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        # Multiclass macro-averaged specificity
        specs = []
        for i in range(len(unique_labels)):
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            fp = np.sum(cm[:, i]) - cm[i, i]
            specs.append(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0)
        specificity = float(np.mean(specs)) if specs else None
        
    # 4. ROC-AUC
    roc_auc = None
    if y_prob is not None:
        try:
            if is_binary:
                # If 2D probability matrix, select positive column
                prob_scores = y_prob[:, 1] if (y_prob.ndim == 2 and y_prob.shape[1] == 2) else y_prob
                # Encode binary labels to 0 and 1
                pos_label = unique_labels[1]
                y_true_bin = (y_true == pos_label).astype(int)
                roc_auc = float(roc_auc_score(y_true_bin, prob_scores))
            else:
                if y_prob.ndim == 2 and y_prob.shape[1] == num_classes:
                    roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            roc_auc = None
            
    # Per-class metrics
    class_report_dict = classification_report(y_true, y_pred, labels=unique_labels, output_dict=True, zero_division=0)
    
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "f1_weighted": round(f1_weighted, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_weighted": round(rec_weighted, 4),
        "sensitivity": round(sensitivity, 4) if sensitivity is not None else None,
        "specificity": round(specificity, 4) if specificity is not None else None,
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": {
            "labels": [str(lbl) for lbl in unique_labels],
            "matrix": cm_list
        },
        "per_class_metrics": {
            str(lbl): {
                "precision": round(class_report_dict.get(str(lbl), {}).get("precision", 0.0), 4),
                "recall": round(class_report_dict.get(str(lbl), {}).get("recall", 0.0), 4),
                "f1_score": round(class_report_dict.get(str(lbl), {}).get("f1-score", 0.0), 4),
                "support": class_report_dict.get(str(lbl), {}).get("support", 0)
            } for lbl in unique_labels if str(lbl) in class_report_dict
        }
    }
