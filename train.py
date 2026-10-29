"""
NeuroPredictAI - Complete Model Training & Benchmark Pipeline
Executes end-to-end training following DEXATHON 2026 specifications:
- Data retrieval from SQLite database
- Patient-level splitting (zero data leakage)
- Feature engineering & multiple imputation
- 5-Fold Stratified Cross Validation
- Calibrated Training for Logistic Regression, Random Forest, & XGBoost
- Evaluation on holdout test set & saving model artifacts
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from src.db import init_db, query_dataframe
from src.preprocessing import patient_level_split, ClinicalPreprocessingPipeline, CLINICAL_FEATURE_COLS
from src.models import run_cross_validation, train_and_calibrate_models
from src.explainability import get_global_feature_importance

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, "adni_oasis_dataset.csv")
    db_path = os.path.join(data_dir, "neuropredict.sqlite")
    
    print("=" * 65)
    print("NeuroPredictAI - DEXATHON 2026 Model Training Pipeline")
    print("=" * 65)
    
    # 1. SQL Data Retrieval & Database Initialization
    print("\n[Step 1/6] Initializing SQLite database and querying clinical cohort...")
    init_db(db_path=db_path, seed_csv_path=csv_path)
    
    sql_query = """
    SELECT 
        v.visit_id, v.patient_id AS Subject_ID, v.visit_number AS Visit_Number,
        v.age AS Age, p.gender AS Gender, p.education_years AS Education_Years,
        v.ses AS Socioeconomic_Status, v.mmse AS MMSE, v.cdr AS CDR,
        v.moca AS MoCA, v.faq AS FAQ, v.etiv AS eTIV, v.nwbv AS nWBV,
        v.asf AS ASF, v.apoe_e4_alleles AS APOE_e4_Alleles,
        v.systolic_bp AS Systolic_BP, v.cholesterol AS Cholesterol,
        v.blood_glucose AS Blood_Glucose, v.detailed_diagnosis AS Detailed_Diagnosis,
        v.alzheimer_risk_class AS Alzheimer_Risk_Class
    FROM clinical_visits v
    JOIN patients p ON v.patient_id = p.patient_id
    """
    df = query_dataframe(sql_query, db_path=db_path)
    print(f"Loaded {len(df)} records for {df['Subject_ID'].nunique()} unique patients from SQL.")
    
    # 2. Patient-Level Data Splitting
    print("\n[Step 2/6] Performing Patient-Level Split (80% train, 20% holdout test)...")
    train_df, test_df = patient_level_split(df, patient_id_col="Subject_ID", test_size=0.20, random_state=42)
    print(f"Training set: {len(train_df)} visits ({train_df['Subject_ID'].nunique()} patients)")
    print(f"Holdout test set: {len(test_df)} visits ({test_df['Subject_ID'].nunique()} patients)")
    print(f"Train Class distribution: {train_df['Alzheimer_Risk_Class'].value_counts().to_dict()}")
    print(f"Test Class distribution: {test_df['Alzheimer_Risk_Class'].value_counts().to_dict()}")
    
    # 3. Fitting Preprocessing Pipeline (Strict Featurization Ordering)
    print("\n[Step 3/6] Fitting clinical preprocessing & imputation pipeline on train data only...")
    pipeline = ClinicalPreprocessingPipeline(feature_cols=CLINICAL_FEATURE_COLS)
    X_train_scaled, X_train_raw = pipeline.fit_transform(train_df)
    X_test_scaled, X_test_raw = pipeline.transform(test_df)
    
    y_train = train_df["Alzheimer_Risk_Class"].values
    y_test = test_df["Alzheimer_Risk_Class"].values
    
    # Save preprocessing pipeline
    pipeline_path = os.path.join(models_dir, "scaler_pipeline.joblib")
    joblib.dump(pipeline, pipeline_path)
    print(f"Preprocessing pipeline saved to {pipeline_path}")
    
    # Calculate baseline cohort means for patient explainability
    baseline_means = X_train_raw.mean().to_dict()
    cohort_stats_path = os.path.join(models_dir, "cohort_baseline_stats.json")
    with open(cohort_stats_path, "w") as f:
        json.dump(baseline_means, f, indent=2)
        
    # 4. 5-Fold Stratified Cross-Validation
    print("\n[Step 4/6] Executing 5-Fold Stratified Cross-Validation across all 3 architectures...")
    cv_results = run_cross_validation(X_train_scaled, y_train, n_splits=5, random_state=42)
    for model_name, metrics in cv_results.items():
        print(f"\n--- {model_name} (5-Fold CV) ---")
        print(f"  Accuracy:  {metrics['accuracy_mean']:.4f} (+/- {metrics['accuracy_std']:.4f})")
        print(f"  Precision: {metrics['precision_mean']:.4f}")
        print(f"  Recall:    {metrics['recall_mean']:.4f}")
        print(f"  F1-Score:  {metrics['f1_mean']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc_mean']:.4f} (+/- {metrics['roc_auc_std']:.4f})")
        
    # 5. Training, Probability Calibration & Holdout Test Set Evaluation
    print("\n[Step 5/6] Training calibrated models and evaluating on holdout test set...")
    trained_models, test_evaluations, curves = train_and_calibrate_models(
        X_train_scaled, y_train, X_test_scaled, y_test, models_dir
    )
    
    print("\n" + "=" * 65)
    print("FINAL HOLDOUT TEST EVALUATION RESULTS:")
    print("=" * 65)
    for model_name, metrics in test_evaluations.items():
        print(f"\n>>> {model_name}:")
        print(f"    Accuracy:    {metrics['accuracy'] * 100:.2f}%")
        print(f"    Sensitivity: {metrics['sensitivity'] * 100:.2f}% (Recall)")
        print(f"    Specificity: {metrics['specificity'] * 100:.2f}%")
        print(f"    Precision:   {metrics['precision'] * 100:.2f}%")
        print(f"    F1-Score:    {metrics['f1_score']:.4f}")
        print(f"    ROC-AUC:     {metrics['roc_auc']:.4f}")
        cm = metrics["confusion_matrix"]
        print(f"    Confusion Matrix: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")
        
    # 6. Global Feature Importance Extraction
    print("\n[Step 6/6] Computing Explainable AI feature importance metrics...")
    feature_importances = {}
    for name, model in trained_models.items():
        df_imp = get_global_feature_importance(model, CLINICAL_FEATURE_COLS)
        feature_importances[name] = df_imp.to_dict(orient="records")
        
    # Save combined report
    full_report = {
        "test_evaluations": test_evaluations,
        "cv_results": cv_results,
        "feature_importances": feature_importances,
        "curves": curves,
        "dataset_meta": {
            "total_visits": len(df),
            "unique_patients": int(df["Subject_ID"].nunique()),
            "train_size": len(train_df),
            "test_size": len(test_df),
            "feature_count": len(CLINICAL_FEATURE_COLS)
        }
    }
    
    report_file = os.path.join(models_dir, "benchmark_report.json")
    with open(report_file, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nComplete Benchmark and Validation Report saved to {report_file}")
    print("\nSUCCESS: All models trained, calibrated, and ready for deployment in Streamlit dashboard!")

if __name__ == "__main__":
    main()
