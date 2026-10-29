"""
NeuroPredictAI - Explainable AI (XAI) & Interpretability Engine
Provides global feature importance and patient-level local attribution breakdowns,
empowering clinicians with transparent, interpretable reasoning behind risk scores.
"""

import numpy as np
import pandas as pd

FEATURE_DESCRIPTIONS = {
    "Age": "Patient Chronological Age (years)",
    "Gender_Encoded": "Biological Sex Indicator",
    "Education_Years": "Formal Education Duration (Cognitive Reserve)",
    "Socioeconomic_Status": "Socioeconomic Stratification Index",
    "MMSE": "Mini-Mental State Examination (0-30)",
    "CDR": "Clinical Dementia Rating Global Score",
    "MoCA": "Montreal Cognitive Assessment Score",
    "FAQ": "Functional Activities Questionnaire Score",
    "eTIV": "Estimated Total Intracranial Volume (mm³)",
    "nWBV": "Normalized Whole Brain Volume (Atrophy Marker)",
    "ASF": "Atlas Scaling Factor",
    "APOE_e4_Alleles": "APOE ε4 Genetic Allele Count",
    "Systolic_BP": "Systolic Blood Pressure (Vascular Health)",
    "Cholesterol": "Total Serum Cholesterol Level",
    "Blood_Glucose": "Fasting Plasma Glucose (Metabolic)",
    "Brain_Atrophy_Index": "Derived Brain Parenchymal Atrophy (1 - nWBV)",
    "Cognitive_Decline_Ratio": "Proportional Cognitive Loss on MMSE",
    "Cardiovascular_Risk_Index": "Multi-System Vascular Risk Ratio",
    "Age_Vulnerability_Score": "Standardized Age-Associated Risk Exposure"
}

FEATURE_DOMAINS = {
    "MMSE": "🧠 Cognitive Function",
    "CDR": "🧠 Cognitive Function",
    "MoCA": "🧠 Cognitive Function",
    "FAQ": "🧠 Cognitive Function",
    "Cognitive_Decline_Ratio": "🧠 Cognitive Function",
    "eTIV": "🩻 Neuroimaging Biomarkers",
    "nWBV": "🩻 Neuroimaging Biomarkers",
    "ASF": "🩻 Neuroimaging Biomarkers",
    "Brain_Atrophy_Index": "🩻 Neuroimaging Biomarkers",
    "APOE_e4_Alleles": "🧬 Genetics & Demographics",
    "Age": "🧬 Genetics & Demographics",
    "Gender_Encoded": "🧬 Genetics & Demographics",
    "Education_Years": "🧬 Genetics & Demographics",
    "Socioeconomic_Status": "🧬 Genetics & Demographics",
    "Age_Vulnerability_Score": "🧬 Genetics & Demographics",
    "Systolic_BP": "🫀 Vascular & Metabolic",
    "Cholesterol": "🫀 Vascular & Metabolic",
    "Blood_Glucose": "🫀 Vascular & Metabolic",
    "Cardiovascular_Risk_Index": "🫀 Vascular & Metabolic"
}

def get_global_feature_importance(model, feature_names):
    """
    Extracts normalized feature importance weights from trained models.
    Supports CalibratedClassifierCV wrappers, XGBoost, and Random Forest.
    """
    estimator = model
    if hasattr(model, "calibrated_classifiers_"):
        # Extract underlying base estimator from first calibrated fold
        estimator = model.calibrated_classifiers_[0].estimator
    elif hasattr(model, "estimator"):
        estimator = model.estimator
        
    importances = None
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        # Absolute standardized coefficients for linear models
        importances = np.abs(estimator.coef_[0])
    else:
        # Fallback equal weights
        importances = np.ones(len(feature_names)) / len(feature_names)
        
    # Normalize to 100%
    norm_importances = (importances / np.sum(importances)) * 100.0
    
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance_Percent": np.round(norm_importances, 2),
        "Domain": [FEATURE_DOMAINS.get(f, "General") for f in feature_names],
        "Description": [FEATURE_DESCRIPTIONS.get(f, f) for f in feature_names]
    }).sort_values(by="Importance_Percent", ascending=False).reset_index(drop=True)
    
    return df_imp

def explain_patient_prediction(patient_features_dict, baseline_means_dict, feature_importances_df):
    """
    Generates a localized, patient-specific risk attribution waterfall.
    Calculates positive risk drivers (elevating Alzheimer risk) and protective factors.
    """
    attribution_records = []
    imp_map = dict(zip(feature_importances_df["Feature"], feature_importances_df["Importance_Percent"]))
    
    for feat, val in patient_features_dict.items():
        if feat not in baseline_means_dict or feat not in imp_map:
            continue
            
        base_val = baseline_means_dict[feat]
        importance = imp_map[feat]
        
        # Clinical directionality:
        # For MMSE, MoCA, nWBV, Education: lower is higher risk -> delta = base - val
        # For CDR, FAQ, Age, APOE, Atrophy, Vascular: higher is higher risk -> delta = val - base
        inverse_features = ["MMSE", "MoCA", "nWBV", "Education_Years"]
        if feat in inverse_features:
            deviation = (base_val - val) / (abs(base_val) + 1e-5)
        else:
            deviation = (val - base_val) / (abs(base_val) + 1e-5)
            
        attribution_score = round(float(deviation * (importance / 10.0)), 2)
        direction = "Risk Elevating" if attribution_score > 0 else "Protective / Normal"
        
        attribution_records.append({
            "Feature": feat,
            "Domain": FEATURE_DOMAINS.get(feat, "General"),
            "Patient_Value": round(float(val), 2),
            "Cohort_Mean": round(float(base_val), 2),
            "Attribution_Score": attribution_score,
            "Clinical_Impact": direction,
            "Description": FEATURE_DESCRIPTIONS.get(feat, feat)
        })
        
    df_attr = pd.DataFrame(attribution_records).sort_values(by="Attribution_Score", key=abs, ascending=False).reset_index(drop=True)
    return df_attr

def generate_clinical_recommendations(risk_prob, patient_features):
    """Generates tailored clinician recommendations based on calculated risk and markers."""
    recommendations = []
    
    if risk_prob >= 0.65:
        tier = "High Risk / Probable Dementia Pathway"
        color = "#e63946"
        badge = "HIGH RISK"
        recommendations.append("Immediate comprehensive neuropsychological battery referral (ADAS-Cog, WAIS-IV).")
        recommendations.append("Order high-resolution volumetric 3T Brain MRI to assess hippocampal volume and temporal horn enlargement.")
        recommendations.append("Consider CSF biomarker analysis (Aβ42/Aβ40 ratio, p-tau181) or Amyloid PET scan.")
        recommendations.append("Initiate caregiver counseling, advance care planning, and safety assessment.")
    elif risk_prob >= 0.35:
        tier = "Moderate Risk / Mild Cognitive Impairment (MCI) Monitoring"
        color = "#f4a261"
        badge = "MODERATE RISK"
        recommendations.append("Schedule 6-month longitudinal cognitive reassessment tracking MMSE and MoCA trajectory.")
        recommendations.append("Target modifiable vascular risk factors: optimize blood pressure (<130/80) and manage fasting glucose.")
        recommendations.append("Prescribe structured aerobic physical activity (150 min/week) and Mediterranean-DASH Intervention for Neurodegenerative Delay (MIND diet).")
        recommendations.append("Review medications for anticholinergic cognitive burden.")
    else:
        tier = "Low Risk / Cognitively Stable"
        color = "#2a9d8f"
        badge = "LOW RISK"
        recommendations.append("Routine annual cognitive wellness evaluation recommended.")
        recommendations.append("Maintain active cognitive engagement, social interaction, and restorative sleep hygiene.")
        recommendations.append("Continue standard preventive health checkups.")
        
    # Specific targeted biomarker alerts
    if patient_features.get("APOE_e4_Alleles", 0) >= 1:
        recommendations.append("Genetic Advisory: Patient carries APOE ε4 allele(s); heightened surveillance indicated.")
    if patient_features.get("nWBV", 1.0) < 0.72:
        recommendations.append("Structural Alert: Normalized Whole Brain Volume indicates accelerated parenchymal loss.")
    if patient_features.get("Systolic_BP", 120) >= 140:
        recommendations.append("Vascular Alert: Elevated systolic blood pressure detected; aggressive vascular management recommended.")

    return {
        "tier": tier,
        "badge": badge,
        "color": color,
        "recommendations": recommendations
    }
