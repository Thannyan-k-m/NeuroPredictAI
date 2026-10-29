"""
NeuroPredictAI - ADNI & OASIS Harmonized Clinical Dataset Generator
Simulates realistic longitudinal clinical research data inspired by the
Alzheimer's Disease Neuroimaging Initiative (ADNI) and Open Access Series of Imaging Studies (OASIS).
"""

import numpy as np
import pandas as pd
import os

def generate_cohort(n_subjects=650, seed=42):
    np.random.seed(seed)
    records = []
    
    for sub_id in range(1001, 1001 + n_subjects):
        subject_code = f"SUB-{sub_id}"
        gender = np.random.choice(["F", "M"], p=[0.54, 0.46])
        education = int(np.clip(np.random.normal(14.5, 3.2), 6, 22))
        ses = np.random.choice([1, 2, 3, 4, 5], p=[0.20, 0.30, 0.25, 0.15, 0.10])
        
        # Genetic risk factor: APOE-e4 allele count (0, 1, or 2)
        apoe4 = np.random.choice([0, 1, 2], p=[0.65, 0.28, 0.07])
        
        # Base baseline age
        baseline_age = np.random.uniform(62, 86)
        
        # Underlying latent cognitive decline liability (genetic + age + vascular risk)
        base_liability = (
            (baseline_age - 70) * 0.06
            + (apoe4 * 0.95)
            - ((education - 12) * 0.08)
            + np.random.normal(0, 0.6)
        )
        
        # Number of longitudinal visits: 1 to 3
        n_visits = np.random.choice([1, 2, 3], p=[0.35, 0.45, 0.20])
        
        for visit in range(1, n_visits + 1):
            time_elapsed_years = (visit - 1) * np.random.uniform(1.0, 1.8)
            age = round(baseline_age + time_elapsed_years, 1)
            liability = base_liability + (time_elapsed_years * 0.35) + np.random.normal(0, 0.25)
            
            # Clinical Dementia Rating (CDR)
            if liability < -0.3:
                cdr = 0.0
                diagnosis = "Cognitively Normal"
                target = 0
            elif liability < 0.65:
                cdr = 0.5
                # Transitional MCI
                diagnosis = "Mild Cognitive Impairment"
                target = 1 if liability > 0.3 else 0
            elif liability < 1.4:
                cdr = 1.0
                diagnosis = "Alzheimer's Disease"
                target = 1
            else:
                cdr = 2.0
                diagnosis = "Alzheimer's Disease"
                target = 1
            
            # MMSE (Mini-Mental State Examination, 0-30)
            base_mmse = 29.2 - (liability * 4.8) + np.random.normal(0, 1.3)
            mmse = int(np.clip(round(base_mmse), 10, 30))
            
            # MoCA (Montreal Cognitive Assessment, 0-30)
            base_moca = 27.5 - (liability * 4.5) + np.random.normal(0, 1.5)
            moca = int(np.clip(round(base_moca), 8, 30))
            
            # FAQ (Functional Activities Questionnaire, 0-30; higher = impaired)
            faq = int(np.clip(round(max(0, liability * 6.5 + np.random.normal(0, 2.0))), 0, 30))
            
            # Volumetric Neuroimaging Biomarkers (eTIV, nWBV, ASF from OASIS/ADNI standards)
            # Normalized Whole Brain Volume (nWBV) typically decreases with age and strongly with AD
            nwbv = round(float(np.clip(0.795 - (age - 60) * 0.0028 - (liability * 0.038) + np.random.normal(0, 0.015), 0.63, 0.85)), 4)
            etiv = int(np.clip(np.random.normal(1520 if gender == "M" else 1380, 140), 1100, 2050))
            asf = round(float(np.clip(1948.0 / etiv + np.random.normal(0, 0.03), 0.88, 1.58)), 3)
            
            # Cardiovascular & Metabolic markers
            systolic_bp = int(np.clip(np.random.normal(132 + liability * 3, 14), 105, 185))
            cholesterol = int(np.clip(np.random.normal(210 + liability * 4, 28), 140, 310))
            blood_glucose = int(np.clip(np.random.normal(102 + liability * 5, 18), 72, 195))
            
            records.append({
                "Subject_ID": subject_code,
                "Visit_Number": visit,
                "Age": age,
                "Gender": gender,
                "Education_Years": education,
                "Socioeconomic_Status": ses,
                "MMSE": mmse,
                "CDR": cdr,
                "MoCA": moca,
                "FAQ": faq,
                "eTIV": etiv,
                "nWBV": nwbv,
                "ASF": asf,
                "APOE_e4_Alleles": apoe4,
                "Systolic_BP": systolic_bp,
                "Cholesterol": cholesterol,
                "Blood_Glucose": blood_glucose,
                "Detailed_Diagnosis": diagnosis,
                "Alzheimer_Risk_Class": target
            })
            
    df = pd.DataFrame(records)
    
    # Inject realistic real-world clinical missingness (3% - 6%) in certain fields
    # to evaluate Multiple Imputation as proposed in Slide 4
    mask_ses = np.random.rand(len(df)) < 0.04
    df.loc[mask_ses, "Socioeconomic_Status"] = np.nan
    
    mask_moca = np.random.rand(len(df)) < 0.06
    df.loc[mask_moca, "MoCA"] = np.nan
    
    mask_chol = np.random.rand(len(df)) < 0.04
    df.loc[mask_chol, "Cholesterol"] = np.nan
    
    return df

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df = generate_cohort(n_subjects=650, seed=2026)
    out_csv = os.path.join(out_dir, "adni_oasis_dataset.csv")
    df.to_csv(out_csv, index=False)
    print(f"Generated {len(df)} longitudinal clinical records for {df['Subject_ID'].nunique()} unique patients.")
    print(f"Saved to: {out_csv}")
    print(df["Alzheimer_Risk_Class"].value_counts(normalize=True))
