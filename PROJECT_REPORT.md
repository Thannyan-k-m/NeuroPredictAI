# NeuroPredict AI — Comprehensive Project Report
**Dexathon 2026 | Team UNFAZED | Team ID: DEX26341**

---

## 1. Executive Summary

**NeuroPredict AI** is a clinical healthcare analytics and machine learning platform engineered to identify Alzheimer's disease risk patterns using demographic markers, cognitive test scores, and neuroimaging assessments.

Dementia and Alzheimer’s disease affect over 55 million individuals globally, with projections tripling by 2050. Conventional diagnostic workflows frequently rely on invasive lumbar punctures, expensive PET scans, or subjective behavioral checklists that occur late in disease progression. 

NeuroPredict AI bridges the gap between raw multi-modal clinical registries and actionable diagnostic support. By utilizing harmonized clinical biomarkers (derived from research cohorts such as ADNI and OASIS), the system provides:
1. **Multi-Model Predictive Analytics:** Calibrated models including Logistic Regression, Random Forest, and XGBoost classifiers.
2. **Transparent Explainability:** Feature importance and patient-level TreeSHAP / KernelSHAP force values that explain the precise clinical reasoning behind every prediction.
3. **Clinical Safety & Guardrails:** Uncertainty flags, confidence scoring, audit-logged prediction persistence, and strict disclaimers that keep clinicians in the loop.

---

## 2. System Architecture & Tech Stack

The architecture follows a decoupled client-server design with asynchronous REST APIs and an embedded, ACID-compliant database layer.

```
+---------------------------------------------------------------------------------+
|                                Frontend (Client)                                |
|  - HTML5, Vanilla Modern CSS3 (Glassmorphism & Medical Dark Theme), ES6+ JS     |
|  - Chart.js (ROC Curves, PR Curves, Biomarker Distributions, Radar Charts)      |
|  - Pages: Executive Dashboard, Patient Analysis, Dataset Explorer,              |
|           Model Insights & Benchmarks, Clinical Research Hub                    |
+---------------------------------------+-----------------------------------------+
                                        | HTTP / JSON (Fetch API)
                                        v
+---------------------------------------------------------------------------------+
|                            Backend Application Tier                             |
|  - FastAPI (Python 3.10+)                                                       |
|  - Uvicorn ASGI Server                                                          |
|  - Pydantic Data Validation & Type Safety                                       |
|  - Endpoints: /api/predict, /api/models, /api/analytics, /api/history           |
+-------------------+---------------------------------------+---------------------+
                    |                                       |
                    v                                       v
+------------------------------------+   +----------------------------------------+
|          ML Pipeline Tier          |   |          Data & Storage Tier           |
|  - Scikit-Learn (Pipelines, Stacking)   |  - SQLite3 (Parameterized, WAL Mode)   |
|  - XGBoost (Gradient Boosted Trees)|   |    * Datasets Registry                 |
|  - SHAP (TreeExplainer & Kernel)   |   |    * Models Registry & Benchmarks      |
|  - Joblib (Model Serialization)    |   |    * Patient Prediction Audit Logs     |
|  - CalibratedClassifierCV          |   |  - ADNI / OASIS Harmonized Datasets    |
+------------------------------------+   +----------------------------------------+
```

### Technology Breakdown
- **Frontend Layer:** Semantic HTML5, Modular CSS3 variables, Vanilla JavaScript (ES6 Modules), Chart.js 4.x.
- **Backend Service Layer:** Python 3.10+, FastAPI framework, Uvicorn ASGI, Pydantic schemas.
- **Machine Learning & Analytics:** Scikit-learn, XGBoost, SHAP (SHapley Additive exPlanations), NumPy, Pandas, Joblib.
- **Database & Audit Storage:** SQLite with parameterized SQL queries for persistent audit trails and history.

---

## 3. Dataset Characteristics & Feature Engineering

### 3.1 Clinical Cohort Basis
The platform is calibrated against harmonized longitudinal metrics inspired by the **Alzheimer’s Disease Neuroimaging Initiative (ADNI)** and **Open Access Series of Imaging Studies (OASIS)** cohorts.

| Feature Category | Features Included | Clinical Significance |
| :--- | :--- | :--- |
| **Demographics** | `Age`, `Gender`, `Education_Years` | Baseline neurological risk profiles and cognitive reserve adjustments. |
| **Cognitive Batteries** | `MMSE` (Mini-Mental State Exam), `CDR` (Clinical Dementia Rating), `ADAS13` (Cognitive Subscale) | Core neuropsychological indicators of memory loss and functional impairment. |
| **Volumetric Neuroimaging**| `eTIV` (Estimated Total Intracranial Volume), `nWBV` (Normalized Whole Brain Volume), `ASF` (Atlas Scaling Factor), `Hippocampus_Volume` | Structural magnetic resonance biomarkers reflecting cerebral atrophy. |
| **Functional & Biomarkers**| `FAQ` (Functional Activities Questionnaire), `APOE4` Allele count | Genotypic susceptibility and activities-of-daily-living impairment metrics. |

### 3.2 Feature Engineering Pipeline
The pipeline generates non-linear and interaction features prior to model ingestion:
- **Atrophy Ratios:** $\text{Atrophy\_Ratio} = \frac{\text{Hippocampus\_Volume}}{\text{eTIV}}$
- **Cognitive-Functional Interaction Index:** $\text{Impairment\_Index} = \text{CDR} \times \text{FAQ}$
- **Cognitive Decline Score:** Harmonized composite weighting between low MMSE ($30 - \text{MMSE}$) and elevated ADAS13.
- **Leakage Prevention:** Standard scaling and imputation parameters are fitted strictly on the cross-validation training folds and persisted via Scikit-Learn Pipelines.

---

## 4. Machine Learning Models & Benchmark Results

Three distinct algorithmic paradigms were trained, validated, and calibrated using stratified cross-validation on held-out test splits.

### 4.1 Evaluation Matrix

| Metric | Logistic Regression (Baseline) | Random Forest (Ensemble) | XGBoost (High Performance) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 94.42% | 91.24% | **93.23%** |
| **Sensitivity (Recall)** | 95.98% | 94.25% | **94.83%** |
| **Specificity** | 90.91% | 84.42% | **89.61%** |
| **Precision** | 95.98% | 93.18% | **95.38%** |
| **F1-Score** | 95.98% | 93.71% | **95.10%** |
| **ROC-AUC** | **0.9860** | 0.9776 | **0.9804** |
| **Confusion Matrix (TP / TN / FP / FN)** | 167 / 70 / 7 / 7 | 164 / 65 / 12 / 10 | 165 / 69 / 8 / 9 |

### 4.2 Key Insights
1. **High Clinical Sensitivity:** All models maintain sensitivity above 94%, ensuring minimal false negatives for patients requiring clinical follow-up.
2. **Probability Calibration:** Using `CalibratedClassifierCV` (isotonic & sigmoid methods), predicted output probabilities represent true empirical risk likelihoods rather than arbitrary decision margins.
3. **Overfitting Controls:** Regularization ($L_1/L_2$ for Logistic Regression, tree depth and sub-sampling limits for XGBoost) was tuned to guarantee generalizability across variable demographic distributions.

---

## 5. Model Explainability & Interpretability (SHAP)

To overcome the "black box" limitation in healthcare AI, NeuroPredict AI incorporates **SHAP (SHapley Additive exPlanations)** grounded in cooperative game theory.

### 5.1 Global Interpretability
Across the dataset, the top predictive factors driving model outputs are:
1. **`CDR` (Clinical Dementia Rating):** Primary driver separating normal cognition from mild cognitive impairment (MCI) and dementia.
2. **`MMSE`:** Negative correlation — lower cognitive scores strongly drive positive risk attribution.
3. **`Hippocampus_Volume` & `nWBV`:** Structural volume reductions strongly increase risk weighting.
4. **`FAQ` & `APOE4`:** High functional impairment and presence of one or two $\epsilon4$ alleles add significant marginal risk.

### 5.2 Local (Patient-Level) Explanation
For individual patient assessments, the system generates localized waterfall/force values:
- Shows the baseline base-rate probability across the population ($E[f(x)]$).
- Graphically displays each individual biomarker pushing the score toward higher risk (+SHAP) or lower risk (-SHAP).
- Produces clinical guidance notes based on the dominant contributing features.

---

## 6. Frontend User Experience & Modules

The frontend interface is organized into five dedicated modules:

1. **Executive Dashboard (`dashboard.html`):**
   - High-level KPI summary cards (Total Screenings, Risk Stratification breakdown, High-Risk Alerts).
   - Real-time recent patient screening history table with status badges.
   - Comparative ROC curve previews and biomarker correlation charts.

2. **Patient Risk Assessment (`patient-analysis.html`):**
   - Interactive clinical data entry form with immediate input bounds validation.
   - Model selector (XGBoost, Random Forest, Logistic Regression).
   - Comprehensive risk output: Risk Category (Low, Moderate, High), Calibrated Probability Gauge, Confidence Interval, SHAP Local Feature Waterfall, and Recommended Clinical Follow-up Actions.
   - Built-in CSV / PDF summary report export.

3. **Dataset Explorer (`dataset-analysis.html`):**
   - Dataset upload and validation (schema checking, missing value audits).
   - Interactive column distributions, correlation matrices, and statistical dispersion graphs.

4. **Model Insights & Benchmarking (`model-insights.html`):**
   - Side-by-side performance comparison tables.
   - Interactive ROC & Precision-Recall curves plotted with Chart.js.
   - Global feature importance rankings and confusion matrix visualizers.

5. **Clinical Research Hub (`research.html`):**
   - Literature references, biomarker documentation, and safety notes explaining the translational path from research benchmarks to clinical deployment.

---

## 7. Healthcare Safety, Ethics & Privacy

NeuroPredict AI is designed under strict ethical healthcare AI frameworks:

- **Human-in-the-Loop:** Predictions are clearly labeled as decision-support insights and do not provide automated or definitive diagnoses.
- **Uncertainty Disclosures:** Predictions falling within indeterminate probability margins (e.g., 40%–60%) explicitly flag "Moderate / Borderline Risk — Inconclusive screening, formal specialist assessment recommended."
- **Data Protection & De-identification:** The schema stores anonymized patient reference identifiers (`patient_ref`), preventing the storage of personally identifiable information (PII).
- **Auditability:** Every prediction query, input snapshot, model version, and output probability is immutably logged in the database for retrospective clinical audits.

---

## 8. Installation & Verification Guide

### Prerequisites
- Python 3.10+
- Modern Web Browser (Chrome, Firefox, Edge, Safari)
- Git

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/Thannyan-k-m/NeuroPredictAI.git
cd NeuroPredictAI

# 2. Set up Backend
cd backend
python -m venv ..\.venv
..\.venv\Scripts\activate      # On Linux/macOS: source ../.venv/bin/activate
pip install -r requirements.txt

# 3. Start Backend Server
uvicorn main:app --reload --port 8000
```

### Running Frontend
```bash
# In a separate terminal:
cd frontend
python -m http.server 5500
```
- Open `http://127.0.0.1:5500` in your browser.
- Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### Running Verification Tests
```bash
pytest backend/tests/ -v
```

---

## 9. Future Roadmap

1. **DICOM / NIfTI Volumetric Ingestion:** Direct segmentation pipeline using 3D convolutional neural networks (e.g., 3D ResNet / nnU-Net) on raw MRI T1-weighted scans.
2. **Multi-Modal Foundation Model Integration:** Incorporating clinical visit text notes via medical LLMs for holistic patient summaries.
3. **FHIR / HL7 Interoperability:** Standardized integration with Hospital Information Systems (HIS) and Electronic Health Record (EHR) systems.
4. **Longitudinal Trajectory Forecasting:** Time-series recurrence models predicting rate-of-conversion from Mild Cognitive Impairment (MCI) to Alzheimer's Disease over 24- and 48-month windows.
