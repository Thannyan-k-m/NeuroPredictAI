# NeuroPredict AI — Clinical Healthcare Analytics & ML Platform
**Dexathon 2026 | Team UNFAZED | Team ID DEX26341**

NeuroPredict AI is an AI/ML healthcare analytics platform designed to identify Alzheimer's-associated risk patterns using patient demographic information, cognitive assessment scores, and clinical/neuroimaging measurements.

> **Healthcare Safety Notice:**
> NeuroPredict AI provides preliminary research predictions and does not replace professional medical evaluation. This is a research prototype, not a medical diagnostic system.

---

## Technology Stack
- **Frontend**: HTML5, Vanilla CSS3, Vanilla JavaScript (ES6+), Chart.js, Fetch API
- **Backend**: Python, FastAPI, Uvicorn, Pydantic
- **Machine Learning**: Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Joblib
- **Database**: SQLite (parameterized SQL queries, dataset registry, model audits, prediction history)

---

## Project Structure
```text
NeuroPredictAI/
├── frontend/
│   ├── index.html
│   ├── pages/
│   │   ├── dashboard.html
│   │   ├── patient-analysis.html
│   │   ├── dataset-analysis.html
│   │   ├── model-insights.html
│   │   └── research.html
│   ├── css/
│   │   ├── style.css
│   │   ├── dashboard.css
│   │   ├── patient-analysis.css
│   │   ├── dataset-analysis.css
│   │   └── model-insights.css
│   ├── js/
│   │   ├── api.js
│   │   ├── dashboard.js
│   │   ├── patient-analysis.js
│   │   ├── dataset-analysis.js
│   │   ├── model-insights.js
│   │   └── research.js
│   └── assets/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── src/
│   │   ├── data_loader.py
│   │   ├── preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── train.py
│   │   ├── evaluation.py
│   │   ├── prediction.py
│   │   ├── explainability.py
│   │   └── database.py
│   ├── models/
│   ├── data/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── sample/
│   └── tests/
│       ├── test_preprocessing.py
│       ├── test_prediction.py
│       └── test_database.py
├── README.md
├── .gitignore
└── .env.example
```

---

## Local Run Instructions

### 1. Start the FastAPI Backend
Open a terminal in the project directory:
```bash
cd backend
# Activate your virtual environment (e.g. on Windows PowerShell)
..\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt

# Run FastAPI with Uvicorn
uvicorn main:app --reload --port 8000
```
- **Backend API URL**: `http://127.0.0.1:8000`
- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`

### 2. Start the Frontend Server
Open a second terminal:
```bash
cd frontend
python -m http.server 5500
```
- **Frontend App URL**: `http://127.0.0.1:5500`
- **Dashboard**: `http://127.0.0.1:5500/pages/dashboard.html`

---

## Running Test Suite
Execute unit tests covering database CRUD, leakage-free preprocessing, feature engineering, model training, and patient prediction:
```bash
pytest backend/tests/ -v
```
