"""
NeuroPredictAI - High-Performance Web Server & REST API
Serves modern HTML/CSS/JavaScript frontend and exposes clinical ML inference and SQLite APIs.
Uses Python's standard library ThreadingHTTPServer for zero-dependency, ultra-fast async execution.
"""

from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import json
import os
import urllib.parse
import joblib
import pandas as pd
import numpy as np

# Internal modules
from src.db import (
    get_connection, query_dataframe, execute_custom_sql,
    log_prediction, get_recent_audits, DEFAULT_DB_PATH
)
from src.preprocessing import CLINICAL_FEATURE_COLS
from src.explainability import (
    get_global_feature_importance,
    explain_patient_prediction,
    generate_clinical_recommendations,
    FEATURE_DOMAINS,
    FEATURE_DESCRIPTIONS
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PORT = 8000

# Cache loaded models and artifacts
class ModelRegistry:
    pipeline = None
    models = {}
    benchmark_report = None
    baseline_stats = {}

    @classmethod
    def load(cls):
        pipeline_path = os.path.join(MODELS_DIR, "scaler_pipeline.joblib")
        if os.path.exists(pipeline_path):
            cls.pipeline = joblib.load(pipeline_path)
            
        xgb_path = os.path.join(MODELS_DIR, "xgboost_calibrated.joblib")
        rf_path = os.path.join(MODELS_DIR, "random_calibrated.joblib")
        lr_path = os.path.join(MODELS_DIR, "logistic_calibrated.joblib")
        
        if os.path.exists(xgb_path):
            cls.models["XGBoost (High Performance)"] = joblib.load(xgb_path)
        if os.path.exists(rf_path):
            cls.models["Random Forest (Ensemble)"] = joblib.load(rf_path)
        if os.path.exists(lr_path):
            cls.models["Logistic Regression (Baseline)"] = joblib.load(lr_path)
            
        report_path = os.path.join(MODELS_DIR, "benchmark_report.json")
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                cls.benchmark_report = json.load(f)
                
        baseline_path = os.path.join(MODELS_DIR, "cohort_baseline_stats.json")
        if os.path.exists(baseline_path):
            with open(baseline_path, "r") as f:
                cls.baseline_stats = json.load(f)

ModelRegistry.load()

class NeuroPredictRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def _send_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query_params = urllib.parse.parse_qs(parsed.query)

        # 1. API: Overview & Metadata
        if path == "/api/overview":
            try:
                conn = get_connection()
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM clinical_visits")
                total_visits = cur.fetchone()[0]
                cur.execute("SELECT COUNT(DISTINCT patient_id) FROM clinical_visits")
                total_patients = cur.fetchone()[0]
                cur.execute("SELECT detailed_diagnosis, COUNT(*) FROM clinical_visits GROUP BY detailed_diagnosis")
                diag_counts = dict(cur.fetchall())
                conn.close()

                self._send_json({
                    "total_visits": total_visits,
                    "total_patients": total_patients,
                    "diagnoses": diag_counts,
                    "features_count": len(CLINICAL_FEATURE_COLS),
                    "models_loaded": list(ModelRegistry.models.keys()),
                    "team": {
                        "event": "DEXATHON 2026",
                        "team_name": "UNFAZED",
                        "team_id": "DEX26341",
                        "institution": "Sathyabama Institute of Science and Technology"
                    }
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        # 2. API: Cohort Data with filter & pagination
        if path == "/api/cohort":
            try:
                limit = int(query_params.get("limit", [50])[0])
                offset = int(query_params.get("offset", [0])[0])
                diag = query_params.get("diagnosis", [None])[0]

                where_clause = ""
                params = []
                if diag and diag != "all":
                    where_clause = "WHERE v.detailed_diagnosis = ?"
                    params.append(diag)

                sql = f"""
                SELECT 
                    v.visit_id, v.patient_id, v.visit_number,
                    v.age, p.gender, p.education_years,
                    v.ses, v.mmse, v.cdr, v.moca, v.faq,
                    v.etiv, v.nwbv, v.asf, v.apoe_e4_alleles,
                    v.systolic_bp, v.cholesterol, v.blood_glucose,
                    v.detailed_diagnosis, v.alzheimer_risk_class
                FROM clinical_visits v
                JOIN patients p ON v.patient_id = p.patient_id
                {where_clause}
                ORDER BY v.visit_id ASC
                LIMIT ? OFFSET ?
                """
                params.extend([limit, offset])
                df = query_dataframe(sql, params=params)
                self._send_json({
                    "records": df.to_dict(orient="records"),
                    "count": len(df),
                    "limit": limit,
                    "offset": offset
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        # 3. API: Benchmark Metrics, Curves & 5-Fold CV
        if path == "/api/benchmarks":
            if ModelRegistry.benchmark_report:
                self._send_json(ModelRegistry.benchmark_report)
            else:
                self._send_json({"error": "Benchmark report not generated yet"}, status=404)
            return

        # 4. API: Explainable AI Global Importances
        if path == "/api/xai":
            try:
                model_name = query_params.get("model", ["XGBoost (High Performance)"])[0]
                model = ModelRegistry.models.get(model_name)
                if not model:
                    model = list(ModelRegistry.models.values())[0] if ModelRegistry.models else None

                if model:
                    df_imp = get_global_feature_importance(model, CLINICAL_FEATURE_COLS)
                    self._send_json({
                        "model": model_name,
                        "importances": df_imp.to_dict(orient="records"),
                        "domains": FEATURE_DOMAINS,
                        "descriptions": FEATURE_DESCRIPTIONS
                    })
                else:
                    self._send_json({"error": "Model not available"}, status=404)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        # 5. API: Recent Prediction Audits
        if path == "/api/audits":
            try:
                df_audits = get_recent_audits(limit=15)
                self._send_json({"audits": df_audits.to_dict(orient="records")})
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        # Fallback to serving static files from FRONTEND_DIR
        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Read JSON body
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            payload = {}

        # 1. API: Custom SQL Execution
        if path == "/api/sql":
            query = payload.get("query", "").strip()
            if not query:
                self._send_json({"error": "Query cannot be empty"}, status=400)
                return
            success, res_df, msg = execute_custom_sql(query)
            if success and res_df is not None:
                self._send_json({
                    "success": True,
                    "columns": list(res_df.columns),
                    "rows": res_df.to_dict(orient="records"),
                    "row_count": len(res_df),
                    "message": msg
                })
            elif success:
                self._send_json({"success": True, "message": msg, "row_count": 0, "rows": []})
            else:
                self._send_json({"success": False, "error": msg}, status=400)
            return

        # 2. API: Clinical Risk Prediction
        if path == "/api/predict":
            try:
                model_name = payload.get("model", "XGBoost (High Performance)")
                patient_data = payload.get("patient", {})
                notes = payload.get("notes", "Routine evaluation")

                model = ModelRegistry.models.get(model_name)
                pipeline = ModelRegistry.pipeline

                if not model or not pipeline:
                    self._send_json({"error": "Model or pipeline not loaded"}, status=500)
                    return

                # Ensure required fields exist
                patient_row = {
                    "Subject_ID": patient_data.get("patient_id", "PT-ANON"),
                    "Visit_Number": 1,
                    "Age": float(patient_data.get("age", 72.0)),
                    "Gender": str(patient_data.get("gender", "F")),
                    "Education_Years": int(patient_data.get("education_years", 14)),
                    "Socioeconomic_Status": float(patient_data.get("ses", 2.0)),
                    "MMSE": float(patient_data.get("mmse", 24.0)),
                    "CDR": float(patient_data.get("cdr", 0.5)),
                    "MoCA": float(patient_data.get("moca", 22.0)),
                    "FAQ": float(patient_data.get("faq", 4.0)),
                    "eTIV": float(patient_data.get("etiv", 1480.0)),
                    "nWBV": float(patient_data.get("nwbv", 0.73)),
                    "ASF": round(1948.0 / float(patient_data.get("etiv", 1480.0)), 3),
                    "APOE_e4_Alleles": int(patient_data.get("apoe_e4_alleles", 0)),
                    "Systolic_BP": float(patient_data.get("systolic_bp", 130.0)),
                    "Cholesterol": float(patient_data.get("cholesterol", 205.0)),
                    "Blood_Glucose": float(patient_data.get("blood_glucose", 100.0))
                }

                df_patient = pd.DataFrame([patient_row])
                X_scaled, df_eng = pipeline.transform(df_patient)
                
                # Calibrated inference
                risk_prob = float(model.predict_proba(X_scaled)[0, 1])
                pred_class = int(model.predict(X_scaled)[0])

                # Clinical recommendations
                rec_info = generate_clinical_recommendations(risk_prob, df_eng.iloc[0].to_dict())

                # Local Explainable AI attribution
                df_imp_current = get_global_feature_importance(model, CLINICAL_FEATURE_COLS)
                patient_attr = explain_patient_prediction(
                    df_eng.iloc[0].to_dict(),
                    ModelRegistry.baseline_stats,
                    df_imp_current
                )

                # Log to SQLite
                log_prediction(
                    patient_id=patient_row["Subject_ID"],
                    model_name=model_name,
                    risk_prob=risk_prob,
                    pred_class=pred_class,
                    risk_tier=rec_info["tier"],
                    input_summary=f"Age:{patient_row['Age']}|MMSE:{patient_row['MMSE']}|CDR:{patient_row['CDR']}|nWBV:{patient_row['nWBV']}|APOE:{patient_row['APOE_e4_Alleles']}",
                    notes=notes
                )

                self._send_json({
                    "patient_id": patient_row["Subject_ID"],
                    "model_used": model_name,
                    "risk_probability": round(risk_prob, 4),
                    "risk_percentage": round(risk_prob * 100, 2),
                    "predicted_class": pred_class,
                    "tier_info": rec_info,
                    "local_attributions": patient_attr.to_dict(orient="records"),
                    "engineered_features": df_eng.iloc[0].to_dict()
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        self._send_json({"error": "Endpoint not found"}, status=404)

def run_server(port=PORT):
    server = ThreadingHTTPServer(("0.0.0.0", port), NeuroPredictRequestHandler)
    print(f"============================================================")
    print(f"Server URL: http://localhost:{port}")
    print(f"Frontend Root: {FRONTEND_DIR}")
    print(f"============================================================")
    server.serve_forever()

if __name__ == "__main__":
    run_server(PORT)
