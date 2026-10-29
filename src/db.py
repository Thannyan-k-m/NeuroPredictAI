"""
NeuroPredictAI - SQL Data Retrieval & Management Layer
Handles database schema creation, seeding from clinical cohort,
and executing clinical SQL queries.
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "neuropredict.sqlite")

def get_connection(db_path=DEFAULT_DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path=DEFAULT_DB_PATH, seed_csv_path=None):
    """Initializes tables and optionally seeds from the ADNI/OASIS CSV dataset."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    
    # 1. Patients Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT PRIMARY KEY,
        gender TEXT NOT NULL,
        education_years INTEGER,
        apoe4_alleles INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 2. Clinical Visits Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS clinical_visits (
        visit_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL,
        visit_number INTEGER NOT NULL,
        age REAL NOT NULL,
        ses REAL,
        mmse REAL,
        cdr REAL,
        moca REAL,
        faq REAL,
        etiv REAL,
        nwbv REAL,
        asf REAL,
        apoe_e4_alleles INTEGER,
        systolic_bp REAL,
        cholesterol REAL,
        blood_glucose REAL,
        detailed_diagnosis TEXT,
        alzheimer_risk_class INTEGER,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
    );
    """)
    
    # 3. Model Prediction Audits
    cur.execute("""
    CREATE TABLE IF NOT EXISTS prediction_audit_log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        model_name TEXT,
        risk_probability REAL,
        predicted_class INTEGER,
        risk_tier TEXT,
        input_summary TEXT,
        clinician_notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    conn.commit()
    
    # Seed data if clinical_visits is empty
    cur.execute("SELECT COUNT(*) FROM clinical_visits")
    count = cur.fetchone()[0]
    
    if count == 0 and seed_csv_path and os.path.exists(seed_csv_path):
        df = pd.read_csv(seed_csv_path)
        
        # Insert unique patients
        unique_pts = df.drop_duplicates(subset=["Subject_ID"])
        for _, row in unique_pts.iterrows():
            cur.execute("""
            INSERT OR IGNORE INTO patients (patient_id, gender, education_years, apoe4_alleles)
            VALUES (?, ?, ?, ?)
            """, (row["Subject_ID"], row["Gender"], int(row["Education_Years"]), int(row["APOE_e4_Alleles"])))
        
        # Insert visits
        for _, row in df.iterrows():
            cur.execute("""
            INSERT INTO clinical_visits (
                patient_id, visit_number, age, ses, mmse, cdr, moca, faq,
                etiv, nwbv, asf, apoe_e4_alleles, systolic_bp, cholesterol,
                blood_glucose, detailed_diagnosis, alzheimer_risk_class
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["Subject_ID"], int(row["Visit_Number"]), float(row["Age"]),
                None if pd.isna(row["Socioeconomic_Status"]) else float(row["Socioeconomic_Status"]),
                None if pd.isna(row["MMSE"]) else float(row["MMSE"]),
                float(row["CDR"]),
                None if pd.isna(row["MoCA"]) else float(row["MoCA"]),
                float(row["FAQ"]),
                float(row["eTIV"]), float(row["nWBV"]), float(row["ASF"]),
                int(row["APOE_e4_Alleles"]),
                float(row["Systolic_BP"]),
                None if pd.isna(row["Cholesterol"]) else float(row["Cholesterol"]),
                float(row["Blood_Glucose"]),
                row["Detailed_Diagnosis"],
                int(row["Alzheimer_Risk_Class"])
            ))
        
        conn.commit()
        print(f"Database successfully seeded with {len(df)} visits across {len(unique_pts)} patients.")
        
    conn.close()

def query_dataframe(query, params=None, db_path=DEFAULT_DB_PATH):
    """Executes a SQL SELECT query and returns a pandas DataFrame."""
    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query(query, conn, params=params)
        return df
    finally:
        conn.close()

def execute_custom_sql(query, db_path=DEFAULT_DB_PATH):
    """Safely executes custom SQL query from the dashboard console."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    try:
        cur.execute(query)
        if cur.description:
            columns = [d[0] for d in cur.description]
            rows = cur.fetchall()
            df = pd.DataFrame(rows, columns=columns)
            return True, df, f"{len(df)} rows returned."
        else:
            conn.commit()
            return True, None, f"Query executed successfully ({cur.rowcount} rows affected)."
    except Exception as e:
        return False, None, str(e)
    finally:
        conn.close()

def log_prediction(patient_id, model_name, risk_prob, pred_class, risk_tier, input_summary, notes="", db_path=DEFAULT_DB_PATH):
    """Records a clinical risk prediction into the audit log."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO prediction_audit_log (
        patient_id, model_name, risk_probability, predicted_class, risk_tier, input_summary, clinician_notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (patient_id, model_name, float(risk_prob), int(pred_class), risk_tier, input_summary, notes))
    conn.commit()
    conn.close()

def get_recent_audits(limit=20, db_path=DEFAULT_DB_PATH):
    """Retrieves recent prediction audit records."""
    query = "SELECT * FROM prediction_audit_log ORDER BY created_at DESC LIMIT ?"
    return query_dataframe(query, params=(limit,), db_path=db_path)

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "adni_oasis_dataset.csv")
    init_db(seed_csv_path=csv_path)
    df_check = query_dataframe("SELECT COUNT(*) AS total_visits FROM clinical_visits")
    print("Database initialization test:", df_check.to_dict(orient="records"))
