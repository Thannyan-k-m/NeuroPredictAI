/**
 * NeuroPredictAI - Central Frontend API Client
 * Interfaces with FastAPI endpoints using the standard Fetch API.
 */

const API_BASE = "http://127.0.0.1:8000";

const api = {
  // App & Health
  async getHealth() {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error("Backend service unreachable");
    return res.json();
  },

  // Dataset Operations
  async uploadDataset(file) {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${API_BASE}/api/dataset/upload`, {
      method: "POST",
      body: formData
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Upload failed" }));
      throw new Error(err.detail || "Failed to upload dataset");
    }
    return res.json();
  },

  async getDatasetSummary() {
    const res = await fetch(`${API_BASE}/api/dataset/summary`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Dataset not found" }));
      throw new Error(err.detail || "Failed to fetch dataset summary");
    }
    return res.json();
  },

  async getDatasetPreview(n_rows = 10) {
    const res = await fetch(`${API_BASE}/api/dataset/preview?n_rows=${n_rows}`);
    if (!res.ok) throw new Error("Failed to fetch dataset preview");
    return res.json();
  },

  async getDatasetColumns() {
    const res = await fetch(`${API_BASE}/api/dataset/columns`);
    if (!res.ok) throw new Error("Failed to fetch column metadata");
    return res.json();
  },

  async getDatasetAnalytics() {
    const res = await fetch(`${API_BASE}/api/dataset/analytics`);
    if (!res.ok) throw new Error("Failed to fetch dataset analytics");
    return res.json();
  },

  // Model Operations
  async trainModel(model_type, target_column = null) {
    const res = await fetch(`${API_BASE}/api/model/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_type, target_column })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Training error" }));
      throw new Error(err.detail || "Model training failed");
    }
    return res.json();
  },

  async getTrainingStatus() {
    const res = await fetch(`${API_BASE}/api/model/status`);
    if (!res.ok) throw new Error("Failed to get model status");
    return res.json();
  },

  async getModelMetrics(model_name = null) {
    const url = model_name 
      ? `${API_BASE}/api/model/metrics?model_name=${encodeURIComponent(model_name)}`
      : `${API_BASE}/api/model/metrics`;
    const res = await fetch(url);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Not trained" }));
      throw new Error(err.detail || "Model metrics not available");
    }
    return res.json();
  },

  async compareModels() {
    const res = await fetch(`${API_BASE}/api/model/compare`);
    if (!res.ok) throw new Error("Failed to fetch model comparison");
    return res.json();
  },

  // Inference & Explainability
  async predictPatient(model_name, features, patient_ref = null) {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_name,
        features,
        patient_ref
      })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Prediction error" }));
      throw new Error(err.detail || "Inference failed");
    }
    return res.json();
  },

  async getPredictionHistory(limit = 50) {
    const res = await fetch(`${API_BASE}/api/predict/history?limit=${limit}`);
    if (!res.ok) throw new Error("Failed to fetch prediction history");
    return res.json();
  },

  async getModelExplain(model_name = "Random Forest") {
    const res = await fetch(`${API_BASE}/api/model/explain?model_name=${encodeURIComponent(model_name)}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "SHAP explanation unavailable" }));
      throw new Error(err.detail || "Explainability failed");
    }
    return res.json();
  },

  async getFeatureImportance(model_name = "Random Forest") {
    const res = await fetch(`${API_BASE}/api/model/feature-importance?model_name=${encodeURIComponent(model_name)}`);
    if (!res.ok) throw new Error("Feature importance unavailable");
    return res.json();
  }
};
