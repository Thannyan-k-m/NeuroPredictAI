/**
 * NeuroPredictAI - Model Insights Logic
 * Manages model benchmarking, holdout test metrics visualization, confusion matrix rendering,
 * and SHAP global feature importance charts.
 */

let benchmarkChart = null;
let importanceChart = null;

document.addEventListener("DOMContentLoaded", async () => {
  setupEventListeners();
  await refreshModelInsights();
});

function setupEventListeners() {
  const selector = document.getElementById("model-selector");
  selector.addEventListener("change", async () => {
    await loadSingleModelMetrics(selector.value);
  });

  const btnTrain = document.getElementById("btn-train-selected");
  btnTrain.addEventListener("click", async () => {
    await triggerModelTraining(selector.value);
  });
}

async function triggerModelTraining(modelName) {
  const btnTrain = document.getElementById("btn-train-selected");
  const statusDiv = document.getElementById("model-training-status");
  
  btnTrain.disabled = true;
  statusDiv.innerHTML = `<span class="loading-spinner" style="vertical-align:middle; width:16px; height:16px;"></span> Training <strong>${modelName}</strong> on active dataset...`;

  try {
    let modelType = "random_forest";
    if (modelName.toLowerCase().includes("xgb")) modelType = "xgboost";
    if (modelName.toLowerCase().includes("logistic")) modelType = "logistic_regression";

    const res = await api.trainModel(modelType);
    statusDiv.innerHTML = `✅ Successfully trained <strong>${res.model_name}</strong> (Accuracy: ${(res.metrics.accuracy * 100).toFixed(1)}%).`;
    await refreshModelInsights();
  } catch (err) {
    statusDiv.innerHTML = `❌ Training failed: ${err.message}`;
  } finally {
    btnTrain.disabled = false;
  }
}

async function refreshModelInsights() {
  const selector = document.getElementById("model-selector");
  await Promise.all([
    loadSingleModelMetrics(selector.value),
    loadModelComparison()
  ]);
}

async function loadSingleModelMetrics(modelName) {
  const statusDiv = document.getElementById("model-training-status");

  try {
    const [modelData, importanceData] = await Promise.all([
      api.getModelMetrics(modelName),
      api.getFeatureImportance(modelName).catch(() => null)
    ]);

    statusDiv.innerHTML = `Dataset used: <code>${modelData.dataset_filename || 'adni_oasis_dataset.csv'}</code> • Last trained: ${new Date(modelData.training_timestamp).toLocaleDateString()}`;

    const metrics = modelData.metrics;

    // Stat cards
    document.getElementById("val-accuracy").innerText = `${(metrics.accuracy * 100).toFixed(1)}%`;
    document.getElementById("val-f1").innerText = `${(metrics.f1_score * 100).toFixed(1)}%`;
    document.getElementById("val-prec-rec").innerText = `P: ${(metrics.precision * 100).toFixed(0)}% | R: ${(metrics.recall * 100).toFixed(0)}%`;
    document.getElementById("val-roc").innerText = metrics.roc_auc !== null ? `${(metrics.roc_auc * 100).toFixed(1)}%` : "N/A";
    
    document.getElementById("val-sensitivity").innerText = metrics.sensitivity !== null ? `${(metrics.sensitivity * 100).toFixed(1)}%` : "N/A";
    document.getElementById("val-specificity").innerText = metrics.specificity !== null ? `${(metrics.specificity * 100).toFixed(1)}%` : "N/A";

    // Confusion Matrix
    renderConfusionMatrix(metrics.confusion_matrix);

    // Global Feature Importance
    if (importanceData && importanceData.features) {
      renderGlobalImportance(importanceData.features);
    }

  } catch (err) {
    statusDiv.innerHTML = `<span style="color:#b91c1c;">Model has not been trained yet.</span>`;
    document.getElementById("val-accuracy").innerText = "—";
    document.getElementById("val-f1").innerText = "—";
    document.getElementById("val-prec-rec").innerText = "—";
    document.getElementById("val-roc").innerText = "—";
    document.getElementById("cm-container").innerHTML = `<div style="text-align:center; padding:24px; color:#64748b;">Model has not been trained yet.</div>`;
  }
}

function renderConfusionMatrix(cmData) {
  const container = document.getElementById("cm-container");
  if (!cmData || !cmData.matrix || cmData.matrix.length !== 2) {
    container.innerHTML = `<div style="text-align:center; padding:12px; color:#64748b;">Confusion matrix not available.</div>`;
    return;
  }

  const m = cmData.matrix;
  container.innerHTML = `
    <div class="confusion-matrix-grid">
      <div class="cm-cell highlight">
        <div class="cm-value">${m[0][0]}</div>
        <div class="cm-label">True Negative</div>
      </div>
      <div class="cm-cell">
        <div class="cm-value">${m[0][1]}</div>
        <div class="cm-label">False Positive</div>
      </div>
      <div class="cm-cell">
        <div class="cm-value">${m[1][0]}</div>
        <div class="cm-label">False Negative</div>
      </div>
      <div class="cm-cell highlight">
        <div class="cm-value">${m[1][1]}</div>
        <div class="cm-label">True Positive</div>
      </div>
    </div>
  `;
}

async function loadModelComparison() {
  try {
    const comparison = await api.compareModels();
    if (!comparison || !comparison.models || comparison.models.length === 0) return;

    const ctx = document.getElementById("chart-models-benchmark").getContext("2d");
    if (benchmarkChart) benchmarkChart.destroy();

    const labels = comparison.models.map(m => m.model_name);
    const accuracies = comparison.models.map(m => (m.accuracy * 100).toFixed(1));
    const f1s = comparison.models.map(m => (m.f1_score * 100).toFixed(1));
    const rocs = comparison.models.map(m => (m.roc_auc ? (m.roc_auc * 100).toFixed(1) : 0));

    benchmarkChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Accuracy (%)",
            data: accuracies,
            backgroundColor: "rgba(30, 58, 138, 0.85)",
            borderRadius: 4
          },
          {
            label: "F1 Score (%)",
            data: f1s,
            backgroundColor: "rgba(6, 182, 212, 0.85)",
            borderRadius: 4
          },
          {
            label: "ROC-AUC (%)",
            data: rocs,
            backgroundColor: "rgba(16, 185, 129, 0.85)",
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, max: 100, ticks: { callback: v => v + "%" } }
        },
        plugins: {
          legend: { position: "bottom" }
        }
      }
    });

  } catch (err) {
    console.error("Comparison load error:", err);
  }
}

function renderGlobalImportance(features) {
  const ctx = document.getElementById("chart-global-importance").getContext("2d");
  if (importanceChart) importanceChart.destroy();

  const top10 = features.slice(0, 10);
  const labels = top10.map(f => f.feature);
  const data = top10.map(f => f.importance);

  importanceChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Relative Feature Importance (%)",
        data: data,
        backgroundColor: "rgba(6, 182, 212, 0.8)",
        borderColor: "#0891b2",
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          beginAtZero: true,
          ticks: { callback: v => v + "%" }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}
