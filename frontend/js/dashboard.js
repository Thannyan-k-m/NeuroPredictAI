/**
 * NeuroPredictAI - Dashboard Logic
 * Fetches real dataset statistics, holds real Chart.js charts, and handles model comparison metrics.
 */

let classChart = null;
let modelChart = null;
let ageChart = null;
let cognitiveChart = null;

document.addEventListener("DOMContentLoaded", async () => {
  await loadDashboardData();
});

async function loadDashboardData() {
  try {
    // 1. Check health
    const health = await api.getHealth();
    document.getElementById("backend-status-badge").innerHTML = `<span class="status-dot"></span> Backend Active`;
    
    // 2. Fetch dataset summary & analytics
    const [summary, analytics, trainingStatus, modelComparison] = await Promise.all([
      api.getDatasetSummary().catch(() => null),
      api.getDatasetAnalytics().catch(() => null),
      api.getTrainingStatus().catch(() => ({ status: "Idle" })),
      api.compareModels().catch(() => ({ models: [] }))
    ]);

    // Update Quick Banner & Stat Cards
    if (summary) {
      document.getElementById("cohort-status-text").innerText = 
        `Loaded: ${summary.filename} (${summary.total_records.toLocaleString()} patient records, ${summary.total_columns} columns)`;
      document.getElementById("stat-records").innerText = summary.total_records.toLocaleString();
      document.getElementById("stat-features").innerText = summary.total_columns;
    } else {
      document.getElementById("cohort-status-text").innerText = "No dataset loaded. Upload a CSV to begin analysis.";
      document.getElementById("stat-records").innerText = "0";
      document.getElementById("stat-features").innerText = "0";
    }

    document.getElementById("stat-models").innerText = modelComparison.models ? modelComparison.models.length : "0";
    document.getElementById("stat-training-status").innerText = trainingStatus.is_training ? "Training..." : (trainingStatus.status || "Idle");

    // Render Charts with actual data
    if (analytics) {
      renderClassDistribution(analytics.target_distribution);
      renderHistograms(analytics.feature_histograms);
    }
    
    if (modelComparison && modelComparison.models && modelComparison.models.length > 0) {
      renderModelComparison(modelComparison.models);
    }

  } catch (err) {
    console.error("Dashboard data load error:", err);
    document.getElementById("cohort-status-text").innerText = "Unable to connect to backend service. Please ensure FastAPI is running on port 8000.";
    document.getElementById("backend-status-badge").className = "status-badge";
    document.getElementById("backend-status-badge").style.background = "#fee2e2";
    document.getElementById("backend-status-badge").style.color = "#b91c1c";
    document.getElementById("backend-status-badge").innerHTML = `⚠️ Backend Offline`;
  }
}

function renderClassDistribution(distribution) {
  const ctx = document.getElementById("chart-class-distribution").getContext("2d");
  if (classChart) classChart.destroy();

  if (!distribution || Object.keys(distribution).length === 0) {
    ctx.font = "14px -apple-system, sans-serif";
    ctx.fillStyle = "#94a3b8";
    ctx.textAlign = "center";
    ctx.fillText("No class data available", ctx.canvas.width / 2, 140);
    return;
  }

  const labels = Object.keys(distribution);
  const data = Object.values(distribution);

  const colors = [
    "rgba(6, 182, 212, 0.8)",
    "rgba(30, 58, 138, 0.8)",
    "rgba(245, 158, 11, 0.8)",
    "rgba(16, 185, 129, 0.8)"
  ];

  classChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: labels.map(l => `Class ${l}`),
      datasets: [{
        data: data,
        backgroundColor: colors.slice(0, labels.length),
        borderWidth: 2,
        borderColor: "#ffffff"
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom" },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.label}: ${ctx.raw} records`
          }
        }
      }
    }
  });
}

function renderModelComparison(models) {
  const ctx = document.getElementById("chart-model-comparison").getContext("2d");
  if (modelChart) modelChart.destroy();

  const labels = models.map(m => m.model_name);
  const accuracies = models.map(m => (m.accuracy ? m.accuracy * 100 : 0));
  const f1Scores = models.map(m => (m.f1_score ? m.f1_score * 100 : 0));
  const precisions = models.map(m => (m.precision ? m.precision * 100 : 0));

  modelChart = new Chart(ctx, {
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
          data: f1Scores,
          backgroundColor: "rgba(6, 182, 212, 0.85)",
          borderRadius: 4
        },
        {
          label: "Precision (%)",
          data: precisions,
          backgroundColor: "rgba(16, 185, 129, 0.85)",
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          ticks: { callback: v => v + "%" }
        }
      },
      plugins: {
        legend: { position: "bottom" }
      }
    }
  });
}

function renderHistograms(histograms) {
  if (!histograms) return;

  // Age chart
  if (histograms["Age"]) {
    const ageData = histograms["Age"];
    const ctx = document.getElementById("chart-age-distribution").getContext("2d");
    if (ageChart) ageChart.destroy();

    ageChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: ageData.labels,
        datasets: [{
          label: "Patients",
          data: ageData.counts,
          backgroundColor: "rgba(59, 130, 246, 0.7)",
          borderColor: "#3b82f6",
          borderWidth: 1,
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          y: { beginAtZero: true, ticks: { precision: 0 } },
          x: { ticks: { maxRotation: 45, minRotation: 45, font: { size: 10 } } }
        }
      }
    });
  }

  // Cognitive Score (MMSE)
  const mmseKey = histograms["MMSE"] ? "MMSE" : Object.keys(histograms)[0];
  if (mmseKey && histograms[mmseKey]) {
    const cogData = histograms[mmseKey];
    const ctx = document.getElementById("chart-cognitive-distribution").getContext("2d");
    if (cognitiveChart) cognitiveChart.destroy();

    cognitiveChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: cogData.labels,
        datasets: [{
          label: `${mmseKey} Count`,
          data: cogData.counts,
          backgroundColor: "rgba(16, 185, 129, 0.7)",
          borderColor: "#10b981",
          borderWidth: 1,
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          y: { beginAtZero: true, ticks: { precision: 0 } },
          x: { ticks: { maxRotation: 45, minRotation: 45, font: { size: 10 } } }
        }
      }
    });
  }
}
