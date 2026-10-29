/**
 * NeuroPredictAI - Patient Analysis Logic
 * Dynamically builds patient input form matching active dataset schema,
 * collects inputs, triggers inference via FastAPI, renders SHAP contribution chart,
 * and maintains history log from SQLite.
 */

let shapChart = null;
let currentFeatureMeta = [];

document.addEventListener("DOMContentLoaded", async () => {
  await initPatientForm();
  await refreshPredictionHistory();
  setupEvents();
});

function setupEvents() {
  document.getElementById("patient-form").addEventListener("submit", handlePredict);
  document.getElementById("btn-sample-data").addEventListener("click", loadSampleCase);
}

async function initPatientForm() {
  try {
    const colData = await api.getDatasetColumns();
    currentFeatureMeta = colData.input_features || [];
    renderDynamicForm(currentFeatureMeta);
  } catch (err) {
    console.error("Failed to load dynamic columns, using clinical defaults:", err);
    renderDefaultForm();
  }
}

function renderDynamicForm(features) {
  const demoSection = document.getElementById("demographic-section");
  const cogSection = document.getElementById("cognitive-section");
  const bioSection = document.getElementById("biomarker-section");

  demoSection.innerHTML = "";
  cogSection.innerHTML = "";
  bioSection.innerHTML = "";

  if (!features || features.length === 0) {
    renderDefaultForm();
    return;
  }

  features.forEach(feat => {
    const name = feat.name;
    const nameLower = name.toLowerCase();

    // Destination bucket
    let targetSection = bioSection;
    if (nameLower.includes("age") || nameLower.includes("gender") || nameLower.includes("sex") || nameLower.includes("edu") || nameLower.includes("ses")) {
      targetSection = demoSection;
    } else if (nameLower.includes("mmse") || nameLower.includes("cdr") || nameLower.includes("moca") || nameLower.includes("faq") || nameLower.includes("cog")) {
      targetSection = cogSection;
    }

    const defaultVal = feat.mean !== undefined ? feat.mean : (feat.median !== undefined ? feat.median : (feat.type === "categorical" ? "Female" : 0));
    const isCat = feat.type === "categorical" || nameLower.includes("gender") || nameLower.includes("sex");

    const groupDiv = document.createElement("div");
    groupDiv.className = "form-group";

    let inputHtml = "";
    if (isCat) {
      inputHtml = `
        <select name="${name}" id="input-${name}">
          <option value="Female" selected>Female</option>
          <option value="Male">Male</option>
        </select>
      `;
    } else {
      const minVal = feat.min !== undefined ? feat.min : "";
      const maxVal = feat.max !== undefined ? feat.max : "";
      inputHtml = `
        <input type="number" step="any" name="${name}" id="input-${name}" value="${defaultVal}" required>
      `;
    }

    groupDiv.innerHTML = `
      <label for="input-${name}">${name}</label>
      ${inputHtml}
      <span class="form-hint">${feat.type === 'numerical' && feat.min !== undefined ? `Range: ${feat.min} - ${feat.max}` : ''}</span>
    `;

    targetSection.appendChild(groupDiv);
  });
}

function renderDefaultForm() {
  const demoSection = document.getElementById("demographic-section");
  const cogSection = document.getElementById("cognitive-section");
  const bioSection = document.getElementById("biomarker-section");

  demoSection.innerHTML = `
    <div class="form-group">
      <label for="input-Age">Age (Years)</label>
      <input type="number" id="input-Age" name="Age" value="73" required>
      <span class="form-hint">Range: 50 - 95</span>
    </div>
    <div class="form-group">
      <label for="input-Gender">Gender</label>
      <select id="input-Gender" name="Gender">
        <option value="Female" selected>Female</option>
        <option value="Male">Male</option>
      </select>
    </div>
    <div class="form-group">
      <label for="input-Education_Years">Education (Years)</label>
      <input type="number" id="input-Education_Years" name="Education_Years" value="14" required>
    </div>
  `;

  cogSection.innerHTML = `
    <div class="form-group">
      <label for="input-MMSE">MMSE Score (0-30)</label>
      <input type="number" step="0.5" id="input-MMSE" name="MMSE" value="23" required>
      <span class="form-hint">≤24 indicates cognitive impairment</span>
    </div>
    <div class="form-group">
      <label for="input-CDR">Clinical Dementia Rating (CDR)</label>
      <input type="number" step="0.5" id="input-CDR" name="CDR" value="0.5" required>
      <span class="form-hint">0=Normal, 0.5=MCI, 1.0=Mild Dementia</span>
    </div>
  `;

  bioSection.innerHTML = `
    <div class="form-group">
      <label for="input-eTIV">eTIV (mm³)</label>
      <input type="number" id="input-eTIV" name="eTIV" value="1450" required>
    </div>
    <div class="form-group">
      <label for="input-nWBV">nWBV (Normalized Brain Volume)</label>
      <input type="number" step="0.001" id="input-nWBV" name="nWBV" value="0.71" required>
      <span class="form-hint">Typical healthy adult: >0.75</span>
    </div>
    <div class="form-group">
      <label for="input-ASF">Atlas Scaling Factor (ASF)</label>
      <input type="number" step="0.001" id="input-ASF" name="ASF" value="1.21" required>
    </div>
  `;
}

function loadSampleCase() {
  const sampleValues = {
    Age: 76,
    Gender: "Female",
    Education_Years: 12,
    Socioeconomic_Status: 3,
    MMSE: 21,
    CDR: 1.0,
    MoCA: 18,
    FAQ: 12,
    eTIV: 1420,
    nWBV: 0.69,
    ASF: 1.23,
    APOE_e4_Alleles: 1,
    Systolic_BP: 142,
    Cholesterol: 228,
    Blood_Glucose: 114
  };

  Object.entries(sampleValues).forEach(([k, v]) => {
    const input = document.getElementById(`input-${k}`);
    if (input) input.value = v;
  });
}

async function handlePredict(e) {
  e.preventDefault();

  const form = document.getElementById("patient-form");
  const formData = new FormData(form);
  const features = {};

  formData.forEach((value, key) => {
    // Cast numeric if applicable
    if (!isNaN(value) && value.trim() !== "") {
      features[key] = Number(value);
    } else {
      features[key] = value;
    }
  });

  const modelName = document.getElementById("select-model").value;
  const loadingSpan = document.getElementById("predict-loading");
  const btnPredict = document.getElementById("btn-predict");

  loadingSpan.style.display = "inline-flex";
  btnPredict.disabled = true;

  try {
    const result = await api.predictPatient(modelName, features, `PT-${Math.floor(Math.random() * 90000 + 10000)}`);
    displayPredictionResult(result);
    await refreshPredictionHistory();
  } catch (err) {
    alert("Inference Error: " + err.message);
  } finally {
    loadingSpan.style.display = "none";
    btnPredict.disabled = false;
  }
}

function displayPredictionResult(res) {
  const container = document.getElementById("result-container");
  container.style.display = "block";

  document.getElementById("res-model-name").innerText = res.model_name;
  document.getElementById("res-class").innerText = res.predicted_class;
  
  const prob = res.probability !== null ? (res.probability * 100).toFixed(1) : "N/A";
  document.getElementById("res-probability").innerText = prob !== "N/A" ? `${prob}%` : "Not Provided";
  
  const probFill = document.getElementById("res-prob-fill");
  if (res.probability !== null) {
    probFill.style.width = `${Math.min(res.probability * 100, 100)}%`;
  } else {
    probFill.style.width = "50%";
  }

  const badge = document.getElementById("res-risk-badge");
  const isHigh = String(res.predicted_class).includes("1") || String(res.predicted_class).toLowerCase().includes("demented") || (res.probability && res.probability > 0.5);
  badge.className = `risk-badge ${isHigh ? 'high' : 'low'}`;

  // Render SHAP attribution chart
  if (res.explanation && res.explanation.contributions) {
    renderShapChart(res.explanation.contributions);
  }

  // Scroll to results smoothly
  container.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderShapChart(contributions) {
  const ctx = document.getElementById("chart-shap-contributions").getContext("2d");
  if (shapChart) shapChart.destroy();

  const labels = contributions.map(c => c.feature);
  const values = contributions.map(c => c.contribution);
  const backgroundColors = values.map(v => v >= 0 ? "rgba(239, 68, 68, 0.75)" : "rgba(16, 185, 129, 0.75)");

  shapChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Feature Attribution (SHAP)",
        data: values,
        backgroundColor: backgroundColors,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` Impact: ${ctx.raw > 0 ? '+' : ''}${ctx.raw.toFixed(4)}`
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: "Relative SHAP Attribution Weight" }
        }
      }
    }
  });
}

async function refreshPredictionHistory() {
  try {
    const history = await api.getPredictionHistory(15);
    const tbody = document.getElementById("history-body");

    if (!history || history.length === 0) {
      tbody.innerHTML = "<tr><td colspan='7' style='text-align:center; color:#64748b;'>No predictions recorded yet.</td></tr>";
      return;
    }

    tbody.innerHTML = history.map(item => {
      const probStr = item.probability !== null ? `${(item.probability * 100).toFixed(1)}%` : "N/A";
      const dt = new Date(item.created_at).toLocaleString();
      return `
        <tr>
          <td>#${item.prediction_id}</td>
          <td><code>${item.patient_ref || 'ANON'}</code></td>
          <td><strong>${item.model_name}</strong></td>
          <td><span style="font-weight:600;">Class ${item.predicted_class}</span></td>
          <td>${probStr}</td>
          <td><span style="background:#ecfdf5; color:#059669; font-size:0.75rem; padding:2px 8px; border-radius:4px;">${item.validation_status}</span></td>
          <td style="color:#64748b; font-size:0.8rem;">${dt}</td>
        </tr>
      `;
    }).join("");
  } catch (err) {
    console.error("Failed to load history:", err);
  }
}
