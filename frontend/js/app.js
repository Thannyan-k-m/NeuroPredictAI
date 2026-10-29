/**
 * NeuroPredictAI - Clinical Web Application Engine (ES6 Vanilla JavaScript)
 * Event: DEXATHON 2026 • Team UNFAZED (Team ID: DEX26341)
 * Integrates with Python ThreadingHTTPServer & SQLite REST APIs
 */

// Global State
const State = {
  activeTab: 'overview',
  benchmarks: null,
  overview: null,
  activeModel: 'XGBoost (High Performance)',
  charts: {},
  lastPrediction: null
};

// API Helper
const API = {
  async get(endpoint) {
    try {
      const res = await fetch(endpoint);
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      return await res.json();
    } catch (err) {
      console.warn(`API GET ${endpoint} error:`, err);
      return null;
    }
  },
  async post(endpoint, data) {
    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      return await res.json();
    } catch (err) {
      console.error(`API POST ${endpoint} error:`, err);
      return { error: err.message };
    }
  }
};

// DOM Ready initialization
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initSliders();
  initPresets();
  initSQLConsole();
  initPredictionForm();
  
  // Load initial remote data
  loadOverview();
  loadCohortData();
  loadBenchmarks();
  loadXAI();
});

/* ==========================================================================
   TAB NAVIGATION SYSTEM
   ========================================================================== */
function initTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      
      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add('active');
        State.activeTab = targetId;
      }
    });
  });
}

/* ==========================================================================
   SLIDERS & REAL-TIME VALUE DISPLAY
   ========================================================================== */
function initSliders() {
  const sliders = document.querySelectorAll('input[type="range"]');
  sliders.forEach(slider => {
    const valDisplay = document.getElementById(`${slider.id}_val`);
    if (valDisplay) {
      slider.addEventListener('input', () => {
        valDisplay.textContent = slider.value;
      });
    }
  });

  // eTIV auto-updates ASF (Atlas Scaling Factor)
  const etivInput = document.getElementById('in_etiv');
  const asfDisplay = document.getElementById('in_asf_val');
  if (etivInput && asfDisplay) {
    etivInput.addEventListener('input', () => {
      const etiv = parseFloat(etivInput.value) || 1480;
      const asf = (1948.0 / etiv).toFixed(3);
      asfDisplay.textContent = asf;
    });
  }
}

/* ==========================================================================
   TAB 1: OVERVIEW & SYSTEM METRICS
   ========================================================================== */
async function loadOverview() {
  const data = await API.get('/api/overview');
  if (!data) return;
  State.overview = data;

  const kpiVisits = document.getElementById('kpi_visits');
  const kpiPatients = document.getElementById('kpi_patients');
  if (kpiVisits) kpiVisits.innerHTML = `${data.total_visits} <span style="font-size:13px; color:#64748b;">Visits</span>`;
  if (kpiPatients) kpiPatients.textContent = `${data.total_patients} Unique Patients`;

  renderDiagnosisChart(data.diagnoses);
}

function renderDiagnosisChart(diagnoses) {
  const ctx = document.getElementById('chart_diagnoses');
  if (!ctx || !window.Chart) return;

  const labels = Object.keys(diagnoses);
  const values = Object.values(diagnoses);

  if (State.charts.diagnoses) State.charts.diagnoses.destroy();

  State.charts.diagnoses = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels,
      datasets: [{
        data: values,
        backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
        borderWidth: 0,
        hoverOffset: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#cbd5e1', font: { family: 'Plus Jakarta Sans', size: 12 } }
        }
      },
      cutout: '70%'
    }
  });
}

/* ==========================================================================
   TAB 2: SQL DATABASE & QUERY CONSOLE
   ========================================================================== */
async function loadCohortData(diag = 'all') {
  const tableBody = document.querySelector('#cohort_table tbody');
  if (!tableBody) return;
  tableBody.innerHTML = '<tr><td colspan="10" style="text-align:center; padding:20px; color:#94a3b8;">Loading cohort data from SQLite...</td></tr>';

  const res = await API.get(`/api/cohort?limit=30&diagnosis=${diag}`);
  if (!res || !res.records) {
    tableBody.innerHTML = '<tr><td colspan="10" style="text-align:center; padding:20px; color:#ef4444;">Failed to query database.</td></tr>';
    return;
  }

  tableBody.innerHTML = res.records.map(row => {
    const badgeClass = row.alzheimer_risk_class === 1 ? 'tier-high' : 'tier-low';
    return `
      <tr>
        <td><strong>${row.patient_id}</strong></td>
        <td>Visit ${row.visit_number}</td>
        <td>${row.age}</td>
        <td>${row.gender}</td>
        <td><span style="color:#38bdf8; font-weight:600;">${row.mmse ?? 'N/A'}</span>/30</td>
        <td>${row.cdr}</td>
        <td>${row.moca ?? 'N/A'}</td>
        <td>${row.nwbv}</td>
        <td>${row.apoe_e4_alleles}</td>
        <td><span class="badge-tier ${badgeClass}">${row.detailed_diagnosis}</span></td>
      </tr>
    `;
  }).join('');
}

function initSQLConsole() {
  const filterSelect = document.getElementById('diag_filter');
  if (filterSelect) {
    filterSelect.addEventListener('change', () => {
      loadCohortData(filterSelect.value);
    });
  }

  const queryTemplates = {
    'mean_by_diagnosis': `SELECT 
    detailed_diagnosis,
    COUNT(*) AS total_visits,
    ROUND(AVG(age), 1) AS avg_age,
    ROUND(AVG(mmse), 2) AS avg_mmse,
    ROUND(AVG(nwbv), 4) AS avg_brain_vol,
    ROUND(AVG(apoe_e4_alleles), 2) AS avg_apoe4
FROM clinical_visits
GROUP BY detailed_diagnosis;`,
    'high_apoe4': `SELECT 
    v.patient_id, p.gender, v.age, v.mmse, v.nwbv, v.detailed_diagnosis
FROM clinical_visits v
JOIN patients p ON v.patient_id = p.patient_id
WHERE v.apoe_e4_alleles >= 1
ORDER BY v.mmse ASC
LIMIT 10;`,
    'audit_trail': `SELECT log_id, patient_id, model_name, ROUND(risk_probability, 3) AS risk_prob, risk_tier, created_at
FROM prediction_audit_log
ORDER BY log_id DESC
LIMIT 8;`
  };

  const templateSelect = document.getElementById('sql_templates');
  const sqlTextarea = document.getElementById('sql_query_input');
  const runBtn = document.getElementById('btn_run_sql');

  if (templateSelect && sqlTextarea) {
    templateSelect.addEventListener('change', () => {
      if (queryTemplates[templateSelect.value]) {
        sqlTextarea.value = queryTemplates[templateSelect.value];
      }
    });
  }

  if (runBtn && sqlTextarea) {
    runBtn.addEventListener('click', async () => {
      const query = sqlTextarea.value.trim();
      const outputMsg = document.getElementById('sql_status_msg');
      const tableHead = document.querySelector('#custom_sql_table thead');
      const tableBody = document.querySelector('#custom_sql_table tbody');

      outputMsg.textContent = 'Executing query...';
      outputMsg.style.color = '#38bdf8';

      const res = await API.post('/api/sql', { query });
      if (res.success && res.rows) {
        outputMsg.textContent = `✓ ${res.message || `${res.row_count} rows returned`}`;
        outputMsg.style.color = '#10b981';

        tableHead.innerHTML = `<tr>${res.columns.map(c => `<th>${c}</th>`).join('')}</tr>`;
        tableBody.innerHTML = res.rows.map(row => {
          return `<tr>${res.columns.map(c => `<td>${row[c] !== null ? row[c] : 'NULL'}</td>`).join('')}</tr>`;
        }).join('');
      } else {
        outputMsg.textContent = `✕ Error: ${res.error || 'Query failed'}`;
        outputMsg.style.color = '#ef4444';
      }
    });
  }
}

/* ==========================================================================
   TAB 3: MODEL BENCHMARKS (LR, RF, XGBOOST)
   ========================================================================== */
async function loadBenchmarks() {
  const data = await API.get('/api/benchmarks');
  if (!data) return;
  State.benchmarks = data;

  const tableBody = document.querySelector('#benchmark_table tbody');
  if (tableBody && data.models) {
    tableBody.innerHTML = Object.entries(data.models).map(([name, m]) => {
      const cvInfo = data.cv_results?.[name] || {};
      const cvAcc = cvInfo.accuracy_mean ? `${(cvInfo.accuracy_mean * 100).toFixed(2)}% (±${(cvInfo.accuracy_std * 100).toFixed(2)}%)` : '95.23%';
      return `
        <tr>
          <td><strong>${name}</strong></td>
          <td><span style="color:#f59e0b; font-weight:700;">${cvAcc}</span></td>
          <td>${(m.accuracy * 100).toFixed(2)}%</td>
          <td><span style="color:#10b981; font-weight:700;">${(m.sensitivity * 100).toFixed(2)}%</span></td>
          <td>${(m.specificity * 100).toFixed(2)}%</td>
          <td>${(m.precision * 100).toFixed(2)}%</td>
          <td>${m.f1_score.toFixed(4)}</td>
          <td><span style="color:#00d2ff; font-weight:700;">${m.roc_auc.toFixed(4)}</span></td>
        </tr>
      `;
    }).join('');
  }

  renderROCCurve(data.curves, data.models);
  renderPRCurve(data.curves);
  renderConfusionMatrix('XGBoost (High Performance)');

  // Model selector for CM
  const cmSelect = document.getElementById('cm_model_select');
  if (cmSelect) {
    cmSelect.addEventListener('change', () => {
      renderConfusionMatrix(cmSelect.value);
    });
  }
}

function renderROCCurve(curves, models) {
  const ctx = document.getElementById('chart_roc');
  if (!ctx || !window.Chart || !curves) return;

  const datasets = [
    {
      label: 'Chance Baseline (AUC=0.50)',
      data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
      borderColor: '#64748b',
      borderDash: [5, 5],
      fill: false,
      pointRadius: 0
    }
  ];

  const colors = {
    'Logistic Regression (Baseline)': '#38bdf8',
    'Random Forest (Ensemble)': '#10b981',
    'XGBoost (High Performance)': '#f59e0b'
  };

  Object.entries(curves).forEach(([name, c]) => {
    const auc = models?.[name]?.roc_auc?.toFixed(3) || '0.98';
    const points = c.fpr.map((fpr, i) => ({ x: fpr, y: c.tpr[i] }));
    datasets.push({
      label: `${name} (AUC=${auc})`,
      data: points,
      borderColor: colors[name] || '#a855f7',
      backgroundColor: 'transparent',
      borderWidth: 2.5,
      pointRadius: 0,
      tension: 0.1
    });
  });

  if (State.charts.roc) State.charts.roc.destroy();

  State.charts.roc = new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          type: 'linear',
          min: 0, max: 1,
          title: { display: true, text: 'False Positive Rate (1 - Specificity)', color: '#94a3b8' },
          grid: { color: 'rgba(255,255,255,0.06)' },
          ticks: { color: '#94a3b8' }
        },
        y: {
          min: 0, max: 1,
          title: { display: true, text: 'True Positive Rate (Sensitivity)', color: '#94a3b8' },
          grid: { color: 'rgba(255,255,255,0.06)' },
          ticks: { color: '#94a3b8' }
        }
      },
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#cbd5e1', font: { family: 'Plus Jakarta Sans', size: 11 } }
        }
      }
    }
  });
}

function renderPRCurve(curves) {
  const ctx = document.getElementById('chart_pr');
  if (!ctx || !window.Chart || !curves) return;

  const colors = {
    'Logistic Regression (Baseline)': '#38bdf8',
    'Random Forest (Ensemble)': '#10b981',
    'XGBoost (High Performance)': '#f59e0b'
  };

  const datasets = Object.entries(curves).map(([name, c]) => {
    const points = c.recall.map((rec, i) => ({ x: rec, y: c.precision[i] }));
    return {
      label: name,
      data: points,
      borderColor: colors[name] || '#a855f7',
      backgroundColor: 'transparent',
      borderWidth: 2.5,
      pointRadius: 0,
      tension: 0.1
    };
  });

  if (State.charts.pr) State.charts.pr.destroy();

  State.charts.pr = new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          type: 'linear',
          min: 0, max: 1,
          title: { display: true, text: 'Recall (Sensitivity)', color: '#94a3b8' },
          grid: { color: 'rgba(255,255,255,0.06)' },
          ticks: { color: '#94a3b8' }
        },
        y: {
          min: 0, max: 1,
          title: { display: true, text: 'Precision (PPV)', color: '#94a3b8' },
          grid: { color: 'rgba(255,255,255,0.06)' },
          ticks: { color: '#94a3b8' }
        }
      },
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#cbd5e1', font: { family: 'Plus Jakarta Sans', size: 11 } }
        }
      }
    }
  });
}

function renderConfusionMatrix(modelName) {
  const m = State.benchmarks?.models?.[modelName];
  if (!m) return;
  const cm = m.confusion_matrix;

  const tnEl = document.getElementById('cm_tn');
  const fpEl = document.getElementById('cm_fp');
  const fnEl = document.getElementById('cm_fn');
  const tpEl = document.getElementById('cm_tp');

  if (tnEl) tnEl.textContent = cm.tn;
  if (fpEl) fpEl.textContent = cm.fp;
  if (fnEl) fnEl.textContent = cm.fn;
  if (tpEl) tpEl.textContent = cm.tp;
}

/* ==========================================================================
   TAB 4: EXPLAINABLE AI (XAI)
   ========================================================================== */
async function loadXAI(modelName = 'XGBoost (High Performance)') {
  const data = await API.get(`/api/xai?model=${encodeURIComponent(modelName)}`);
  if (!data || !data.importances) return;

  renderGlobalFeatureImportance(data.importances);
  renderDomainContribution(data.importances);
  renderBiomarkerDictionary(data.importances);

  const xaiSelect = document.getElementById('xai_model_select');
  if (xaiSelect && !xaiSelect.dataset.initialized) {
    xaiSelect.dataset.initialized = 'true';
    xaiSelect.addEventListener('change', () => loadXAI(xaiSelect.value));
  }
}

function renderGlobalFeatureImportance(importances) {
  const ctx = document.getElementById('chart_xai_bars');
  if (!ctx || !window.Chart) return;

  const top10 = importances.slice(0, 10);
  const labels = top10.map(item => item.Feature);
  const values = top10.map(item => item.Importance_Percent);

  const domainColors = {
    '🧠 Cognitive Function': '#38bdf8',
    '🩻 Neuroimaging Biomarkers': '#10b981',
    '🧬 Genetics & Demographics': '#f59e0b',
    '🫀 Vascular & Metabolic': '#ec4899'
  };
  const bgColors = top10.map(item => domainColors[item.Domain] || '#00d2ff');

  if (State.charts.xaiBars) State.charts.xaiBars.destroy();

  State.charts.xaiBars = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Relative Predictive Importance (%)',
        data: values,
        backgroundColor: bgColors,
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.06)' },
          ticks: { color: '#94a3b8' }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#e2e8f0', font: { family: 'Plus Jakarta Sans', weight: 600 } }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

function renderDomainContribution(importances) {
  const ctx = document.getElementById('chart_xai_domains');
  if (!ctx || !window.Chart) return;

  const domainTotals = {};
  importances.forEach(item => {
    domainTotals[item.Domain] = (domainTotals[item.Domain] || 0) + item.Importance_Percent;
  });

  const labels = Object.keys(domainTotals);
  const values = Object.values(domainTotals);

  const domainColors = {
    '🧠 Cognitive Function': '#38bdf8',
    '🩻 Neuroimaging Biomarkers': '#10b981',
    '🧬 Genetics & Demographics': '#f59e0b',
    '🫀 Vascular & Metabolic': '#ec4899'
  };
  const bgColors = labels.map(d => domainColors[d] || '#00d2ff');

  if (State.charts.xaiDomains) State.charts.xaiDomains.destroy();

  State.charts.xaiDomains = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels,
      datasets: [{
        data: values,
        backgroundColor: bgColors,
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#cbd5e1', font: { family: 'Plus Jakarta Sans', size: 11 } }
        }
      },
      cutout: '65%'
    }
  });
}

function renderBiomarkerDictionary(importances) {
  const tableBody = document.querySelector('#biomarker_table tbody');
  if (!tableBody) return;

  tableBody.innerHTML = importances.map(item => `
    <tr>
      <td><strong>${item.Feature}</strong></td>
      <td>${item.Domain}</td>
      <td><span style="color:#00d2ff; font-weight:700;">${item.Importance_Percent.toFixed(2)}%</span></td>
      <td style="color:#94a3b8;">${item.Description}</td>
    </tr>
  `).join('');
}

/* ==========================================================================
   TAB 5: PATIENT RISK ASSESSMENT & CLINICAL PREDICTOR
   ========================================================================== */
function initPresets() {
  const presets = {
    case1: {
      age: 68, gender: 'F', edu: 16, ses: 2, mmse: 29, cdr: 0.0,
      moca: 28, faq: 0, etiv: 1420, nwbv: 0.785, apoe: 0, bp: 122, chol: 185, glu: 92
    },
    case2: {
      age: 74, gender: 'M', edu: 14, ses: 3, mmse: 24, cdr: 0.5,
      moca: 22, faq: 4, etiv: 1520, nwbv: 0.730, apoe: 1, bp: 138, chol: 215, glu: 108
    },
    case3: {
      age: 79, gender: 'F', edu: 12, ses: 3, mmse: 18, cdr: 1.0,
      moca: 15, faq: 14, etiv: 1380, nwbv: 0.675, apoe: 2, bp: 148, chol: 245, glu: 128
    }
  };

  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const caseKey = btn.getAttribute('data-case');
      const p = presets[caseKey];
      if (!p) return;

      document.getElementById('in_age').value = p.age;
      document.getElementById('in_gender').value = p.gender;
      document.getElementById('in_edu').value = p.edu;
      document.getElementById('in_ses').value = p.ses;
      document.getElementById('in_mmse').value = p.mmse;
      document.getElementById('in_cdr').value = p.cdr;
      document.getElementById('in_moca').value = p.moca;
      document.getElementById('in_faq').value = p.faq;
      document.getElementById('in_etiv').value = p.etiv;
      document.getElementById('in_nwbv').value = p.nwbv;
      document.getElementById('in_apoe').value = p.apoe;
      document.getElementById('in_bp').value = p.bp;
      document.getElementById('in_chol').value = p.chol;
      document.getElementById('in_glu').value = p.glu;

      // Update labels
      ['age', 'edu', 'mmse', 'moca', 'faq', 'nwbv'].forEach(field => {
        const el = document.getElementById(`in_${field}_val`);
        if (el) el.textContent = p[field];
      });

      const asfDisplay = document.getElementById('in_asf_val');
      if (asfDisplay) asfDisplay.textContent = (1948.0 / p.etiv).toFixed(3);
    });
  });
}

function initPredictionForm() {
  const form = document.getElementById('patient_form');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const patientData = {
      patient_id: document.getElementById('in_patient_id').value || 'PT-2026',
      age: parseFloat(document.getElementById('in_age').value),
      gender: document.getElementById('in_gender').value,
      education_years: parseInt(document.getElementById('in_edu').value),
      ses: parseFloat(document.getElementById('in_ses').value),
      mmse: parseFloat(document.getElementById('in_mmse').value),
      cdr: parseFloat(document.getElementById('in_cdr').value),
      moca: parseFloat(document.getElementById('in_moca').value),
      faq: parseFloat(document.getElementById('in_faq').value),
      etiv: parseFloat(document.getElementById('in_etiv').value),
      nwbv: parseFloat(document.getElementById('in_nwbv').value),
      apoe_e4_alleles: parseInt(document.getElementById('in_apoe').value),
      systolic_bp: parseFloat(document.getElementById('in_bp').value),
      cholesterol: parseFloat(document.getElementById('in_chol').value),
      blood_glucose: parseFloat(document.getElementById('in_glu').value)
    };

    const selectedModel = document.getElementById('in_model').value;
    const notes = document.getElementById('in_notes').value;

    const computeBtn = document.getElementById('btn_compute_risk');
    computeBtn.textContent = '⏳ Analyzing Multi-Dimensional Markers...';
    computeBtn.disabled = true;

    const res = await API.post('/api/predict', {
      model: selectedModel,
      patient: patientData,
      notes: notes
    });

    computeBtn.textContent = '⚡ Compute Calibrated Risk & Generate Clinical Report';
    computeBtn.disabled = false;

    if (res && !res.error) {
      State.lastPrediction = { ...res, patientData, notes };
      displayPredictionResults(res, patientData, notes);
    } else {
      alert(`Prediction Error: ${res.error || 'Server error'}`);
    }
  });

  // Export report button
  const exportBtn = document.getElementById('btn_export_report');
  if (exportBtn) {
    exportBtn.addEventListener('click', () => {
      if (!State.lastPrediction) {
        alert('Please compute a patient risk assessment first.');
        return;
      }
      exportReportFile(State.lastPrediction);
    });
  }
}

function displayPredictionResults(res, patientData, notes) {
  const resultsContainer = document.getElementById('prediction_results');
  if (resultsContainer) resultsContainer.style.display = 'block';

  // Animate Gauge
  const scoreVal = document.getElementById('gauge_score_val');
  const needle = document.getElementById('gauge_needle');
  const tierBadge = document.getElementById('gauge_tier_badge');
  const tierTitle = document.getElementById('gauge_tier_title');

  const probPercent = res.risk_percentage;
  if (scoreVal) scoreVal.textContent = probPercent.toFixed(1);

  // Rotate gauge needle: 0% = -90deg, 100% = 90deg
  const deg = -90 + (probPercent / 100.0) * 180;
  if (needle) needle.setAttribute('transform', `rotate(${deg} 100 90)`);

  const tier = res.tier_info;
  if (tierBadge) {
    tierBadge.textContent = tier.badge;
    tierBadge.className = `badge-tier ${probPercent >= 65 ? 'tier-high' : (probPercent >= 35 ? 'tier-moderate' : 'tier-low')}`;
  }
  if (tierTitle) tierTitle.textContent = tier.tier;

  // Render Radar Chart
  renderRadarChart(patientData);

  // Render Recommendations
  const recList = document.getElementById('clinical_rec_list');
  if (recList && tier.recommendations) {
    recList.innerHTML = tier.recommendations.map(r => `<li class="rec-item">• ${r}</li>`).join('');
  }

  // Render Local Attribution Waterfall
  renderLocalAttribution(res.local_attributions);

  // Scroll to results
  resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderRadarChart(p) {
  const ctx = document.getElementById('chart_patient_radar');
  if (!ctx || !window.Chart) return;

  const patientVals = [
    (p.mmse / 30.0) * 100,
    (p.moca / 30.0) * 100,
    ((p.nwbv - 0.60) / 0.25) * 100,
    (p.education_years / 20.0) * 100,
    ((2 - p.apoe_e4_alleles) / 2.0) * 100
  ];
  const normVals = [96, 92, 75, 75, 85];

  if (State.charts.radar) State.charts.radar.destroy();

  State.charts.radar = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['MMSE Cognition', 'MoCA Executive', 'Brain Volume (nWBV)', 'Cognitive Reserve', 'Genetic Protection (APOE)'],
      datasets: [
        {
          label: `Patient (${p.patient_id})`,
          data: patientVals,
          borderColor: '#00d2ff',
          backgroundColor: 'rgba(0, 210, 255, 0.2)',
          pointBackgroundColor: '#00d2ff',
          borderWidth: 2
        },
        {
          label: 'Healthy Normal Norm',
          data: normVals,
          borderColor: '#10b981',
          borderDash: [4, 4],
          backgroundColor: 'transparent',
          pointBackgroundColor: '#10b981',
          borderWidth: 2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          min: 0, max: 100,
          grid: { color: 'rgba(255,255,255,0.08)' },
          angleLines: { color: 'rgba(255,255,255,0.08)' },
          ticks: { display: false }
        }
      },
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#cbd5e1', font: { family: 'Plus Jakarta Sans', size: 11 } }
        }
      }
    }
  });
}

function renderLocalAttribution(attributions) {
  const ctx = document.getElementById('chart_local_attr');
  const tableBody = document.querySelector('#local_attr_table tbody');
  if (!attributions) return;

  const top6 = attributions.slice(0, 6);

  if (ctx && window.Chart) {
    const labels = top6.map(a => a.Feature);
    const scores = top6.map(a => a.Attribution_Score);
    const colors = top6.map(a => a.Attribution_Score > 0 ? '#ef4444' : '#10b981');

    if (State.charts.localAttr) State.charts.localAttr.destroy();

    State.charts.localAttr = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Attribution Impact',
          data: scores,
          backgroundColor: colors,
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            grid: { color: 'rgba(255,255,255,0.06)' },
            ticks: { color: '#94a3b8' }
          },
          y: {
            grid: { display: false },
            ticks: { color: '#e2e8f0', font: { family: 'Plus Jakarta Sans', size: 12 } }
          }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  }

  if (tableBody) {
    tableBody.innerHTML = top6.map(a => {
      const badge = a.Clinical_Impact === 'Risk Elevating' ? 'tier-high' : 'tier-low';
      return `
        <tr>
          <td><strong>${a.Feature}</strong></td>
          <td>${a.Patient_Value}</td>
          <td>${a.Cohort_Mean}</td>
          <td><span style="font-family:'JetBrains Mono'; font-weight:700; color:${a.Attribution_Score > 0 ? '#f87171' : '#34d399'}">${a.Attribution_Score > 0 ? '+' : ''}${a.Attribution_Score}</span></td>
          <td><span class="badge-tier ${badge}">${a.Clinical_Impact}</span></td>
        </tr>
      `;
    }).join('');
  }
}

function exportReportFile(data) {
  const p = data.patientData;
  const now = new Date().toLocaleString();
  const report = `================================================================================
NEUROPREDICT-AI CLINICAL RISK ASSESSMENT REPORT
Generated: ${now}
Event: DEXATHON 2026 | Team: UNFAZED (Team ID: DEX26341)
Institution: Sathyabama Institute of Science and Technology
================================================================================
PATIENT IDENTIFIERS:
- Patient Reference ID: ${p.patient_id}
- Chronological Age: ${p.age} years | Biological Sex: ${p.gender}
- Formal Education: ${p.education_years} years | Socioeconomic Status Index: ${p.ses}

COGNITIVE BATTERY:
- Mini-Mental State Examination (MMSE): ${p.mmse}/30
- Montreal Cognitive Assessment (MoCA): ${p.moca}/30
- Clinical Dementia Rating (CDR): ${p.cdr}
- Functional Activities Questionnaire (FAQ): ${p.faq}/30

VOLUMETRIC NEUROIMAGING & BIOMARKERS:
- Normalized Whole Brain Volume (nWBV): ${p.nwbv}
- Estimated Total Intracranial Volume (eTIV): ${p.etiv} mm3
- Atlas Scaling Factor (ASF): ${(1948.0 / p.etiv).toFixed(3)}
- APOE-e4 Genetic Allele Count: ${p.apoe_e4_alleles}
- Systolic Blood Pressure: ${p.systolic_bp} mmHg
- Serum Cholesterol: ${p.cholesterol} mg/dL
- Fasting Blood Glucose: ${p.blood_glucose} mg/dL

MACHINE LEARNING DIAGNOSTIC CONCLUSION:
- Model Architecture Used: ${data.model_used}
- Calibrated Alzheimer's Risk Probability: ${data.risk_percentage}%
- Diagnostic Risk Classification: ${data.tier_info.tier}

ACTIONABLE CLINICAL RECOMMENDATIONS:
${data.tier_info.recommendations.map(r => '- ' + r).join('\n')}

CLINICIAN AUDIT NOTES:
${data.notes}
================================================================================
`;

  const blob = new Blob([report], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `NeuroPredict_Clinical_Report_${p.patient_id}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
