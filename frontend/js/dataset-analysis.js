/**
 * NeuroPredictAI - Dataset Analysis Logic
 * Manages CSV file uploads with progress states, tabular preview generation,
 * column missing-value audits, and correlation heatmaps with real backend data.
 */

document.addEventListener("DOMContentLoaded", async () => {
  setupFileUpload();
  await refreshDatasetView();
});

function setupFileUpload() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      await handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", async () => {
    if (fileInput.files.length > 0) {
      await handleFile(fileInput.files[0]);
    }
  });
}

async function handleFile(file) {
  const statusDiv = document.getElementById("upload-status");
  statusDiv.style.display = "block";
  statusDiv.style.backgroundColor = "#eff6ff";
  statusDiv.style.color = "#1d4ed8";
  statusDiv.style.border = "1px solid #bfdbfe";
  statusDiv.innerHTML = `<span class="loading-spinner" style="vertical-align: middle; margin-right: 8px;"></span> Uploading and analyzing <strong>${file.name}</strong>...`;

  try {
    const result = await api.uploadDataset(file);
    statusDiv.style.backgroundColor = "#ecfdf5";
    statusDiv.style.color = "#047857";
    statusDiv.style.border = "1px solid #a7f3d0";
    statusDiv.innerHTML = `✅ Successfully uploaded <strong>${file.name}</strong> (${result.summary.total_records} records, ${result.summary.total_columns} columns).`;
    await refreshDatasetView();
  } catch (err) {
    statusDiv.style.backgroundColor = "#fef2f2";
    statusDiv.style.color = "#b91c1c";
    statusDiv.style.border = "1px solid #fecaca";
    statusDiv.innerHTML = `❌ Upload failed: ${err.message}`;
  }
}

async function refreshDatasetView() {
  try {
    const [summary, preview, analytics] = await Promise.all([
      api.getDatasetSummary(),
      api.getDatasetPreview(10),
      api.getDatasetAnalytics()
    ]);

    // Update active badge
    document.getElementById("active-filename-text").innerText = summary.filename;

    // Stat cards
    document.getElementById("meta-rows").innerText = summary.total_records.toLocaleString();
    document.getElementById("meta-cols").innerText = summary.total_columns;
    document.getElementById("meta-missing").innerText = summary.missing_summary.total_missing_cells.toLocaleString();
    document.getElementById("meta-duplicates").innerText = summary.duplicate_records;

    // Render Preview Table
    renderPreviewTable(preview);

    // Render Columns Table
    renderColumnsTable(summary.columns);

    // Render Correlation Matrix
    renderCorrelationHeatmap(analytics.correlation_matrix);

  } catch (err) {
    console.error("Error refreshing dataset view:", err);
    document.getElementById("active-filename-text").innerText = "No dataset";
  }
}

function renderPreviewTable(preview) {
  const thead = document.getElementById("preview-header");
  const tbody = document.getElementById("preview-body");

  if (!preview || !preview.columns || preview.columns.length === 0) {
    thead.innerHTML = "<th>No data</th>";
    tbody.innerHTML = "<tr><td>Upload a dataset to see records.</td></tr>";
    return;
  }

  // Headers
  thead.innerHTML = preview.columns.map(col => `<th>${col}</th>`).join("");

  // Rows
  tbody.innerHTML = preview.data.map(row => {
    return `<tr>${preview.columns.map(col => {
      const val = row[col];
      return `<td>${val !== null && val !== undefined ? val : '<span style="color:#94a3b8; font-style:italic;">null</span>'}</td>`;
    }).join("")}</tr>`;
  }).join("");
}

function renderColumnsTable(columns) {
  const tbody = document.getElementById("columns-body");
  if (!columns || columns.length === 0) {
    tbody.innerHTML = "<tr><td colspan='7'>No columns found.</td></tr>";
    return;
  }

  tbody.innerHTML = columns.map(col => {
    let roleBadge = `<span style="background:#f1f5f9; color:#475569; padding:2px 8px; border-radius:4px; font-size:0.75rem;">Feature</span>`;
    if (col.is_target_candidate) {
      roleBadge = `<span style="background:#ecfdf5; color:#047857; font-weight:600; padding:2px 8px; border-radius:4px; font-size:0.75rem;">Target</span>`;
    } else if (col.is_identifier) {
      roleBadge = `<span style="background:#fef2f2; color:#b91c1c; padding:2px 8px; border-radius:4px; font-size:0.75rem;">ID (Ignored)</span>`;
    }

    return `
      <tr>
        <td><strong>${col.name}</strong></td>
        <td><span style="text-transform:capitalize;">${col.type}</span></td>
        <td><code>${col.dtype}</code></td>
        <td>${col.missing_count}</td>
        <td>${col.missing_percentage}%</td>
        <td>${col.unique_values}</td>
        <td>${roleBadge}</td>
      </tr>
    `;
  }).join("");
}

function renderCorrelationHeatmap(corr) {
  const container = document.getElementById("correlation-container");
  if (!corr || !corr.features || corr.features.length === 0) {
    container.innerHTML = `<p style="text-align: center; color: #64748b;">Not enough numerical features to render correlation heatmap.</p>`;
    return;
  }

  const { features, values } = corr;

  let html = `<table class="heatmap-table"><thead><tr><th></th>`;
  features.forEach(f => {
    html += `<th>${f}</th>`;
  });
  html += `</tr></thead><tbody>`;

  values.forEach((row, i) => {
    html += `<tr><th>${features[i]}</th>`;
    row.forEach(val => {
      // Calculate background color based on correlation intensity (-1 to 1)
      let bg = "#ffffff";
      let textCol = "#0f172a";
      if (val > 0) {
        const alpha = Math.min(Math.abs(val), 1.0);
        bg = `rgba(6, 182, 212, ${alpha * 0.75})`;
        if (alpha > 0.6) textCol = "#ffffff";
      } else if (val < 0) {
        const alpha = Math.min(Math.abs(val), 1.0);
        bg = `rgba(239, 68, 68, ${alpha * 0.75})`;
        if (alpha > 0.6) textCol = "#ffffff";
      }
      html += `<td style="background-color:${bg}; color:${textCol}; font-weight:600;">${val.toFixed(2)}</td>`;
    });
    html += `</tr>`;
  });

  html += `</tbody></table>`;
  container.innerHTML = html;
}
