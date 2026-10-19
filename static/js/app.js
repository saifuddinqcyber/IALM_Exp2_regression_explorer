const $ = (id) => document.getElementById(id);
const colors = ["#6366f1", "#0d9488", "#e87943"];
const datasetFeatures = {
  study: "Study hours",
  advertising: "Advertising spend (units)",
  energy: "Temperature index",
  diabetes: "BMI (scaled)"
};
function updateInputFeature() {
  const feature = datasetFeatures[$("dataset").value];
  $("input-feature").innerHTML = `<option value="${feature}">${feature}</option>`;
}

function fmt(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "—";
}

async function loadDatasetPreview() {
  const key = $("dataset").value;
  const body = $("dataset-preview");
  body.innerHTML = '<tr><td colspan="2">Loading dataset preview…</td></tr>';
  try {
    const response = await fetch(`/api/dataset/${encodeURIComponent(key)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not load dataset preview.");
    $("preview-x-heading").textContent = data.x_name;
    $("preview-y-heading").textContent = data.y_name;
    $("preview-description").textContent = `First ${data.rows.length} rows of ${data.name}`;
    $("preview-count").textContent = `${data.observations} ROWS`;
    body.innerHTML = data.rows.map((row, index) => `<tr><td>${fmt(row.x)}</td><td>${fmt(row.y)}</td></tr>`).join("");
  } catch (error) {
    body.innerHTML = `<tr><td colspan="2">${error.message}</td></tr>`;
  }
}

function plotLayout(xTitle, yTitle) {
  return {
    autosize: true,
    margin: {l: 58, r: 18, t: 12, b: 55},
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    font: {family: "Inter, Segoe UI, sans-serif", size: 11, color: "#748096"},
    xaxis: {title: {text: xTitle, standoff: 12}, gridcolor: "#e9edf4", zerolinecolor: "#e9edf4", automargin: true},
    yaxis: {title: {text: yTitle, standoff: 12}, gridcolor: "#e9edf4", zerolinecolor: "#e9edf4", automargin: true},
    showlegend: false
  };
}

const plotConfig = {responsive: true, displayModeBar: false};

function renderFit(result) {
  const fit = result.fit;
  Plotly.newPlot("fit-chart", [
    {x: fit.x, y: fit.y, type: "scatter", mode: "markers", name: "Observed values",
      marker: {color: "#818cf8", size: 7, opacity: .82}, hovertemplate: "X: %{x:.3f}<br>Y: %{y:.3f}<extra>Observed</extra>"},
    {x: fit.curve_x, y: fit.curve_y, type: "scatter", mode: "lines", name: "Best model fit",
      line: {color: "#e87943", width: 3}, hovertemplate: "X: %{x:.3f}<br>Fit: %{y:.3f}<extra>Fit</extra>"}
  ], plotLayout(result.dataset.x_name, result.dataset.y_name), plotConfig);
}

function renderPrediction(result) {
  const actual = result.test.actual;
  const predicted = result.test.predicted;
  const low = Math.min(...actual, ...predicted);
  const high = Math.max(...actual, ...predicted);
  const pad = Math.max((high-low)*.1, 1);
  Plotly.newPlot("prediction-chart", [
    {x: [low-pad, high+pad], y: [low-pad, high+pad], type: "scatter", mode: "lines",
      line: {color: "#94a3b8", dash: "dash", width: 2}, hoverinfo: "skip"},
    {x: actual, y: predicted, type: "scatter", mode: "markers",
      marker: {color: "#6366f1", size: 9, opacity: .82},
      hovertemplate: "Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>"}
  ], plotLayout("Actual values", "Predicted values"), plotConfig);
}

function renderResiduals(result) {
  const actual = result.test.actual;
  const residuals = result.test.residuals;
  Plotly.newPlot("residual-chart", [
    {x: actual, y: actual.map(() => 0), type: "scatter", mode: "lines",
      line: {color: "#94a3b8", dash: "dash", width: 2}, hoverinfo: "skip"},
    {x: actual, y: residuals, type: "scatter", mode: "markers",
      marker: {color: "#0d9488", size: 9, opacity: .85},
      hovertemplate: "Actual: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>"}
  ], plotLayout("Actual values", "Residual (actual − predicted)"), plotConfig);
}

function renderTable(metrics, bestModel) {
  $("metrics-table").innerHTML = metrics.map((m) => `
    <tr class="${m.model === bestModel ? "best-row" : ""}">
      <td>${m.model}</td><td>${fmt(m.mae)}</td><td>${fmt(m.mse)}</td><td>${fmt(m.rmse)}</td>
      <td>${fmt(m.r2)}</td><td>${m.model === bestModel ? '<span class="best-label">Best RMSE</span>' : '<span class="muted">Compared</span>'}</td>
    </tr>`).join("");
}

function renderBars(metrics) {
  const metricDefs = [
    {key: "mae", label: "MAE", lower: true},
    {key: "mse", label: "MSE", lower: true},
    {key: "rmse", label: "RMSE", lower: true},
    {key: "r2", label: "R²", lower: false}
  ];
  $("performance-bars").innerHTML = metricDefs.map((metric) => {
    const values = metrics.map(m => m[metric.key]);
    const min = Math.min(...values), max = Math.max(...values);
    return `<div><div class="metric-bars-title">${metric.label}</div>${metrics.map((m, i) => {
      let score;
      if (max === min) score = 100;
      else if (metric.lower) score = 100 * (max - m[metric.key]) / (max - min);
      else score = 100 * (m[metric.key] - min) / (max - min);
      return `<div class="bar-row"><span class="bar-label">${m.model}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.max(0,Math.min(100,score))}%;background:${colors[i % colors.length]}"></div></div><span class="bar-value">${fmt(m[metric.key])}</span></div>`;
    }).join("")}</div>`;
  }).join("");
}

function renderResult(result) {
  const best = result.metrics.find(m => m.model === result.best_model);
  $("mae").textContent = fmt(best.mae);
  $("mse").textContent = fmt(best.mse);
  $("rmse").textContent = fmt(best.rmse);
  $("r2").textContent = fmt(best.r2);
  $("best-model-pill").textContent = result.best_model.toUpperCase();
  $("fit-description").textContent = `${result.dataset.x_name} vs ${result.dataset.y_name} · Best-performing fitted model`;
  $("observation-count").textContent = result.dataset.observations;
  $("input-name").textContent = result.dataset.x_name;
  $("target-name").textContent = result.dataset.y_name;
  renderFit(result);
  renderPrediction(result);
  renderResiduals(result);
  renderTable(result.metrics, result.best_model);
  renderBars(result.metrics);
  $("conclusion").innerHTML = `<strong>${result.best_model}</strong> has the lowest RMSE (${fmt(best.rmse)}) on the held-out test set for <strong>${result.dataset.name}</strong>. Its MAE is ${fmt(best.mae)}, MSE is ${fmt(best.mse)}, and R² is ${fmt(best.r2)}. Lower MAE, MSE, and RMSE indicate smaller prediction errors; a higher R² indicates that the model explains more variation in the target. Confirm the metrics against the actual-versus-predicted and residual plots before selecting a model. ${result.dataset.description}`;
  $("status").textContent = `Analysis complete · ${result.dataset.observations} observations · ${(result.test_size*100).toFixed(0)}% test split`;
  if (window.Plotly) {
    ["fit-chart", "prediction-chart", "residual-chart"].forEach(id => {
      const chart = $(id);
      if (chart && chart.data) Plotly.Plots.resize(chart);
    });
  }
}

async function runAnalysis() {
  const button = $("run-analysis");
  button.disabled = true;
  button.textContent = "Training models…";
  $("status").textContent = "Training and evaluating regression models…";
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({dataset: $("dataset").value, test_size: $("test-split").value, regression_method: $("regression-method").value, input_feature: $("input-feature").value})
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Could not complete analysis.");
    renderResult(result);
  } catch (error) {
    $("status").textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = "Run regression";
  }
}

updateInputFeature();
loadDatasetPreview();
$("run-analysis").addEventListener("click", runAnalysis);
$("dataset").addEventListener("change", () => { updateInputFeature(); loadDatasetPreview(); runAnalysis(); });
$("regression-method").addEventListener("change", runAnalysis);
$("input-feature").addEventListener("change", runAnalysis);
$("test-split").addEventListener("change", runAnalysis);
$("reset-analysis").addEventListener("click", () => {
  $("dataset").value = "study";
  $("test-split").value = "0.2";
  $("regression-method").value = "Best performing model";
  updateInputFeature();
  runAnalysis();
});
runAnalysis();

document.querySelectorAll(".nav-button").forEach((link) => {
  link.addEventListener("click", () => {
    document.querySelectorAll(".nav-button").forEach((item) => item.classList.remove("active"));
    link.classList.add("active");
  });
});
