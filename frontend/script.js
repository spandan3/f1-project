// Point this at your FastAPI backend
const BASE_URL = "http://127.0.0.1:8000/predict";

const qs = (id) => document.getElementById(id);

function setHidden(el, hidden) {
  if (el) el.hidden = hidden;
}

function showStatus(message, type = "ok") {
  const bar = qs("status-bar");
  bar.textContent = message;
  bar.className = "status-bar " + type;
  bar.hidden = false;
}

function clearStatus() {
  const bar = qs("status-bar");
  bar.hidden = true;
}

function setLoading(isLoading) {
  qs("loading").hidden = !isLoading;
  qs("submit-btn").disabled = isLoading;
}

function resetUI() {
  qs("predictions-body").innerHTML = "";
  qs("metrics-body").innerHTML = "";
  qs("race-meta").textContent = "";

  setHidden(qs("predictions-wrapper"), true);
  setHidden(qs("metrics-wrapper"), true);

  qs("predictions-empty").textContent = "Run a prediction to see results.";
  qs("metrics-empty").textContent =
    "Metrics (NDCG, Top-3 hit, Spearman, RMSE if present) will appear here.";
  clearStatus();
}

function renderPredictions(preds) {
  const body = qs("predictions-body");
  if (!preds || preds.length === 0) {
    qs("predictions-empty").textContent = "No predictions returned.";
    return;
  }

  // Sort by predicted rank if present
  const sorted = [...preds].sort((a, b) => {
    const pa = a.pred_rank ?? a.predicted_position ?? 9999;
    const pb = b.pred_rank ?? b.predicted_position ?? 9999;
    return pa - pb;
  });

  const first = sorted[0];
  const name = first.event_name || first.race_id || "";
  const year = first.event_year != null ? ` (${first.event_year})` : "";
  qs("race-meta").textContent = name + year;

  for (const row of sorted) {
    const tr = document.createElement("tr");

    const predPos = row.pred_rank ?? row.predicted_position ?? "";
    const driver = row.Driver || row.driver || row.driver_name || "";
    const team = row.TeamName || row.team || "";
    const grid = row.grid_pos ?? row.grid ?? "";
    const score = row.score != null ? row.score.toFixed(4) : "";
    const race = row.event_name || row.race_id || "";
    const yearVal = row.event_year ?? "";

    tr.innerHTML = `
      <td>${predPos}</td>
      <td>${driver}</td>
      <td>${team}</td>
      <td>${grid}</td>
      <td>${score}</td>
      <td>${race}</td>
      <td>${yearVal}</td>
    `;
    body.appendChild(tr);
  }

  setHidden(qs("predictions-wrapper"), false);
  qs("predictions-empty").textContent = "";
}

function renderMetrics(metrics) {
  if (!metrics) {
    qs("metrics-empty").textContent = "Backend returned no metrics.";
    return;
  }

  const body = qs("metrics-body");
  for (const [name, val] of Object.entries(metrics)) {
    const tr = document.createElement("tr");
    const formatted =
      typeof val === "number" ? val.toFixed(4) : String(val ?? "");
    tr.innerHTML = `<td>${name}</td><td>${formatted}</td>`;
    body.appendChild(tr);
  }

  setHidden(qs("metrics-wrapper"), false);
  qs("metrics-empty").textContent = "";
}

qs("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  resetUI();
  setLoading(true);

  const year = qs("year").value.trim();
  const raceId = qs("race-id").value.trim();

  if (!year) {
    showStatus("Please enter a year.", "error");
    setLoading(false);
    return;
  }

  const url =
    BASE_URL +
    `?year=${encodeURIComponent(year)}` +
    (raceId ? `&race_id=${encodeURIComponent(raceId)}` : "");

  try {
    const res = await fetch(url);
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text}`);
    }
    const data = await res.json();

    renderPredictions(data.predictions || []);
    renderMetrics(data.metrics || null);
    showStatus("Prediction completed successfully.", "ok");
  } catch (err) {
    console.error(err);
    showStatus("Error contacting backend: " + err.message, "error");
  } finally {
    setLoading(false);
  }
});

qs("reset-btn").addEventListener("click", () => {
  qs("predict-form").reset();
  resetUI();
});
