const state = {
  strategies: [],
  timeframes: [],
  paramDefaults: {},
  pollTimer: null,
  activeJobId: "",
  bestAIPick: null,
};

const el = {
  apiStatus: document.getElementById("api-status"),
  singleForm: document.getElementById("single-form"),
  singleStrategy: document.getElementById("single-strategy"),
  singleTimeframe: document.getElementById("single-timeframe"),
  singleStart: document.getElementById("single-start"),
  singleEnd: document.getElementById("single-end"),
  singleMaxTokens: document.getElementById("single-max-tokens"),
  singleMinCandles: document.getElementById("single-min-candles"),
  singleAsync: document.getElementById("single-async"),
  singleParams: document.getElementById("single-params"),
  singleAddParam: document.getElementById("single-add-param"),
  singleResetParams: document.getElementById("single-reset-params"),
  compareForm: document.getElementById("compare-form"),
  compareWindowMode: document.getElementById("compare-window-mode"),
  compareStart: document.getElementById("compare-start"),
  compareEnd: document.getElementById("compare-end"),
  compareMaxTokens: document.getElementById("compare-max-tokens"),
  compareMinCandles: document.getElementById("compare-min-candles"),
  compareAsync: document.getElementById("compare-async"),
  compareStrategies: document.getElementById("compare-strategies"),
  compareTimeframes: document.getElementById("compare-timeframes"),
  compareOverrides: document.getElementById("compare-overrides"),
  aiMaxRuns: document.getElementById("ai-max-runs"),
  aiTopN: document.getElementById("ai-top-n"),
  aiMinTrades: document.getElementById("ai-min-trades"),
  aiMaxDd: document.getElementById("ai-max-dd"),
  aiObjective: document.getElementById("ai-objective"),
  aiSeed: document.getElementById("ai-seed"),
  aiForceRefresh: document.getElementById("ai-force-refresh"),
  launchAi: document.getElementById("launch-ai"),
  aiApplyRun: document.getElementById("ai-apply-run"),
  summaryCards: document.getElementById("summary-cards"),
  leaderboardTable: document.getElementById("leaderboard-table"),
  blotterTable: document.getElementById("blotter-table"),
  jobsTable: document.getElementById("jobs-table"),
  equityChart: document.getElementById("equity-chart"),
  refreshJobs: document.getElementById("refresh-jobs"),
  activeJob: document.getElementById("active-job"),
  paramTemplate: document.getElementById("param-row-template"),
};

function setStatus(text, mode) {
  el.apiStatus.textContent = text;
  el.apiStatus.className = `status-badge ${mode}`;
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = body?.detail || body?.error || `${res.status} ${res.statusText}`;
    throw new Error(msg);
  }
  return body;
}

function option(value, label) {
  const o = document.createElement("option");
  o.value = value;
  o.textContent = label;
  return o;
}

function toInputValue(v) {
  if (typeof v === "boolean") return v ? "true" : "false";
  if (v === null || v === undefined) return "";
  return String(v);
}

function parseValue(raw) {
  const t = raw.trim();
  if (t === "") return "";
  if (t === "true") return true;
  if (t === "false") return false;
  if (t === "null") return null;
  if (!Number.isNaN(Number(t)) && t !== "") return Number(t);
  return t;
}

function rowForParam(key = "", value = "") {
  const node = el.paramTemplate.content.firstElementChild.cloneNode(true);
  const keyInput = node.querySelector(".param-key");
  const valueInput = node.querySelector(".param-value");
  const removeBtn = node.querySelector("button");
  keyInput.value = key;
  valueInput.value = toInputValue(value);
  removeBtn.addEventListener("click", () => node.remove());
  return node;
}

function findParamRowByKey(key) {
  const rows = Array.from(el.singleParams.querySelectorAll(".param-row"));
  return rows.find((row) => row.querySelector(".param-key").value.trim() === key) || null;
}

function upsertSingleParam(key, value) {
  let row = findParamRowByKey(key);
  if (!row) {
    row = rowForParam(key, value);
    el.singleParams.appendChild(row);
    return;
  }
  row.querySelector(".param-value").value = toInputValue(value);
}

function renderParamEditor(strategyId) {
  el.singleParams.innerHTML = "";
  const defaults = state.paramDefaults[strategyId] || {};
  const entries = Object.entries(defaults);
  if (!entries.length) {
    el.singleParams.appendChild(rowForParam("", ""));
    return;
  }
  entries.forEach(([k, v]) => el.singleParams.appendChild(rowForParam(k, v)));
}

function renderParamEditorWithOverrides(strategyId, overrides) {
  renderParamEditor(strategyId);
  if (!overrides || typeof overrides !== "object") return;
  Object.entries(overrides).forEach(([k, v]) => upsertSingleParam(k, v));
}

function collectSingleOverrides() {
  const rows = Array.from(el.singleParams.querySelectorAll(".param-row"));
  const overrides = {};
  rows.forEach((row) => {
    const k = row.querySelector(".param-key").value.trim();
    const v = row.querySelector(".param-value").value;
    if (!k) return;
    const parsed = parseValue(v);
    if (parsed === "") return;
    overrides[k] = parsed;
  });
  return overrides;
}

function selectedCompareStrategies() {
  return Array.from(el.compareStrategies.querySelectorAll("input[type=checkbox]:checked")).map((n) => n.value);
}

function selectedCompareTimeframes() {
  return Array.from(el.compareTimeframes.querySelectorAll("input[type=checkbox]:checked")).map((n) => n.value);
}

function money(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "-";
  return `$${Number(v).toFixed(2)}`;
}

function pct(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "-";
  const n = Number(v);
  return `${n >= 0 ? "+" : ""}${n.toFixed(2)}%`;
}

function renderSummary(summary) {
  if (!summary || !summary.performance) {
    el.summaryCards.innerHTML = "";
    return;
  }
  const p = summary.performance;
  const cards = [
    ["Strategy", summary.strategy_name || summary.strategy_id],
    ["Timeframe", summary.timeframe || "-"],
    ["Return", pct(p.return_pct)],
    ["Final Equity", money(p.final_equity)],
    ["PnL", money(p.total_pnl)],
    ["Win Rate", pct(p.win_rate_pct)],
    ["Profit Factor", (p.profit_factor ?? "-").toString()],
    ["Max Drawdown", pct(p.max_drawdown_pct)],
    ["Trades", String(p.trades ?? 0)],
  ];
  el.summaryCards.innerHTML = cards
    .map(([k, v]) => `<div class="kpi"><span class="k">${k}</span><span class="v">${v}</span></div>`)
    .join("");
}

function renderAISummary(summary, topPick) {
  if (!summary) {
    el.summaryCards.innerHTML = "";
    return;
  }
  const cards = [
    ["Mode", "IA Backtest"],
    ["Cached", summary.cached ? "yes" : "no"],
    ["Runs Planned", String(summary.runs_planned ?? "-")],
    ["Runs OK", String(summary.runs_ok ?? "-")],
    ["Quality OK", String(summary.quality_ok ?? "-")],
    ["Objective", summary.objective || "-"],
    ["Top N", String(summary.top_n ?? "-")],
    ["Best Strategy", topPick?.strategy_name || topPick?.strategy_id || "-"],
    ["Best TF", topPick?.timeframe || "-"],
    ["Best Return", pct(topPick?.return_pct)],
  ];
  el.summaryCards.innerHTML = cards
    .map(([k, v]) => `<div class="kpi"><span class="k">${k}</span><span class="v">${v}</span></div>`)
    .join("");
}

function renderTable(table, rows) {
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");
  tbody.innerHTML = "";

  if (!rows || !rows.length) {
    thead.innerHTML = "";
    tbody.innerHTML = `<tr><td>Aucune donnée</td></tr>`;
    return;
  }

  const cols = Object.keys(rows[0]);
  thead.innerHTML = `<tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr>`;

  const body = rows
    .map((row) => {
      const cells = cols
        .map((c) => {
          const val = row[c];
          let cls = "";
          if (typeof val === "number" && /return|drawdown|pnl|pct/i.test(c)) cls = val >= 0 ? "good" : "bad";
          let out = "-";
          if (val !== null && val !== undefined) {
            out = typeof val === "object" ? JSON.stringify(val) : String(val);
          }
          return `<td class="${cls}">${out}</td>`;
        })
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");
  tbody.innerHTML = body;
}

function renderJobsTable(items) {
  const rows = (items || []).map((j) => ({
    job_id: j.job_id,
    kind: j.kind,
    status: j.status,
    progress: `${Math.round((j.progress || 0) * 100)}%`,
    created_at: j.created_at || "-",
    action: `<button class="btn-mini" data-job-open="${j.job_id}">Open</button>`,
  }));
  renderTable(el.jobsTable, rows);

  el.jobsTable.querySelectorAll("[data-job-open]").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      const jobId = e.currentTarget.getAttribute("data-job-open");
      await openJob(jobId);
    });
  });
}

function renderEquityChart(items) {
  if (!items || !items.length) {
    el.equityChart.innerHTML = "<div class='muted'>No equity data</div>";
    return;
  }
  const values = items.map((x) => Number(x.equity_mtm)).filter((x) => Number.isFinite(x));
  if (!values.length) {
    el.equityChart.innerHTML = "<div class='muted'>No equity data</div>";
    return;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = (max - min) * 0.08 || 1;
  const lo = min - pad;
  const hi = max + pad;
  const w = 1000;
  const h = 180;

  const pts = values
    .map((v, i) => {
      const x = (i / Math.max(1, values.length - 1)) * w;
      const y = h - ((v - lo) / (hi - lo)) * h;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");

  el.equityChart.innerHTML = `
    <svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-label="equity chart">
      <defs>
        <linearGradient id="eqStroke" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#ff9d00" />
          <stop offset="100%" stop-color="#23d18b" />
        </linearGradient>
      </defs>
      <polyline fill="none" stroke="url(#eqStroke)" stroke-width="3" points="${pts}" />
    </svg>
  `;
}

function setBestAIPick(pick) {
  state.bestAIPick = pick || null;
  if (!el.aiApplyRun) return;
  const enabled = !!state.bestAIPick;
  el.aiApplyRun.disabled = !enabled;
  el.aiApplyRun.title = enabled ? "Apply best AI config and run detailed single backtest" : "Run IA Backtest first";
}

async function openJob(jobId) {
  if (!jobId) return;
  state.activeJobId = jobId;
  el.activeJob.textContent = `Active job: ${jobId}`;

  try {
    const job = await api(`/api/v1/backtests/${jobId}`);
    if (job.kind === "single" && job.status === "completed") {
      setBestAIPick(null);
      const summary = await api(`/api/v1/backtests/${jobId}/summary`);
      renderSummary(summary.summary);
      const blotter = await api(`/api/v1/backtests/${jobId}/blotter?limit=200`);
      renderTable(el.blotterTable, blotter.items || []);
      const equity = await api(`/api/v1/backtests/${jobId}/equity?limit=500`);
      renderEquityChart(equity.items || []);
      return;
    }

    if (job.kind === "compare" && job.status === "completed") {
      setBestAIPick(null);
      const lb = await api(`/api/v1/backtests/${jobId}/leaderboard?top=100`);
      renderTable(el.leaderboardTable, lb.items || []);
      renderSummary(null);
      renderTable(el.blotterTable, []);
      renderEquityChart([]);
      return;
    }

    if (job.kind === "ai_optimize" && job.status === "completed") {
      let aiItems = [];
      let aiSummary = null;
      try {
        const ai = await api(`/api/v1/backtests/${jobId}/ai-top?top=100`);
        aiItems = ai.items || [];
        aiSummary = ai.summary || null;
      } catch (_err) {
        // Fallback: status payload already contains compact top picks/summary.
        aiItems = job.top_picks || [];
        aiSummary = job.summary || null;
      }
      const topPick = (aiItems || [])[0] || null;
      setBestAIPick(topPick);
      renderTable(el.leaderboardTable, aiItems || []);
      renderAISummary(aiSummary, topPick);
      renderTable(el.blotterTable, []);
      renderEquityChart([]);
      if (!topPick) {
        alert("IA job termine mais aucun top pick n'a ete retourne.");
      }
    }
  } catch (err) {
    alert(`Open job error: ${err.message}`);
  }
}

async function refreshJobs() {
  const jobs = await api("/api/v1/jobs");
  renderJobsTable(jobs.items || []);
}

function startPolling(jobId) {
  state.activeJobId = jobId;
  el.activeJob.textContent = `Active job: ${jobId}`;
  if (state.pollTimer) clearInterval(state.pollTimer);

  state.pollTimer = setInterval(async () => {
    try {
      const job = await api(`/api/v1/backtests/${jobId}`);
      await refreshJobs();
      if (job.status === "completed") {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        await openJob(jobId);
      }
      if (job.status === "failed") {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        alert(`Job failed: ${job.error || "Unknown error"}`);
      }
    } catch (err) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      console.error(err);
    }
  }, 2200);
}

function buildSinglePayload() {
  return {
    strategy_id: el.singleStrategy.value,
    timeframe: el.singleTimeframe.value,
    start: el.singleStart.value || "",
    end: el.singleEnd.value || "",
    max_tokens: Number(el.singleMaxTokens.value || 25),
    min_candles: Number(el.singleMinCandles.value || 200),
    overrides: collectSingleOverrides(),
    run_async: !!el.singleAsync.checked,
  };
}

async function submitSinglePayload(payload) {
  try {
    const data = await api("/api/v1/backtests", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (payload.run_async) {
      startPolling(data.job_id);
      await refreshJobs();
      return;
    }
    renderSummary(data.summary);
    renderTable(el.blotterTable, []);
    renderEquityChart([]);
    renderTable(el.leaderboardTable, []);
  } catch (err) {
    alert(`Single backtest error: ${err.message}`);
  }
}

async function launchSingle(ev) {
  if (ev && typeof ev.preventDefault === "function") ev.preventDefault();
  const payload = buildSinglePayload();
  await submitSinglePayload(payload);
}

function parseCompareOverrides() {
  let parsedOverrides = {};
  const rawOverrides = el.compareOverrides.value.trim();
  if (!rawOverrides) return parsedOverrides;
  try {
    parsedOverrides = JSON.parse(rawOverrides);
    return parsedOverrides;
  } catch {
    alert("JSON invalide dans Shared Overrides");
    return null;
  }
}

async function launchCompare(ev) {
  ev.preventDefault();
  const parsedOverrides = parseCompareOverrides();
  if (parsedOverrides === null) return;

  const payload = {
    strategy_ids: selectedCompareStrategies(),
    timeframes: selectedCompareTimeframes(),
    window_mode: el.compareWindowMode.value,
    start: el.compareStart.value || "",
    end: el.compareEnd.value || "",
    max_tokens: Number(el.compareMaxTokens.value || 25),
    min_candles: Number(el.compareMinCandles.value || 200),
    overrides: parsedOverrides,
    run_async: !!el.compareAsync.checked,
  };

  try {
    const data = await api("/api/v1/backtests/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (payload.run_async) {
      startPolling(data.job_id);
      await refreshJobs();
      return;
    }
    setBestAIPick(null);
    renderTable(el.leaderboardTable, data.leaderboard_top || []);
    renderSummary(null);
    renderTable(el.blotterTable, []);
    renderEquityChart([]);
  } catch (err) {
    alert(`Compare backtest error: ${err.message}`);
  }
}

async function launchAIBacktest() {
  const parsedOverrides = parseCompareOverrides();
  if (parsedOverrides === null) return;

  const payload = {
    strategy_ids: selectedCompareStrategies(),
    timeframes: selectedCompareTimeframes(),
    start: el.compareStart.value || "",
    end: el.compareEnd.value || "",
    max_tokens: Number(el.compareMaxTokens.value || 25),
    min_candles: Number(el.compareMinCandles.value || 200),
    max_runs: Number(el.aiMaxRuns.value || 120),
    top_n: Number(el.aiTopN.value || 5),
    min_trades: Number(el.aiMinTrades.value || 5),
    max_drawdown_pct: Number(el.aiMaxDd.value || 25),
    objective: el.aiObjective.value || "balanced",
    seed: Number(el.aiSeed.value || 42),
    force_refresh: !!el.aiForceRefresh.checked,
    overrides: parsedOverrides,
    run_async: true,
  };

  try {
    setBestAIPick(null);
    const data = await api("/api/v1/backtests/ai-run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    startPolling(data.job_id);
    await refreshJobs();
  } catch (err) {
    alert(`IA backtest error: ${err.message}`);
  }
}

async function applyAndRunBestAI() {
  const pick = state.bestAIPick;
  if (!pick) {
    alert("Aucun best pick IA disponible. Lance d'abord IA Backtest.");
    return;
  }

  el.singleStrategy.value = pick.strategy_id;
  el.singleTimeframe.value = pick.timeframe;
  if (typeof pick.start === "string" && pick.start.length >= 10) {
    el.singleStart.value = pick.start.slice(0, 10);
  }
  if (typeof pick.end === "string" && pick.end.length >= 10) {
    el.singleEnd.value = pick.end.slice(0, 10);
  }
  el.singleMaxTokens.value = String(Number(el.compareMaxTokens.value || 25));
  el.singleMinCandles.value = String(Number(el.compareMinCandles.value || 200));

  const overrides = pick.overrides && typeof pick.overrides === "object" ? pick.overrides : {};
  renderParamEditorWithOverrides(pick.strategy_id, overrides);

  // Force async to fetch detailed outputs once the run completes.
  el.singleAsync.checked = true;
  await submitSinglePayload(buildSinglePayload());
}

function buildComparePickers() {
  el.compareStrategies.innerHTML = "";
  state.strategies.forEach((s) => {
    const wrap = document.createElement("label");
    wrap.className = "chip";
    wrap.innerHTML = `<input type="checkbox" value="${s.id}" checked /> ${s.name}`;
    el.compareStrategies.appendChild(wrap);
  });

  el.compareTimeframes.innerHTML = "";
  state.timeframes.forEach((tf) => {
    const wrap = document.createElement("label");
    wrap.className = "chip";
    wrap.innerHTML = `<input type="checkbox" value="${tf}" checked /> ${tf}`;
    el.compareTimeframes.appendChild(wrap);
  });
}

async function init() {
  try {
    const status = await api("/api/v1/status");
    if (status?.service) setStatus("API Online", "ok");
  } catch {
    setStatus("API Error", "error");
    return;
  }

  try {
    const [strategiesRes, tfRes] = await Promise.all([api("/api/v1/strategies"), api("/api/v1/timeframes")]);
    state.strategies = strategiesRes.items || [];
    state.strategies.forEach((s) => {
      state.paramDefaults[s.id] = s.defaults || {};
      el.singleStrategy.appendChild(option(s.id, `${s.name} (${s.id})`));
    });

    state.timeframes = (tfRes.items || []).filter((x) => x.available).map((x) => x.timeframe);
    if (!state.timeframes.length) state.timeframes = ["15m", "30m", "1h"];
    state.timeframes.forEach((tf) => el.singleTimeframe.appendChild(option(tf, tf)));

    if (state.strategies.length) {
      renderParamEditor(state.strategies[0].id);
    }
    buildComparePickers();
    setBestAIPick(null);
    await refreshJobs();
  } catch (err) {
    setStatus("Boot Error", "error");
    alert(`Init error: ${err.message}`);
  }

  el.singleStrategy.addEventListener("change", () => renderParamEditor(el.singleStrategy.value));
  el.singleAddParam.addEventListener("click", () => el.singleParams.appendChild(rowForParam("", "")));
  el.singleResetParams.addEventListener("click", () => renderParamEditor(el.singleStrategy.value));
  el.singleForm.addEventListener("submit", launchSingle);
  el.compareForm.addEventListener("submit", launchCompare);
  el.launchAi.addEventListener("click", launchAIBacktest);
  if (el.aiApplyRun) {
    el.aiApplyRun.addEventListener("click", applyAndRunBestAI);
  }
  el.refreshJobs.addEventListener("click", refreshJobs);
}

init();
