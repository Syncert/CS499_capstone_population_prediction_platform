import React, { useMemo, useState, useEffect, useRef } from "react";
import { Chart } from "chart.js";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Line } from "react-chartjs-2";
import "./index.css";
import "chart.js/auto";
// in PopulationDashboard.jsx
import GeoPickers from "./GeoPickers.jsx";

// --- Quick notes -------------------------------------------------------------
// 1) Set API_BASE to your FastAPI host (e.g., http://localhost:8000 or http://127.0.0.1:8000).
// 2) The component logs in to FastAPI (/login), stores the JWT, and then calls /predict for
//    each selected model. It renders a multi-series line chart comparing forecasts.
// ----------------------------------------------------------------------------

// === Chart wrapper so Chart.js can fill width reliably ===
function ChartBox({ data, options }) {
  return (
    <div className="chart-box">
      <Line data={data} options={{ responsive: true, maintainAspectRatio: false, ...options }} />
    </div>
  );
}

// Force a resize pass after window or <details> changes
function useForceChartResize() {
  const raf = useRef();

  useEffect(() => {
    const bump = () => {
      cancelAnimationFrame(raf.current);
      raf.current = requestAnimationFrame(() => {
        const instances = Chart.instances ? Object.values(Chart.instances) : [];
        instances.forEach(c => c?.resize?.());
      });
    };

    window.addEventListener("resize", bump);
    const details = Array.from(document.querySelectorAll("details"));
    details.forEach(d => d.addEventListener("toggle", bump));

    return () => {
      window.removeEventListener("resize", bump);
      details.forEach(d => d.removeEventListener("toggle", bump));
      cancelAnimationFrame(raf.current);
    };
  }, []);
}

// const API_BASE_DEFAULT = "http://localhost:8000"; //dev
const API_BASE_DEFAULT = "/api"; //prod, nginx docker environment

const ALL_MODELS = ["linear", "ridge", "xgb", "prophet"]; // toggle per your training

function useApi(baseUrl) {
  const login = async (username, password) => {
    const res = await fetch(`${baseUrl}/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    if (!res.ok) throw new Error(`Login failed: ${res.status}`);
    return res.json(); // { access_token, token_type, expires_in }
  };

  const predict = async (token, { geography, start_year, end_year, model }) => {
    const res = await fetch(`${baseUrl}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ geography, start_year, end_year, model }),
    });
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`Predict ${model} failed: ${res.status} ${detail}`);
    }
    return res.json();
  };

  const metrics = async (token) => {
    const res = await fetch(`${baseUrl}/metrics`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) throw new Error(`Metrics failed: ${res.status}`);
    return res.json();
  };

  const scorecard = async (token, geo) => {
    const r = await fetch(`${baseUrl}/scorecard?geo=${encodeURIComponent(geo)}`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    if (!r.ok) throw new Error(`Scorecard failed: ${r.status}`);
    return r.json(); // { geography, best_model, metrics: [...] }
  };

  const actuals = async (token, geo, start, end) => {
    const r = await fetch(`${baseUrl}/actuals?geo=${encodeURIComponent(geo)}&start=${start}&end=${end}`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    if (!r.ok) throw new Error(`Actuals failed: ${r.status}`);
    return r.json(); // { geography, years, population }
  };

  // inside useApi
  const featuresCompare = async (token, { geo, start, end, model }) => {
    const qs = new URLSearchParams({
      geo: String(geo),
      start: String(start),
      end: String(end),
      model: String(model || "linear"),
    });
    const r = await fetch(`${baseUrl}/features/compare?${qs.toString()}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!r.ok) throw new Error(`features/compare failed: ${r.status}`);
    return r.json();
  };


  return { login, predict, metrics, scorecard, actuals, featuresCompare };
}

export default function PopulationDashboard() {

  useForceChartResize();

  // API/auth
  const [apiBase, setApiBase] = useState(API_BASE_DEFAULT);
  const api = useApi(apiBase);
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("changeme");
  const [token, setToken] = useState("");

  // Controls
  const [geography, setGeography] = useState("US");
  const [startYear, setStartYear] = useState(2012);
  const [endYear, setEndYear] = useState(2030);
  const [selectedModels, setSelectedModels] = useState(["linear", "ridge", "xgb", "prophet"]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  // All-models chart data
  const [series, setSeries] = useState([]); // [{label, years, values}]
  const [featuresByModel, setFeaturesByModel] = useState({});

  // Best-model chart + leaderboard
  const [bestModel, setBestModel] = useState(null);        // "ridge" | "xgb" | ...
  const [bestSeries, setBestSeries] = useState(null);      // { years, values, label }
  const [leaderboard, setLeaderboard] = useState([]);      // rows from /scorecard.metrics

  // for features ACS FRED BLS Charts
  const [featuresBundle, setFeaturesBundle] = useState(null);

  const loggedIn = !!token;
  const years = useMemo(() => {
    const y0 = Number(startYear), y1 = Number(endYear);
    const out = []; for (let y = y0; y <= y1; y++) out.push(y);
    return out;
  }, [startYear, endYear]);

  // login as you already do
  const doLogin = async () => {
    try {
      setMessage("Logging in…");
      const { access_token } = await api.login(username, password);
      setToken(access_token);
      setMessage("Logged in.");
    } catch (e) { setMessage(e.message); }
  };

  const toggleModel = (m) =>
    setSelectedModels((prev) => prev.includes(m) ? prev.filter(x => x !== m) : [...prev, m]);

  // === fetch “Best Model” and leaderboard ===
  const fetchBestAndLeaderboard = async () => {
    if (!loggedIn) return;
    const sc = await api.scorecard(token, geography);
    setBestModel(sc.best_model || null);
    setLeaderboard(sc.metrics || []);

    if (sc.best_model) {
      // best model series + actuals
      const [pred, act] = await Promise.all([
        api.predict(token, {
          geography,
          start_year: Number(startYear),
          end_year: Number(endYear),
          model: sc.best_model,
        }),
        api.actuals(token, geography, Number(startYear), Number(endYear)),
      ]);

      // Align to shared x-axis
      const yrs = years;
      const yhat = yrs.map(y => {
        const i = pred.years.indexOf(y);
        return i >= 0 ? pred.forecast[i] : null;
      });
      const actual = yrs.map(y => {
        const i = act.years.indexOf(y);
        return i >= 0 ? act.population[i] : null;
      });

      setBestSeries({
        label: `${sc.best_model}`,
        years: yrs,
        values: yhat,
        actual,
      });
    } else {
      setBestSeries(null);
    }
  };

  // your existing fetchPredictions, but keep it focused on “all models”
  const fetchPredictions = async () => {
    if (!loggedIn) { setMessage("Please log in first."); return; }
    setBusy(true); setMessage("");
    try {
      // actuals (for the comparison chart baseline)
      const act = await api.actuals(token, geography, Number(startYear), Number(endYear));

      const results = await Promise.all(
        selectedModels.map(async (m) => {
          const res = await api.predict(token, {
            geography,
            start_year: Number(startYear),
            end_year: Number(endYear),
            model: m,
          });
          return { m, res };
        })
      );

      const s = results.map(({ m, res }) => ({
        label: `${m}`,
        years: res.years,
        values: res.forecast,
      }));

      // prepend actuals series for this chart too
      s.unshift({
        label: "actual",
        years: act.years,
        values: act.population,
      });

      setSeries(s);

      const fx = Object.fromEntries(
        results.map(({ m, res }) => [m, res.features_used || []])
      );
      setFeaturesByModel(fx);

      // best model / leaderboard
      await fetchBestAndLeaderboard();
      await fetchFeatures();
    } catch (e) {
      setMessage(e.message);
    } finally {
      setBusy(false);
    }
  };

  // fetch features for ACS FRED BLS Charts
  const fetchFeatures = async () => {
    if (!loggedIn) { setMessage("Please log in first."); return; }
    try {
      const model = bestModel || "linear";
      const data = await api.featuresCompare(token, {
        geo: geography,
        start: Number(startYear),
        end: Number(endYear),
        model,
      });
      setFeaturesBundle(data);
      console.log("features/compare", data);
    } catch (e) {
      console.error(e);
      setMessage(`Features error: ${e.message}`);
    }
  };

  const clearAll = () => {
    setSeries([]);
    setFeaturesByModel({});
    setBestSeries(null);
    setLeaderboard([]);
    setMessage("");
  };

  // chart builders
  const makeChart = (labels, lines) => ({
    labels,
    datasets: lines.map((s) => ({
      label: s.label,
      data: s.values,
      spanGaps: true,
      tension: 0.2,
      pointRadius: 2,
      borderWidth: 2,
      fill: false,
    })),
  });

  const allModelsChart = useMemo(() => {
    const labels = years;
    const aligned = series.map(s => ({
      label: s.label,
      values: labels.map(y => {
        const i = s.years.indexOf(y);
        return i >= 0 ? s.values[i] : null;
      })
    }));
    return makeChart(labels, aligned);
  }, [series, years]);

  const bestChart = useMemo(() => {
    if (!bestSeries) return null;
    const labels = bestSeries.years;
    return makeChart(labels, [
      { label: "actual", values: bestSeries.actual },
      { label: bestSeries.label, values: bestSeries.values },
    ]);
  }, [bestSeries]);

  // Feature charts (ACS, FRED, BLS)
  const FeatureChart = ({ title, s }) => {
    if (!s) return null;
    const labels = s.years;
    const data = {
      labels,
      datasets: [
        { label: `${s.code} (actual)`, data: s.actual, spanGaps: true, tension: 0.2, pointRadius: 2, borderWidth: 2 },
        { label: `${s.code} (projected)`, data: s.projected, spanGaps: true, tension: 0.2, pointRadius: 2, borderDash: [6,4], borderWidth: 2 },
      ],
    };
    return (
      <div className="card">
        <div className="card-body">
          <div className="h3">{title}</div>
          <Line data={data} options={{
            responsive: true,
            plugins: { legend: { position: "bottom" } },
            interaction: { mode: "index", intersect: false },
            scales: { x: { title: { display:true, text:"Year" } } }
          }} />
        </div>
      </div>
    );
  };

  // For downloads
const handleDownloadAllModels = async () => {
  try {
    if (!loggedIn || !geography) return;

    const params = new URLSearchParams({
      geo: String(geography),
      start: String(startYear),
      end: String(endYear),
      model: "all",
      include_future: "1"
    });

    const base = apiBase.trim().replace(/\/+$/, "");
    const url  = `${base}/download/bundle?${params.toString()}`;
    console.log("fetching", url);

    const res = await fetch(url, {
      method: "GET",
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) {
      console.error("download error:", await res.text());
      return;
    }

    const blob = await res.blob();
    const href = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = href;
    a.download = `ppp_${geography}_ALL_${startYear}-${endYear}.zip`; // filename matches "all"
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(href);
  } catch (err) {
    console.error("download exception:", err);
  }
};


  // ─────────────────────────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────────────────────────
  return (
    <div className="container">
      <h1 className="h1">Population Prediction — Front-End</h1>

      {/* CONTROLS (collapsible) */}
      <section className="section">
        <details className="controls" open>
          <summary>Controls</summary>
          <div className="controls-body">
            <div className="card">
              <div className="card-body">
                <div className="toolbar grid-3">
                  <div>
                    <div className="muted">API Base</div>
                    <input value={apiBase} onChange={(e)=>setApiBase(e.target.value)} />
                  </div>
                  <div>
                    <div className="muted">Username</div>
                    <input value={username} onChange={(e)=>setUsername(e.target.value)} />
                  </div>
                  <div>
                    <div className="muted">Password</div>
                    <input type="password" value={password} onChange={(e)=>setPassword(e.target.value)} />
                  </div>
                </div>

                <div style={{height:8}} />

                <div className="toolbar grid-3">
                  <div>
                    <div className="muted">Geography</div>
                    {/* If you wired GeoPickers, drop it here */}
                    {<GeoPickers apiBase={apiBase} onGeoChange={setGeography} />}
                  </div>
                  <div>
                    <div className="muted">Start Year</div>
                    <input type="number" value={startYear} onChange={(e)=>setStartYear(e.target.value)} />
                  </div>
                  <div>
                    <div className="muted">End Year</div>
                    <input type="number" value={endYear} onChange={(e)=>setEndYear(e.target.value)} />
                  </div>
                </div>

                <div style={{height:8}} />

                <div className="toolbar">
                  <button onClick={doLogin} className="badge">
                    {loggedIn ? "Re-Login" : "Login"}
                  </button>
                  <button onClick={fetchPredictions} disabled={busy}>
                    {busy ? "Fetching…" : "Fetch Predictions"}
                  </button>
                  <button onClick={clearAll}>Clear</button>
                  <div className="muted">{message}</div>
                  <button
                      onClick={handleDownloadAllModels}
                      disabled={!bestModel || !loggedIn}
                      title="Download indicators + feature matrix + predictions as a ZIP"
                    >
                      Download data bundle (.zip)
                  </button>
                </div>
              </div>
            </div>
          </div>
        </details>
      </section>

      {/* BEST MODEL PREDICTIONS */}
      <section className="section">
        <div className="h2">Best Model Predictions {bestModel && <span className="badge">{bestModel}</span>}</div>
        <div className="card">
          <div className="card-body">
            {bestChart ? (
            <ChartBox
              data={bestChart}
              options={{
                plugins: { legend: { position: "bottom" } },
                interaction: { mode: "index", intersect: false },
                scales: { x: { title: { display:true, text:"Year" } }, y: { title: { display:true, text:"Population" } } }
              }}
            />
            ) : (
              <div className="muted">Run “Fetch Predictions” to populate the best-model chart.</div>
            )}
          </div>
        </div>
      </section>

      {/* COMPARISON OF ALL MODELS */}
      <section className="section">
        <div className="h2">Comparison of All Model Predictions</div>
        <div className="card">
          <div className="card-body">
            {series.length ? (
            <ChartBox
              data={allModelsChart}
              options={{
                plugins: { legend: { position: "bottom" } },
                interaction: { mode: "index", intersect: false },
                scales: { x: { title: { display:true, text:"Year" } }, y: { title: { display:true, text:"Population" } } }
              }}
            />
            ) : (
              <div className="muted">No series yet. Click “Fetch Predictions”.</div>
            )}
          </div>
        </div>
      </section>

      {/* MODEL LEADERBOARD */}
      <section className="section">
        <div className="h2">Model Leaderboard</div>
        <div className="card">
          <div className="card-body">
            {leaderboard.length ? (
              <table className="table">
                <thead>
                  <tr><th>Rank</th><th>Model</th><th>RMSE (test)</th><th>MAE (test)</th><th>R² (test)</th><th>Trained At</th></tr>
                </thead>
                <tbody>
                  {[...leaderboard]
                    .sort((a,b) => (a.rmse_test ?? 1e9) - (b.rmse_test ?? 1e9))
                    .map(r => (
                      <tr key={r.run_id}>
                        <td>{r.rank_within_geo_code ?? "—"}</td>
                        <td>{r.model}</td>
                        <td>{r.rmse_test != null ? Number(r.rmse_test).toFixed(2) : "—"}</td>
                        <td>{r.mae_test  != null ? Number(r.mae_test).toFixed(2) : "—"}</td>
                        <td>{r.r2_test   != null ? Number(r.r2_test).toFixed(2) : "—"}</td>
                        <td>{r.trained_at?.slice(0,19).replace("T"," ") ?? "—"}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            ) : (
              <div className="muted">Leaderboard will populate after “Fetch Predictions”.</div>
            )}
          </div>
        </div>
      </section>

      {/* FEATURES (ACS/BLS/CPI) */}
      <section className="section">
        <div className="h2">Features</div>
        {!featuresBundle ? (
          <div className="muted">Run “Fetch Predictions” to populate features, or add a button to fetch them.</div>
        ) : (
          <>
            {/* ACS chart: both ACS1 and ACS5 on one plot */}
            {(() => {
            const s1 = featuresBundle.series.find(s => s.code === "ACS1_TOTAL_POP");
            const s5 = featuresBundle.series.find(s => s.code === "ACS5_TOTAL_POP");
            const sp = featuresBundle.series.find(s => s.code === "POPULATION_IMPLIED"); // <-- NEW

            const labels = s1?.years || s5?.years || sp?.years || [];
            const datasets = [
              { label: "ACS1_TOTAL_POP (actual)", data: s1?.actual ?? [], spanGaps: true, tension: 0.2, pointRadius: 2, borderWidth: 2 },
              { label: "ACS5_TOTAL_POP (actual)", data: s5?.actual ?? [], spanGaps: true, tension: 0.2, pointRadius: 2, borderWidth: 2 },
            ];

            if (sp) {
              datasets.push({
                label: "Population (projected)",
                data: sp.projected,
                spanGaps: true,
                tension: 0.2,
                pointRadius: 2,
                borderWidth: 2,
                borderDash: [6, 4],           // dotted
              });
            }

            const data = { labels, datasets };
            return (
              <div className="card">
                <div className="card-body">
                  <div className="h3">ACS Population (1-year vs 5-year)</div>
                  <ChartBox
                    data={data}
                    options={{
                      plugins: { legend: { position: "bottom" } },
                      interaction: { mode: "index", intersect: false },
                      scales: { x: { title: { display:true, text:"Year" } } }
                    }}
                  />
                </div>
              </div>
            );
          })()}

            {/* BLS Unemployment (actual + projected) */}
            <FeatureChart title="BLS Unemployment Rate" s={featuresBundle.series.find(s => s.code==="BLS_UNRATE")} />

            {/* CPI Shelter (actual + projected) */}
            <FeatureChart title="CPI Shelter Index" s={featuresBundle.series.find(s => s.code==="CPI_SHELTER")} />
          </>
        )}
      </section>

    </div>
  );
}