import React, { useMemo, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Line } from "react-chartjs-2";
import "chart.js/auto";

// --- Quick notes -------------------------------------------------------------
// 1) Drop this file into a Vite React app as src/PopulationDashboard.jsx.
// 2) Install deps: npm i react-chartjs-2 chart.js jwt-decode
//    (shadcn/ui is assumed available per the canvas environment; if not, swap to plain inputs/divs.)
// 3) Set API_BASE to your FastAPI host (e.g., http://localhost:8000 or http://127.0.0.1:8000).
// 4) The component logs in to FastAPI (/login), stores the JWT, and then calls /predict for
//    each selected model. It renders a multi-series line chart comparing forecasts.
// ----------------------------------------------------------------------------

const API_BASE_DEFAULT = "http://localhost:8000";
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

  return { login, predict, metrics };
}

export default function PopulationDashboard() {
  // --- API config ------------------------------------------------------------
  const [apiBase, setApiBase] = useState(API_BASE_DEFAULT);
  const api = useApi(apiBase);

  // --- Auth ------------------------------------------------------------------
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("changeme");
  const [token, setToken] = useState("");

  // --- Request controls ------------------------------------------------------
  const [geography, setGeography] = useState("US"); // e.g., "US", "01" (AL state), "01001" (county)
  const [startYear, setStartYear] = useState(2012);
  const [endYear, setEndYear] = useState(2025);
  const [selectedModels, setSelectedModels] = useState(ALL_MODELS);

  // --- Results ---------------------------------------------------------------
  const [series, setSeries] = useState([]); // [{label, years[], values[]}]
  const [featuresByModel, setFeaturesByModel] = useState({});
  const [actuals, setActuals] = useState(null); // {years[], population[]}
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  const loggedIn = !!token;
  const years = useMemo(() => {
    const y0 = Number(startYear), y1 = Number(endYear);
    const out = []; for (let y = y0; y <= y1; y++) out.push(y);
    return out;
  }, [startYear, endYear]);

  const fetchActuals = async () => {
    const url = `${apiBase}/actuals?geo=${encodeURIComponent(geography)}&start=${Number(startYear)}&end=${Number(endYear)}`;
    const res = await fetch(url, { headers: { Authorization: `Bearer ${token}` } });
    if (!res.ok) throw new Error(`Actuals failed: ${res.status}`);
    const j = await res.json();
    setActuals(j);
  };

  const doLogin = async () => {
    try {
      setMessage("Logging in…");
      const { access_token } = await api.login(username, password);
      setToken(access_token);
      setMessage("Logged in.");
    } catch (e) {
      setMessage(e.message);
    }
  };

  const toggleModel = (m) => {
    setSelectedModels((prev) =>
      prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]
    );
  };

  const fetchPredictions = async () => {
    if (!loggedIn) { setMessage("Please log in first."); return; }
    setBusy(true); setMessage("");
    try {
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
      await fetchActuals();

      const s = results.map(({ m, res }) => ({
        label: `${m}`,
        years: res.years,
        values: res.forecast,
      }));

      const fx = Object.fromEntries(
        results.map(({ m, res }) => [m, res.features_used || []])
      );

      setSeries(s);
      setFeaturesByModel(fx);
    } catch (e) {
      setMessage(e.message);
    } finally {
      setBusy(false);
    }
  };

  const clearAll = () => {
    setSeries([]);
    setFeaturesByModel({});
    setMessage("");
  };

  const chartData = useMemo(() => {
   const datasets = series.map((s, i) => ({
     label: s.label,
     data: s.values,
     fill: false,
     tension: 0.15,
     pointRadius: 2,
     borderWidth: 2,
   }));
   if (actuals && actuals.years?.length) {
     datasets.unshift({
       label: "actual",
       data: actuals.population,
       borderDash: [6, 4],
       pointRadius: 1,
       borderWidth: 2,
     });
   }
   return {
      labels: years,
     datasets
    };
 }, [years, series, actuals]);

  return (
    <div className="p-6 grid gap-6 max-w-6xl mx-auto">
      <h1 className="text-2xl font-bold">Population Prediction — Front-End (Week 5)</h1>

      {/* API + Auth */}
      <Card className="shadow">
        <CardContent className="p-4 grid md:grid-cols-4 gap-4 items-end">
          <div>
            <Label>API Base</Label>
            <Input value={apiBase} onChange={(e) => setApiBase(e.target.value)} />
          </div>
          <div>
            <Label>Username</Label>
            <Input value={username} onChange={(e) => setUsername(e.target.value)} />
          </div>
          <div>
            <Label>Password</Label>
            <Input type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
          </div>
          <div className="flex gap-2">
            <Button onClick={doLogin} className="w-full">{loggedIn ? "Re-Login" : "Login"}</Button>
          </div>
        </CardContent>
      </Card>

      {/* Controls */}
      <Card className="shadow">
        <CardContent className="p-4 grid md:grid-cols-6 gap-4 items-end">
          <div className="md:col-span-2">
            <Label>Geography</Label>
            <Input placeholder="US | 2-digit FIPS | 5-digit FIPS" value={geography} onChange={(e) => setGeography(e.target.value)} />
          </div>
          <div>
            <Label>Start Year</Label>
            <Input type="number" value={startYear} onChange={(e) => setStartYear(e.target.value)} />
          </div>
          <div>
            <Label>End Year</Label>
            <Input type="number" value={endYear} onChange={(e) => setEndYear(e.target.value)} />
          </div>
          <div className="md:col-span-2">
            <Label>Models</Label>
            <div className="flex flex-wrap gap-2 pt-2">
              {ALL_MODELS.map((m) => (
                <Button key={m} variant={selectedModels.includes(m) ? "default" : "secondary"} size="sm" onClick={() => toggleModel(m)}>
                  {m}
                </Button>
              ))}
            </div>
          </div>
          <div className="md:col-span-6 flex gap-2">
            <Button onClick={fetchPredictions} disabled={busy}>
              {busy ? "Fetching…" : "Fetch Predictions"}
            </Button>
            <Button variant="secondary" onClick={clearAll}>Clear</Button>
          </div>
        </CardContent>
      </Card>

      {/* Chart */}
      <Card className="shadow">
        <CardContent className="p-4">
          <div className="text-sm text-muted-foreground pb-2">
            Compare forecasts for the selected models. Each series is returned by FastAPI /predict.
          </div>
          <Line data={chartData} options={{
            responsive: true,
            plugins: { legend: { position: "bottom" } },
            interaction: { mode: "index", intersect: false },
            scales: { x: { title: { display: true, text: "Year" } }, y: { title: { display: true, text: "Population" } } }
          }} />
        </CardContent>
      </Card>

      {/* Features used per model */}
      {Object.keys(featuresByModel).length > 0 && (
        <Card className="shadow">
          <CardContent className="p-4">
            <h2 className="font-semibold mb-2">Features used (as reported by /predict)</h2>
            <ul className="grid md:grid-cols-2 gap-4">
              {Object.entries(featuresByModel).map(([m, feats]) => (
                <li key={m} className="text-sm">
                  <span className="font-medium">{m}:</span> {feats && feats.length ? feats.join(", ") : "(none reported)"}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {/* Messages */}
      {message && (
        <div className="text-sm text-red-600">{message}</div>
      )}

      {/* Footer hint */}
      <div className="text-xs text-muted-foreground">
        Tip: For the US/State/County pages you sketched, mount this component three times with different presets
        and add a map widget (e.g., topojson + d3-geo or a static SVG) to highlight the selected region.
      </div>
    </div>
  );
}
