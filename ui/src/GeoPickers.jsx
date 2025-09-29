// src/GeoPickers.jsx
import React, { useEffect, useMemo, useState } from "react";

export default function GeoPickers({ apiBase, onGeoChange }) {
  const [nations, setNations] = useState([]);
  const [states, setStates]   = useState([]);
  const [counties, setCounties] = useState([]);

  const [nation, setNation]   = useState("US"); // default
  const [stateFips, setStateFips] = useState(""); // '06'
  const [countyFips, setCountyFips] = useState(""); // '06037'

  // Load nation + states once
  useEffect(() => {
    (async () => {
      const n = await fetch(`${apiBase}/geos/nations`).then(r => r.json());
      setNations(n);
      const s = await fetch(`${apiBase}/geos/states`).then(r => r.json());
      setStates(s);
    })();
  }, [apiBase]);

  // When state changes, load its counties
  useEffect(() => {
    setCountyFips("");
    setCounties([]);
    if (!stateFips) return;
    (async () => {
      const c = await fetch(`${apiBase}/geos/states/${stateFips}/counties`).then(r => r.json());
      setCounties(c);
    })();
  }, [apiBase, stateFips]);

  // Compute the effective geo_code:
  // county > state > nation
  const effectiveGeo = useMemo(() => {
    if (countyFips) return countyFips;
    if (stateFips)  return stateFips;
    return nation || "US";
  }, [nation, stateFips, countyFips]);

  // Emit to parent whenever it changes
  useEffect(() => {
    onGeoChange?.(effectiveGeo);
  }, [effectiveGeo, onGeoChange]);

  return (
    <div className="flex flex-wrap gap-3 items-end">
      {/* Nation (fixed to US, but future-proofed) */}
      <div>
        <label className="block text-sm mb-1">Nation</label>
        <select
          value={nation}
          onChange={(e) => setNation(e.target.value)}
          className="border rounded px-2 py-1 min-w-[10rem]"
        >
          {nations.map(n => (
            <option key={n.geo_code} value={n.geo_code}>
              {n.geo_name} ({n.geo_code})
            </option>
          ))}
        </select>
      </div>

      {/* State */}
      <div>
        <label className="block text-sm mb-1">State</label>
        <select
          value={stateFips}
          onChange={(e) => setStateFips(e.target.value)}
          className="border rounded px-2 py-1 min-w-[14rem]"
        >
          <option value="">— All States —</option>
          {states.map(s => (
            <option key={s.geo_code} value={s.geo_code}>
              {s.geo_name} ({s.geo_code})
            </option>
          ))}
        </select>
      </div>

      {/* County (enabled only when a state is picked) */}
      <div>
        <label className="block text-sm mb-1">County</label>
        <select
          value={countyFips}
          onChange={(e) => setCountyFips(e.target.value)}
          disabled={!stateFips}
          className="border rounded px-2 py-1 min-w-[18rem] disabled:opacity-60"
        >
          <option value="">— All Counties —</option>
          {counties.map(c => (
            <option key={c.geo_code} value={c.geo_code}>
              {c.geo_name} ({c.geo_code})
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}