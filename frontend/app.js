/* GeoFlare frontend: India map + NASA FIRMS thermal anomalies + AI fire detection */

const FIRMS_REFRESH_MS = 120000;

const state = {
  map: null,
  baseLayers: {},
  indiaLayer: null,
  firesLayer: null,
  detectionsLayer: null,
  gisVisible: true,
  markers: [],
  refreshTimer: null,
  currentFire: null,
  serverConfig: null,
};

const clsMeta = {
  agricultural: { color: '#f59e0b', emoji: '🌾', label: 'Agricultural' },
  forest:       { color: '#16a34a', emoji: '🌲', label: 'Forest' },
  industrial:   { color: '#dc2626', emoji: '🏭', label: 'Industrial' },
  Unclassified: { color: '#9ca3af', emoji: '❔', label: 'Unclassified' },
};

const els = {
  statusDot: document.getElementById('statusDot'),
  statusText: document.getElementById('statusText'),
  firesCount: document.getElementById('firesCount'),
  btnRefresh: document.getElementById('btnRefresh'),
  btnSettings: document.getElementById('btnSettings'),
  settingsModal: document.getElementById('settingsModal'),
  dataSource: document.getElementById('dataSource'),
  serverConfigNote: document.getElementById('serverConfigNote'),
  resultPanel: document.getElementById('resultPanel'),
  panelContent: document.getElementById('panelContent'),
  panelClose: document.getElementById('panelClose'),
  toast: document.getElementById('toast'),
};

/* ---------------- boot ---------------- */

async function init() {
  initMap();
  await Promise.all([
    loadIndiaBoundary(),
    loadServerConfig(),
  ]);
  await refreshFires();
  state.refreshTimer = setInterval(refreshFires, FIRMS_REFRESH_MS);
  buildStackUI();
  await loadDetections();
  bindEvents();
}

function initMap() {
  state.map = L.map('map', { zoomControl: true, attributionControl: true }).setView([22.5, 79.0], 5);

  state.baseLayers = {
    satellite: L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 18,
      attribution: 'Tiles &copy; Esri',
    }),
    dark: L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap &copy; CARTO',
    }),
    light: L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap &copy; CARTO',
    }),
  };
  state.baseLayers.satellite.addTo(state.map);

  state.firesLayer = L.layerGroup().addTo(state.map);
  state.detectionsLayer = L.layerGroup().addTo(state.map);
}

/* ---------------- GIS detection overlay ---------------- */

function buildStackUI() {
  const ctrl = document.getElementById('layerControl');
  if (ctrl) {
    const btn = document.createElement('button');
    btn.className = 'lc-btn active';
    btn.textContent = ' Classified detections';
    btn.title = 'Toggle the saved fire events overlay (from GIS storage)';
    btn.addEventListener('click', () => {
      state.gisVisible = !state.gisVisible;
      btn.classList.toggle('active', state.gisVisible);
      const ov = document.getElementById('detOverlay');
      if (ov) ov.style.display = state.gisVisible ? '' : 'none';
      if (state.gisVisible && state.detectionsLayer.getLayers().length === 0) loadDetections();
    });
    ctrl.appendChild(btn);
  }
  const leg = document.getElementById('legend');
  if (leg) {
    const ov = document.createElement('div');
    ov.id = 'detOverlay';
    ov.innerHTML = `
      <div class="legend-title" style="margin-top:8px;border-top:1px solid rgba(255,255,255,.14);padding-top:8px">Classified detections (GIS· PostGIS)</div>
      <div class="legend-row"><span class="swatch" style="background:#16a34a"></span><span>Forest</span></div>
      <div class="legend-row"><span class="swatch" style="background:#dc2626"></span><span>Industrial</span></div>
      <div class="legend-row"><span class="swatch" style="background:#f59e0b"></span><span>Agricultural</span></div>
      <div class="legend-row"><span class="swatch" style="background:#9ca3af"></span><span>Unclassified</span></div>`;
    leg.appendChild(ov);
  }
}

async function loadDetections() {
  try {
    const res = await fetch('/api/detections?limit=200');
    if (!res.ok) throw new Error('detections endpoint failed');
    const fc = await res.json();
    renderDetections(fc.features || []);
  } catch (e) {
    console.warn('GIS detections overlay unavailable', e);
  }
}

function renderDetections(features) {
  state.detectionsLayer.clearLayers();
  const use = state.gisVisible ? features : [];
  use.forEach(f => {
    const p = f.properties || {};
    const geo = f.geometry;
    if (!geo || geo.type !== 'Point') return;
    const coord = geo.coordinates;
    const label = p.classification || 'Unclassified';
    const meta = clsMeta[String(label)] || clsMeta.Unclassified;
    const m = L.circleMarker([coord[1], coord[0]], {
      radius: 10,
      color: '#ffffff',
      weight: 1.5,
      fillColor: meta.color,
      fillOpacity: 0.9,
      className: 'det-marker',
    });
    m.bindPopup(`
      <div class="popup-title">Classified fire event</div>
      <div class="popup-line">${meta.emoji} <b>${meta.label}</b> · ${((p.confidence || 0) * 100).toFixed(0)}%</div>
      <div class="popup-line">📍 ${coord[1].toFixed(4)}, ${coord[0].toFixed(4)}</div>
      <div class="popup-line">🕐 ${formatIso(p.created_at)}</div>
      ${p.temperature != null ? `<div class="popup-line">🌡 ${(+p.temperature).toFixed(1)} °C · RH ${p.humidity != null ? (+p.humidity).toFixed(0) + ' %' : '—'}</div>` : ''}
      ${p.source ? `<div class="popup-line">source: ${p.source}</div>` : ''}
    `);
    m.addTo(state.detectionsLayer);
  });
}

function formatIso(iso) {
  try {
    return new Date(iso).toLocaleString('en-IN', { hour12: false });
  } catch { return iso || '—'; }
}

/* ---------------- India boundary ---------------- */

async function loadIndiaBoundary() {
  try {
    const g = await fetch('india.geojson').then(r => r.json());
    state.indiaLayer = L.geoJSON(g, {
      style: {
        color: '#7aa2ff',
        weight: 1.2,
        opacity: 0.65,
        fillColor: '#12203c',
        fillOpacity: 0.18,
      },
    }).addTo(state.map);
  } catch (e) {
    console.warn('Could not load India boundary overlay', e);
  }
}

/* ---------------- FIRMS fires ---------------- */

async function refreshFires() {
  setStatus('loading', 'fetching data…');
  try {
    const source = getSetting('dataSource') || (state.serverConfig && state.serverConfig.default_source) || 'demo';
    const url = source === 'demo'
      ? '/api/fires?demo=1'
      : '/api/fires?days=2';
    const res = await fetch(url);
    const data = await res.json();

    if (data.source === 'error') {
      throw new Error(data.error || 'FIRMS error');
    }

    renderFires(data.fires || []);
    els.firesCount.textContent = `${data.count} anomalies`;
    setStatus('live', data.source === 'demo' ? 'demo data' : 'NASA FIRMS live');
  } catch (e) {
    renderFires([]);
    setStatus('err', 'data unavailable');
    toast(e.message || 'Failed to load FIRMS data', true);
  }
}

function frpColor(frp) {
  if (frp == null || frp < 20) return '#22c55e';
  if (frp < 50) return '#eab308';
  if (frp < 100) return '#fb923c';
  return '#ef4444';
}

function renderFires(fires) {
  state.firesLayer.clearLayers();
  state.markers = [];

  fires.forEach((f, i) => {
    const r = Math.min(6 + (f.frp || 10) / 7, 20);
    const marker = L.circleMarker([f.latitude, f.longitude], {
      radius: r,
      color: '#ffffff',
      weight: 1,
      fillColor: frpColor(f.frp),
      fillOpacity: 0.85,
      className: 'fires-marker' + ((f.frp || 0) >= 50 ? ' pulse' : ''),
    });

    const dt = formatAq(f.acq_date, f.acq_time);
    marker.bindPopup(`
      <div class="popup-title">Thermal Anomaly</div>
      <div class="popup-line">📍 ${f.latitude.toFixed(4)}, ${f.longitude.toFixed(4)}</div>
      <div class="popup-line">🕐 ${dt}</div>
      <div class="popup-line">🔥 FRP: <b>${f.frp ?? '—'}</b>  Brightness: ${f.brightness ?? '—'}K</div>
      <div class="popup-line">🛰 ${f.satellite ?? ''} ${f.instrument ?? ''} · ${f.daynight === 'D' ? 'Day' : 'Night'}</div>
      <div class="popup-line">✓ Confidence: ${f.confidence ?? '—'}</div>
      <button class="btn btn-primary popup-cta" onclick="analyzeFire(${i}, ${f.latitude}, ${f.longitude}, ${f.utc || 0})">Run AI Fire Analysis</button>
    `);
    marker.on('click', () => analyzeFire(null, f.latitude, f.longitude, f.utc || 0));
    marker.addTo(state.firesLayer);
    state.markers.push(marker);
  });
}

/* ---------------- AI analysis ---------------- */

async function analyzeFire(id, lat, lon, utc) {
  openPanel(`
    <div class="spinner-wrap">
      <div class="spinner"></div>
      <p>Fetching weather at 📍 ${lat.toFixed(4)}, ${lon.toFixed(4)}<br>Running AI model…</p>
    </div>
  `);

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lat, lon, utc }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Prediction failed');
    renderResult(data);
    if (data.classification) renderClassificationStrip(data.classification);
    loadDetections();
  } catch (e) {
    openPanel(`<div class="error-box">⚠ ${e.message}</div>`);
  }
}

function renderResult(d) {
  const fire = d.prediction === 1;
  const conf = d.confidence;
  const w = d.weather || {};
  const aq = d.air_quality || {};

  const gaugeColor = fire ? `url(#fireGrad)` : `url(#safeGrad)`;
  const c = 2 * Math.PI * 46;
  const offset = c - (conf * c);

  const aqiClass = aq.aqi == null ? '' : aq.aqi <= 50 ? 'aqi-good' : aq.aqi <= 100 ? 'aqi-mod' : 'aqi-bad';
  const liveFeats = getLiveFeatureKeys(d.features_used);

  const content = `
  <div class="result-scroll">
    <div class="coord-head">
      <div>
        <h2>${fire ? '🔥 Fire Likely' : '🌿 No Fire Expected'}</h2>
        <div class="coord-sub">Satellite thermal anomaly · ${d.coordinate.utc ? formatEpoch(d.coordinate.utc) : ''}</div>
        <div class="source-tag">${aq.pm2_5 == null ? 'Weather: median-estimated' : 'Live weather embedded'}</div>
      </div>
      <div class="coords">${d.coordinate.latitude.toFixed(3)},<br>${d.coordinate.longitude.toFixed(3)}</div>
    </div>

    <div class="verdict ${fire ? 'fire' : 'safe'}">
      <h3>${d.message}</h3>
      <p>predicted by AI with ${(conf * 100).toFixed(1)}% confidence</p>
    </div>

    <div class="conf-section">
      <div class="gauge">
        <svg width="108" height="108" viewBox="0 0 108 108">
          <defs>
            <linearGradient id="fireGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stop-color="#ff8726"/><stop offset="100%" stop-color="#ff3d3d"/>
            </linearGradient>
            <linearGradient id="safeGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stop-color="#34d399"/><stop offset="100%" stop-color="#22c55e"/>
            </linearGradient>
          </defs>
          <circle class="bg" cx="54" cy="54" r="46"/>
          <circle class="fg" cx="54" cy="54" r="46" stroke="${gaugeColor}" stroke-dasharray="${c}" stroke-dashoffset="${offset}"/>
        </svg>
        <div class="gauge-text"><div><div class="val">${(conf * 100).toFixed(0)}%</div><div class="lbl">confidence</div></div></div>
      </div>
      <div class="prob-bars">
        <div class="prob-row">
          <div class="top"><span>🔥 Fire</span><span>${(d.probability.fire * 100).toFixed(1)}%</span></div>
          <div class="prob-track"><div class="prob-fill firef" style="width:${(d.probability.fire * 100).toFixed(1)}%"></div></div>
        </div>
        <div class="prob-row">
          <div class="top"><span>🌿 No Fire</span><span>${(d.probability.no_fire * 100).toFixed(1)}%</span></div>
          <div class="prob-track"><div class="prob-fill safef" style="width:${(d.probability.no_fire * 100).toFixed(1)}%"></div></div>
        </div>
      </div>
    </div>

    <div class="section-title">Weather at location</div>
    <div class="weather-grid">
      ${weatherCard('Temperature', w.temperature_c != null ? w.temperature_c + ' °C' : '—')}
      ${weatherCard('Humidity', w.humidity_pct != null ? w.humidity_pct + ' %' : '—')}
      ${weatherCard('Pressure', w.pressure_hpa != null ? w.pressure_hpa + ' hPa' : '—')}
      ${weatherCard('Wind', w.wind_speed_kmh != null ? w.wind_speed_kmh + ' km/h' : '—')}
      ${weatherCard('Cloud', w.cloud_cover_pct != null ? w.cloud_cover_pct + ' %' : '—')}
      ${weatherCard('Condition', w.weather_text || '—')}
      <div class="w-card aqi-card"><div><div class="k">Air Quality (US AQI)</div><div class="v">PM2.5 ${aq.pm2_5 ?? '—'} · PM10 ${aq.pm10 ?? '—'}</div></div><span class="aqi-badge ${aqiClass}">${aq.aqi ?? '—'}</span></div>
    </div>

    <div class="section-title" style="cursor:pointer" onclick="document.getElementById('featureGrid').hidden = !document.getElementById('featureGrid').hidden">
      Model features (input values)
    </div>
    <div class="features-list" id="featureGrid">
      ${Object.entries(d.features_used).map(([k, v]) =>
        `<div class="feat ${liveFeats.has(k) ? 'live' : ''}"><div class="fk">${k}</div><div class="fv">${v == null ? '—' : (+v).toFixed(3)}</div></div>`).join('')}
    </div>

    <div class="note">● = live weather/sensor value. Other features use the training median of the AI model. Satellite confirms a hotspot; the AI cross-checks it against environmental conditions.</div>

    ${renderInputsCards(d.inputs)}
  </div>`;

  openPanel(content);
}

function renderInputsCards(inp) {
  if (!inp) return '';
  const firms = inp.firms || {};
  const osm = inp.osm || {};
  const sat = inp.satellite || {};

  const inpRow = (k, v) => `<div class="inp-row"><span class="k">${k}</span><span class="v">${v == null || v === '' ? '—' : v}</span></div>`;
  const inpBar = (label, val, color) =>
    `<div class="prob-row"><div class="top"><span>${label}</span><span>${(val * 100).toFixed(0)}%</span></div>
     <div class="prob-track"><div class="prob-fill" style="width:${(val * 100).toFixed(0)}%;background:${color}"></div></div></div>`;

  const ships = firms.satellite ? `${firms.satellite} ${firms.instrument || ''}`.trim() : null;
  const acq = firms.acq_date ? `${firms.acq_date} ${firms.acq_time || ''}`.trim() : null;

  let cnnHtml = '';
  if (sat.cnn) {
    if (sat.cnn.applied && sat.cnn.probabilities) {
      const p = sat.cnn.probabilities;
      cnnHtml = `<div class="inp-note">image CNN: fire ${((sat.cnn.fire_prob || 0) * 100).toFixed(0)}% · forest ${((p.forest || 0) * 100).toFixed(0)}% · industrial ${((p.industrial || 0) * 100).toFixed(0)}% · nofire ${((p.nofire || 0) * 100).toFixed(0)}%</div>`;
    } else if (sat.cnn.note) {
      cnnHtml = `<div class="inp-note">${sat.cnn.note}</div>`;
    }
  }

  let detHtml = '';
  if (inp.detection) {
    const dt = inp.detection;
    const v = dt.votes || {};
    const dbar = (label, val, color) =>
      `<div class="prob-row"><div class="top"><span>${label}</span><span>${val == null ? '—' : (val * 100).toFixed(0) + '%'}</span></div>
       <div class="prob-track"><div class="prob-fill" style="width:${val == null ? 0 : (val * 100).toFixed(0)}%;background:${color}"></div></div></div>`;
    detHtml = `
    <div class="section-title">Detection fusion</div>
    <div class="inp-card det-card">
      <div class="inp-bars">
        ${dbar('NASA FIRMS + environment', v.tabular && v.tabular.p, '#f97316')}
        ${dbar('Satellite image CNN', v.image && v.image.p, '#ef4444')}
        ${dbar('OSM land-use', v.osm && v.osm.p, '#3b82f6')}
      </div>
      <div class="det-verdict ${dt.fire_detected ? 'on' : 'off'}">${dt.fire_detected ? '🔥 FIRE DETECTED' : '🌿 NO FIRE'} · fused fire ${((dt.p || 0) * 100).toFixed(0)}%</div>
    </div>`;
  }

  return `
    <div class="section-title">Inputs used</div>
    <div class="inputs-grid">
      <div class="inp-card">
        <div class="inp-head"><span class="inp-emoji">🛰️</span><div><div class="inp-t">NASA FIRMS</div><div class="inp-sub">satellite thermal anomalies</div></div></div>
        <div class="inp-rows">
          ${inpRow('Satellite', ships)}
          ${inpRow('Acquired', acq)}
          ${inpRow('Hotspots (5 km)', firms.num_hotspots_5km != null ? firms.num_hotspots_5km : '—')}
          ${inpRow('Brightness', firms.mean_brightness ? firms.mean_brightness + ' K' : null)}
          ${inpRow('Bright temp t31', firms.max_bright_t31 ? firms.max_bright_t31 + ' K' : null)}
          ${inpRow('FRP (max)', firms.max_frp ? firms.max_frp + ' MW' : null)}
          ${inpRow('Acquisition', firms.daynight === 'D' ? 'Day ☀️' : firms.daynight === 'N' ? 'Night 🌙' : null)}
          ${firms.confidence ? inpRow('Confidence', firms.confidence) : ''}
        </div>
      </div>
      <div class="inp-card">
        <div class="inp-head"><span class="inp-emoji">🗺️</span><div><div class="inp-t">OSM Land-use</div><div class="inp-sub">OpenStreetMap ${osm.radius_km ? '· ' + osm.radius_km + ' km radius' : ''}</div></div></div>
        ${osm.applied
          ? `<div class="inp-bars">${inpBar('Agricultural', osm.agricultural || 0, clsMeta.agricultural.color)}${inpBar('Forest', osm.forest || 0, clsMeta.forest.color)}${inpBar('Industrial', osm.industrial || 0, clsMeta.industrial.color)}</div>`
          : `<div class="inp-note">${osm.note || 'Not applied'}</div>`}
      </div>
      <div class="inp-card">
        <div class="inp-head"><span class="inp-emoji">🖼️</span><div><div class="inp-t">Satellite image</div><div class="inp-sub">live high-res view</div></div></div>
        ${sat.image_url ? `<img class="inp-img" src="${sat.image_url}" alt="satellite view" loading="lazy">` : ''}
        ${cnnHtml}
      </div>
    </div>
    ${detHtml}`;
}

function renderClassificationStrip(c) {
  const meta = clsMeta[String(c.classification)] || clsMeta.Unclassified;
  const host = document.getElementById('panelContent');
  if (!host) return;
  const strip = document.createElement('div');
  strip.className = 'cls-strip';
  strip.innerHTML = `
    <div class="cls-main">
      <span class="cls-emoji">${meta.emoji}</span>
      <div>
        <div class="cls-k">Detected fire type (classified)</div>
        <div class="cls-v">${meta.label} · ${(c.confidence * 100).toFixed(0)}% confidence</div>
        ${c.saved ? `<div class="cls-saved">✓ saved to GIS storage (PostGIS)</div>` : '<div class="cls-saved off">stored to GIS storage for spatial analysis</div>'}
      </div>
    </div>
    <div class="cls-bars">
      ${['agricultural', 'forest', 'industrial'].map(k => `
        <div class="prob-row">
          <div class="top"><span>${clsMeta[k].emoji} ${clsMeta[k].label}</span><span>${((c.probabilities ? c.probabilities[k] : 0) * 100).toFixed(0)}%</span></div>
          <div class="prob-track"><div class="prob-fill" style="width:${((c.probabilities ? c.probabilities[k] : 0) * 100).toFixed(0)}%;background:${clsMeta[k].color}"></div></div>
        </div>`).join('')}
    </div>`;
  const verdict = host.querySelector('.verdict');
  if (verdict) verdict.after(strip);
}

function weatherCard(k, v) {
  return `<div class="w-card"><div class="k">${k}</div><div class="v">${v}</div></div>`;
}

function getLiveFeatureKeys(features) {
  const live = new Set([
    'Temperature_C', 'Humidity_%', 'Pressure_hPa', 'PM2.5', 'PM1.0',
    'hour', 'day_of_week', 'month',
    'temp_humidity_ratio', 'pm_ratio', 'tvoc_eco2_ratio',
  ]);
  if (features && features.hour != null) live.add('hour');
  return live;
}

/* ---------------- panel & UI helpers ---------------- */

function openPanel(html) {
  els.panelContent.innerHTML = html;
  els.resultPanel.classList.remove('closed');
}

function closePanel() {
  els.resultPanel.classList.add('closed');
}

function setStatus(kind, text) {
  els.statusDot.className = 'dot' + (kind === 'live' ? ' live' : kind === 'err' ? ' err' : '');
  els.statusText.textContent = text;
}

function toast(msg, isError = false) {
  els.toast.textContent = msg;
  els.toast.hidden = false;
  els.toast.className = 'toast' + (isError ? ' err' : '');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { els.toast.hidden = true; }, 4200);
}

function formatAq(date, time) {
  if (!date) return '—';
  let t = String(time || '').padStart(4, '0');
  return `${date} ${t.slice(0, 2)}:${t.slice(2)} UTC`;
}

function formatEpoch(sec) {
  try {
    return new Date(sec * 1000).toLocaleString('en-IN', { hour12: false });
  } catch { return '—'; }
}

/* ---------------- settings ---------------- */

function getSetting(key) {
  try { return localStorage.getItem(key) || ''; } catch { return ''; }
}
function setSetting(key, val) {
  try { localStorage.setItem(key, val); } catch {}
}

async function loadServerConfig() {
  try {
    const res = await fetch('/api/config');
    state.serverConfig = await res.json();
  } catch {
    state.serverConfig = null;
  }
  updateConfigNote();
  setDefaultSourceFromConfig();
}

function updateConfigNote() {
  if (!els.serverConfigNote) return;
  const c = state.serverConfig;
  if (!c) { els.serverConfigNote.textContent = '⚠ could not reach server'; return; }
  els.serverConfigNote.textContent = c.firms_configured
    ? '✓ NASA FIRMS key loaded from .env — real satellite data ready'
    : '✗ No FIRMS key on server — add FIRMS_MAP_KEY in .env and restart';
  els.serverConfigNote.className = 'cfg-note ' + (c.firms_configured ? 'cfg-ok' : 'cfg-warn');
}

function setDefaultSourceFromConfig() {
  if (state.serverConfig && state.serverConfig.firms_configured) {
    if (getSetting('dataSource') !== 'demo') setSetting('dataSource', 'firms');
  } else {
    setSetting('dataSource', 'demo');
  }
  els.dataSource.value = getSetting('dataSource') || 'demo';
}

function saveSettings() {
  setSetting('dataSource', els.dataSource.value);
  els.settingsModal.hidden = true;
  refreshFires();
}

function bindEvents() {
  els.btnRefresh.addEventListener('click', refreshFires);
  els.btnSettings.addEventListener('click', () => { els.settingsModal.hidden = false; });
  els.panelClose.addEventListener('click', closePanel);
  els.settingsModal.querySelectorAll('[data-close]').forEach(b =>
    b.addEventListener('click', () => els.settingsModal.hidden = true));
  els.settingsModal.addEventListener('click', (e) => {
    if (e.target === els.settingsModal) els.settingsModal.hidden = true;
  });

  document.querySelectorAll('.lc-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.lc-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const layer = btn.dataset.layer;
      Object.values(state.baseLayers).forEach(l => state.map.removeLayer(l));
      (state.baseLayers[layer] || state.baseLayers.satellite).addTo(state.map);
    });
  });
}

document.addEventListener('DOMContentLoaded', init);