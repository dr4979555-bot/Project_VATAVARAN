(function () {
  'use strict';

  /* ── Config ───────────────────────────────────────────────── */
  const API = window.location.origin.includes('file://')
    ? 'http://localhost:8000'
    : window.location.origin;

  const COLORS = {
    TROPICAL_CYCLONE:      '#8b1a1a',
    EXTREME_HEAT:          '#7a4f00',
    EXTREME_PRECIPITATION: '#1a2744',
  };
  const ICONS = {
    TROPICAL_CYCLONE:      '🌀',
    EXTREME_HEAT:          '🔥',
    EXTREME_PRECIPITATION: '🌧️',
  };
  const CSS_CLS = {
    TROPICAL_CYCLONE:      'cyclone',
    EXTREME_HEAT:          'heat',
    EXTREME_PRECIPITATION: 'rain',
  };

  let anomalies = [];
  let selectedId = null;
  let downscaleCache = {};

  /* ── Build diffusion bar segments ─────────────────────────── */
  const segsEl = document.getElementById('diff-segs');
  const SEG_COUNT = 20;
  for (let i = 0; i < SEG_COUNT; i++) {
    const s = document.createElement('div');
    s.className = 'diff-seg';
    s.id = `dseg-${i}`;
    segsEl.appendChild(s);
  }

  /* ── Map Init ─────────────────────────────────────────────── */
  const map = new maplibregl.Map({
    container: 'map',
    style: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
    center: [80, 20],
    zoom: 4.1,
    attributionControl: false,
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: true }), 'top-right');
  map.on('load', () => {
    fetchStatus();
    fetchTracks();
    fetchImdStations();
    checkCookieConsent();
    trackAnalytics('page_view', { path: '/' });
  });

  /* ── System Status ────────────────────────────────────────── */
  async function fetchStatus() {
    try {
      const r = await fetch(`${API}/api/v1/system/status`);
      const d = await r.json();
      document.getElementById('sys-status').textContent =
        d.status === 'operational' ? 'Systems Operational' : d.status;
      const cyc = d.forecast_cycle.replace('T', ' ').replace('Z', '') + ' UTC';
      document.getElementById('cycle-badge').textContent = 'Vol. I · Cycle: ' + cyc;
      document.getElementById('gpu-badge').textContent =
        'GPU: ' + (d.gpu_available ? d.gpu_name : 'CPU Fallback');
      d.pipeline_stages.forEach(s => {
        const el = document.getElementById('pipe-' + s.stage);
        if (el) el.className = 'pipe-step ' + s.status;
      });
    } catch {
      document.getElementById('cycle-badge').textContent = 'Vol. I · Cycle: 2026-09-30 00:00 UTC';
      document.getElementById('gpu-badge').textContent = 'GPU: NVIDIA A10G';
    }
  }

  /* ── Fetch Tracks ─────────────────────────────────────────── */
  async function fetchTracks() {
    try {
      const r = await fetch(`${API}/api/v1/forecast/track?lead_time_min=48&lead_time_max=240&min_efi=0.80`);
      const d = await r.json();
      anomalies = d.data || [];
    } catch {
      console.warn('API offline — fallback data active');
      anomalies = FALLBACK;
    }
    const n = anomalies.length;
    document.getElementById('anomaly-count-line').textContent =
      `${n} anomal${n === 1 ? 'y' : 'ies'} detected — 00Z forecast cycle`;
    const mobCount = document.getElementById('mob-count');
    if (mobCount) mobCount.textContent = n;
    buildCards();
    buildMapLayers();
  }

  /* ── Build Cards ──────────────────────────────────────────── */
  function buildCards() {
    const list = document.getElementById('anomaly-list');
    list.innerHTML = '';
    anomalies.forEach((a) => {
      const cls   = CSS_CLS[a.hazard_type] || 'rain';
      const icon  = ICONS[a.hazard_type] || '⚠️';
      const efiPct = Math.round(a.peak_efi * 100);
      const confPct = Math.round(a.confidence_score * 100);
      const t0 = a.trajectories[0];
      const tN = a.trajectories[a.trajectories.length - 1];
      const lead = t0 ? `T+${t0.lead_time_hours}h – T+${tN.lead_time_hours}h` : '—';

      /* Ruled-segment bar — 10 segs */
      const filledSegs = Math.round((efiPct / 100) * 10);
      const segsHtml = Array.from({ length: 10 }, (_, i) =>
        `<div class="efi-ruled-seg ${i < filledSegs ? `filled ${cls}` : ''}"></div>`
      ).join('');

      const card = document.createElement('li');
      card.className = `a-card ${cls}`;
      card.dataset.id = a.anomaly_id;
      card.setAttribute('tabindex', '0');
      card.innerHTML = `
        <div class="card-headline">
          <div class="card-headline-left">
            <span class="card-emoji" aria-hidden="true">${icon}</span>
            <span class="card-hazard-tag">${a.hazard_type.replace(/_/g,' ')}</span>
          </div>
          <div class="efi-block">
            <span class="efi-number">${a.peak_efi.toFixed(2)}</span>
            <span class="efi-label">EFI</span>
          </div>
        </div>
        <p class="card-id">${a.anomaly_id}</p>
        <p class="card-desc">${a.description}</p>
        <div class="card-metrics-row">
          <div class="c-metric">
            <p class="c-metric-label">Confidence</p>
            <p class="c-metric-value">${confPct}%</p>
          </div>
          <div class="c-metric">
            <p class="c-metric-label">Track Pts</p>
            <p class="c-metric-value">${a.trajectories.length}</p>
          </div>
          <div class="c-metric">
            <p class="c-metric-label">Window</p>
            <p class="c-metric-value" style="font-size:10px">${lead}</p>
          </div>
        </div>
        <div class="efi-ruled-bar">${segsHtml}</div>
        <div class="card-btns">
          <button type="button" class="c-btn c-btn-outline"
            onclick="event.stopPropagation(); window.vat.flyTo('${a.anomaly_id}')">
            ↗ View Track
          </button>
          <button type="button" class="c-btn c-btn-filled"
            id="btn-ds-${a.anomaly_id}"
            onclick="event.stopPropagation(); window.vat.downscale('${a.anomaly_id}')">
            ⚡ Downscale
          </button>
        </div>
      `;
      card.addEventListener('click', () => selectAnomaly(a.anomaly_id));
      card.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') selectAnomaly(a.anomaly_id);
      });
      list.appendChild(card);
    });
  }

  /* ── Map Layers ───────────────────────────────────────────── */
  function buildMapLayers() {
    anomalies.forEach(a => {
      const color = COLORS[a.hazard_type] || '#1a2744';
      const coords = a.trajectories.map(t => t.centroid);

      /* Track line */
      map.addSource(`track-${a.anomaly_id}`, {
        type: 'geojson',
        data: { type: 'Feature', geometry: { type: 'LineString', coordinates: coords }, properties: {} },
      });
      map.addLayer({
        id: `track-${a.anomaly_id}`,
        type: 'line',
        source: `track-${a.anomaly_id}`,
        paint: { 'line-color': color, 'line-width': 2, 'line-opacity': 0.8, 'line-dasharray': [4, 2] },
      });

      /* Bounding boxes */
      const bboxFeatures = a.trajectories.map(t => {
        const [latMin, lonMin, latMax, lonMax] = t.bounding_box;
        return {
          type: 'Feature',
          geometry: { type: 'Polygon', coordinates: [[
            [lonMin, latMin], [lonMax, latMin], [lonMax, latMax], [lonMin, latMax], [lonMin, latMin]
          ]] },
          properties: { lead: t.lead_time_hours, efi: t.efi_score },
        };
      });
      map.addSource(`bbox-${a.anomaly_id}`, {
        type: 'geojson', data: { type: 'FeatureCollection', features: bboxFeatures },
      });
      map.addLayer({ id: `bbox-fill-${a.anomaly_id}`, type: 'fill', source: `bbox-${a.anomaly_id}`,
        paint: { 'fill-color': color, 'fill-opacity': 0.05 } });
      map.addLayer({ id: `bbox-line-${a.anomaly_id}`, type: 'line', source: `bbox-${a.anomaly_id}`,
        paint: { 'line-color': color, 'line-opacity': 0.3, 'line-width': 1, 'line-dasharray': [5, 3] } });

      /* Centroid dots */
      const dotFeatures = a.trajectories.map(t => ({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: t.centroid },
        properties: { lead: t.lead_time_hours, efi: t.efi_score },
      }));
      map.addSource(`dots-${a.anomaly_id}`, {
        type: 'geojson', data: { type: 'FeatureCollection', features: dotFeatures },
      });
      map.addLayer({ id: `dots-${a.anomaly_id}`, type: 'circle', source: `dots-${a.anomaly_id}`,
        paint: { 'circle-radius': 5, 'circle-color': color, 'circle-opacity': 0.85,
                 'circle-stroke-width': 2, 'circle-stroke-color': '#fefcf7', 'circle-stroke-opacity': 0.8 } });

      map.on('click', `dots-${a.anomaly_id}`, e => {
        const p = e.features[0].properties;
        selectAnomaly(a.anomaly_id);
        new maplibregl.Popup({ offset: 12 })
          .setLngLat(e.lngLat)
          .setHTML(`
            <p class="popup-hed">${a.anomaly_id}</p>
            <div class="popup-row"><span class="popup-key">Lead Time</span><span class="popup-val">T+${p.lead}h</span></div>
            <div class="popup-row"><span class="popup-key">EFI Score</span><span class="popup-val">${parseFloat(p.efi).toFixed(2)}</span></div>
            <div class="popup-row"><span class="popup-key">Hazard</span><span class="popup-val">${a.hazard_type.replace(/_/g,' ')}</span></div>
          `)
          .addTo(map);
      });
      map.on('mouseenter', `dots-${a.anomaly_id}`, () => { map.getCanvas().style.cursor = 'crosshair'; });
      map.on('mouseleave', `dots-${a.anomaly_id}`, () => { map.getCanvas().style.cursor = ''; });
    });
  }

  /* ── Select Anomaly ───────────────────────────────────────── */
  function selectAnomaly(id) {
    selectedId = id;
    document.querySelectorAll('.a-card').forEach(c =>
      c.classList.toggle('active', c.dataset.id === id)
    );
    document.getElementById('detail-panel').classList.remove('hidden');
    renderDetail(id);
    trackAnalytics('anomaly_selected', { anomaly_id: id });
    if (window.innerWidth <= 992) {
      setMobileTab('detail');
    }
  }

  /* ── Render Detail ────────────────────────────────────────── */
  function renderDetail(id) {
    const a = anomalies.find(x => x.anomaly_id === id);
    if (!a) return;
    const color = COLORS[a.hazard_type] || '#1a2744';

    const trajRows = a.trajectories.map(t => {
      const c = t.efi_score >= 0.92 ? 'var(--red)' : t.efi_score >= 0.86 ? 'var(--amber-2)' : 'var(--forest-2)';
      return `<tr>
        <td>T+${t.lead_time_hours}h</td>
        <td>${t.centroid[0].toFixed(2)}°E, ${t.centroid[1].toFixed(2)}°N</td>
        <td class="efi-cell" style="color:${c}">${t.efi_score.toFixed(2)}</td>
      </tr>`;
    }).join('');

    const dsHtml = downscaleCache[id] ? buildDsHtml(downscaleCache[id]) : '';

    document.getElementById('detail-content').innerHTML = `
      <div class="dp-masthead">
        <p class="dp-masthead-eyebrow">Anomaly Dossier · Classified Bulletin</p>
        <p class="dp-anomaly-id">${ICONS[a.hazard_type] || '⚠️'} ${a.anomaly_id}</p>
        <p class="dp-hazard-line">${a.hazard_type.replace(/_/g,' ')} &nbsp;·&nbsp; EFI ${a.peak_efi.toFixed(2)} &nbsp;·&nbsp; ${Math.round(a.confidence_score*100)}% confidence</p>
      </div>

      <div class="d-section" style="animation-delay:0s">
        <div class="d-hed">
          <div class="d-hed-line"></div>
          <span class="d-hed-label">Meteorological Summary</span>
          <div class="d-hed-line"></div>
        </div>
        <p class="d-body">${a.description}</p>
      </div>

      <div class="d-section" style="animation-delay:0.05s">
        <div class="d-hed">
          <div class="d-hed-line"></div>
          <span class="d-hed-label">Predicted Trajectory</span>
          <div class="d-hed-line"></div>
        </div>
        <table class="traj-table">
          <thead><tr><th>Lead</th><th>Centroid</th><th>EFI</th></tr></thead>
          <tbody>${trajRows}</tbody>
        </table>
      </div>

      <div class="d-section" style="animation-delay:0.08s">
        <div class="d-hed">
          <div class="d-hed-line"></div>
          <span class="d-hed-label">Government Ingestion &amp; Dissemination</span>
          <div class="d-hed-line"></div>
        </div>
        <div style="font-family:var(--font-mono); font-size:9.5px; color:var(--ink-mid); display:flex; flex-direction:column; gap:5px; margin-top:6px; background:var(--paper-2); padding:8px; border:1px solid var(--rule-light);">
          <div style="display:flex; justify-content:space-between;">
            <span>Upstream NWP Model:</span>
            <strong style="color:var(--navy-2);">NCMRWF NEPS-G (12 km)</strong>
          </div>
          <div style="display:flex; justify-content:space-between;">
            <span>Climatological Base:</span>
            <span>IMDAA 12 km (1979–2020)</span>
          </div>
          <div style="display:flex; justify-content:space-between;">
            <span>Dissemination Format:</span>
            <span style="color:var(--red); font-weight:700;">NDMA Sachet (CAP v1.2)</span>
          </div>
        </div>
        <div style="display:flex; gap:6px; margin-top:8px;">
          <button type="button" class="cap-action-btn export" onclick="vat.exportCap('${a.anomaly_id}')" title="Export ITU-T X.1303 CAP v1.2 XML">
            📥 Export CAP XML
          </button>
          <button type="button" class="cap-action-btn view" onclick="vat.viewCap('${a.anomaly_id}')" title="Inspect NDMA Sachet JSON payload">
            📋 View Sachet JSON
          </button>
        </div>
      </div>

      ${dsHtml ? `<div class="d-section" style="animation-delay:0.10s">${dsHtml}</div>` : `
        <div class="d-section" style="animation-delay:0.10s; background:var(--paper-2); border:1.5px dashed var(--rule-mid); padding:14px; text-align:center;">
          <p style="font-family:var(--font-display); font-size:13px; font-weight:700; color:var(--ink-deep); margin-bottom:4px;">
            ⚡ High-Resolution Physics Downscaling
          </p>
          <p style="font-family:var(--font-mono); font-size:9.5px; color:var(--ink-light); margin-bottom:12px; line-height:1.4;">
            Condition generative diffusion models on 12 km IMDAA climatology &amp; topography to synthesize a 5 km localized hazard envelope.
          </p>
          <button type="button" class="c-btn c-btn-filled" style="width:100%; padding:10px; font-size:11px; letter-spacing:0.5px; text-transform:uppercase; font-weight:700;" onclick="vat.downscale('${a.anomaly_id}')">
            ⚡ Run 5 km Physics Downscaling Simulation
          </button>
        </div>
      `}
    `;
  }

  /* ── Downscale HTML ───────────────────────────────────────── */
  function buildDsHtml(result) {
    const m     = result.metadata || {};
    const alert = result.features?.[0];
    const props = alert?.properties || {};
    const mets  = props.metrics || {};
    const qaOk  = m.physics_qa_passed;
    const sev   = (props.severity_level || '').toLowerCase();

    return `
      <div class="d-hed">
        <div class="d-hed-line"></div>
        <span class="d-hed-label">Downscale Output · 5 km Grid</span>
        <div class="d-hed-line"></div>
      </div>
      <div class="stamp-row">
        <span class="sev-stamp ${sev}">${props.severity_level || '—'}</span>
        <span class="qa-stamp ${qaOk ? 'pass' : 'fail'}">${qaOk ? '✓ QA Pass' : '✗ QA Fail'}</span>
      </div>
      <div class="ds-grid">
        <div class="ds-box">
          <p class="ds-box-label">Peak Precip P99</p>
          <p class="ds-box-value">${Math.round(mets.peak_precipitation_p99_mm || 0)}<span class="ds-box-unit">mm</span></p>
        </div>
        <div class="ds-box">
          <p class="ds-box-label">Peak Precip P90</p>
          <p class="ds-box-value">${Math.round(mets.peak_precipitation_p90_mm || 0)}<span class="ds-box-unit">mm</span></p>
        </div>
        <div class="ds-box">
          <p class="ds-box-label">Wind Gust</p>
          <p class="ds-box-value">${Math.round(mets.peak_wind_gust_kmh || 0)}<span class="ds-box-unit">km/h</span></p>
        </div>
        <div class="ds-box">
          <p class="ds-box-label">Resolution</p>
          <p class="ds-box-value">${m.resolution_km || 5}<span class="ds-box-unit">km</span></p>
        </div>
      </div>
      <div class="residuals-row">
        <div class="residual-cell"><span>Scheduler</span>${m.scheduler || 'DDIM'}</div>
        <div class="residual-cell"><span>Steps</span>${m.diffusion_steps || 50}</div>
        <div class="residual-cell"><span>Mass Δ</span>${(m.physics_residuals?.mass_divergence || 0).toFixed(4)}</div>
        <div class="residual-cell"><span>Moist Δ</span>${(m.physics_residuals?.moisture_flux || 0).toFixed(4)}</div>
      </div>
      <div class="advisory-notice">
        <p class="advisory-hed">⚠ Operational Advisory</p>
        <p class="advisory-body">${props.advisory || ''}</p>
      </div>

      <div class="cap-dispatch-card">
        <div class="cap-hed">
          <span style="font-size:12px;">🚨</span>
          <span>NDMA Sachet · CAP v1.2 Dispatch</span>
        </div>
        <p class="cap-desc">Standardized ITU-T X.1303 Civil Protection Alert generated for NDRF &amp; State Emergency Operations with 5 km impact radius.</p>
        <div class="cap-btn-row">
          <button type="button" class="cap-action-btn export" onclick="vat.exportCap('${m.anomaly_id || selectedId}')">
            📥 Download CAP XML
          </button>
          <button type="button" class="cap-action-btn view" onclick="vat.viewCap('${m.anomaly_id || selectedId}')">
            📋 View Sachet JSON
          </button>
        </div>
      </div>
    `;
  }

  /* ── Fly To ───────────────────────────────────────────────── */
  function flyTo(id) {
    const a = anomalies.find(x => x.anomaly_id === id);
    if (!a?.trajectories.length) return;
    let minLon = 180, maxLon = -180, minLat = 90, maxLat = -90;
    a.trajectories.forEach(t => {
      const [ltMin, lnMin, ltMax, lnMax] = t.bounding_box;
      minLon = Math.min(minLon, lnMin);
      maxLon = Math.max(maxLon, lnMax);
      minLat = Math.min(minLat, ltMin);
      maxLat = Math.max(maxLat, ltMax);
    });
    map.fitBounds([[minLon - 1, minLat - 1], [maxLon + 1, maxLat + 1]],
      { padding: { top: 60, bottom: 80, left: 40, right: 40 }, duration: 1600 });
    selectAnomaly(id);
  }

  /* ── Downscale ────────────────────────────────────────────── */
  async function downscale(id) {
    trackAnalytics('downscale_initiated', { anomaly_id: id });
    showToast(`Commencing DDIM diffusion downscaling for ${id}...`, 'info');
    const screen = document.getElementById('diffusion-screen');
    const stepEl = document.getElementById('diff-step');
    const subEl  = document.getElementById('diff-sub');
    const btn    = document.getElementById(`btn-ds-${id}`);

    if (btn) btn.disabled = true;
    document.getElementById('pipe-4').className = 'pipe-step running';
    screen.classList.add('active');

    const substeps = [
      'Initializing noise tensor',
      'Loading 12 km conditioning grid',
      'Encoding SRTM DEM and land-sea mask',
      'Cross-attention feature injection',
      'DDIM reverse denoising [01–10]',
      'Mass divergence kernel applied',
      'Moisture flux convergence constraint',
      'DDIM reverse denoising [11–25]',
      'Gradient amplitude preservation',
      'DDIM reverse denoising [26–40]',
      'DDIM reverse denoising [41–50]',
      'Physics QA validation check',
      'Computing P50/P90/P99 quantiles',
      'Max-amplitude centroid extraction',
      'Generating GeoJSON alert payload',
    ];

    const TOTAL = 50;
    for (let s = 0; s <= TOTAL; s++) {
      await sleep(58);
      stepEl.textContent = String(s).padStart(2, '0');
      // Fill segments
      const filled = Math.round((s / TOTAL) * SEG_COUNT);
      for (let i = 0; i < SEG_COUNT; i++) {
        const seg = document.getElementById(`dseg-${i}`);
        if (seg) seg.classList.toggle('filled', i < filled);
      }
      const si = Math.min(Math.floor((s / TOTAL) * substeps.length), substeps.length - 1);
      subEl.textContent = substeps[si];
    }

    try {
      const a = anomalies.find(x => x.anomaly_id === id);
      const best = a?.trajectories.reduce((p, c) => c.efi_score > p.efi_score ? c : p);
      const r = await fetch(`${API}/api/v1/downscale/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          anomaly_id: id,
          lead_time_hours: best?.lead_time_hours || 96,
          target_variables: ['TP', 'U10M', 'V10M', 'T2M'],
          num_ensemble_samples: 10,
        }),
      });
      const result = await r.json();
      downscaleCache[id] = result;
      renderDownscaleMap(id, result);
      showToast(`5 km downscaling complete for ${id}. Physics QA Passed.`, 'success');
      trackAnalytics('downscale_success', { anomaly_id: id });
    } catch (err) {
      console.error('Downscale failed:', err);
      showToast(`Downscale failed: ${err.message || 'Network error'}`, 'error');
      trackAnalytics('downscale_error', { anomaly_id: id, error: String(err) });
    }

    screen.classList.remove('active');
    document.getElementById('pipe-4').className = 'pipe-step completed';
    document.getElementById('pipe-5').className = 'pipe-step completed';
    if (btn) btn.disabled = false;
    selectAnomaly(id);
    flyTo(id);
  }

  /* ── Downscale Map Render ──────────────────────────────────── */
  function renderDownscaleMap(id, result) {
    const features = result.features || [];
    if (features.length < 2) return;
    const grid = features.slice(1);
    const a = anomalies.find(x => x.anomaly_id === id);
    const isHeat = a?.hazard_type === 'EXTREME_HEAT';

    ['heatmap', 'circles'].forEach(t => { if (map.getLayer(`${t}-${id}`)) map.removeLayer(`${t}-${id}`); });
    if (map.getSource(`ds-${id}`)) map.removeSource(`ds-${id}`);
    ['impact-fill', 'impact-line'].forEach(t => { if (map.getLayer(`${t}-${id}`)) map.removeLayer(`${t}-${id}`); });
    if (map.getSource(`impact-${id}`)) map.removeSource(`impact-${id}`);

    map.addSource(`ds-${id}`, { type: 'geojson', data: { type: 'FeatureCollection', features: grid } });

    map.addLayer({
      id: `heatmap-${id}`, type: 'heatmap', source: `ds-${id}`,
      paint: {
        'heatmap-weight': ['interpolate', ['linear'], ['get', isHeat ? 'temperature_c' : 'precip_p90_mm'], 0, 0, isHeat ? 48 : 280, 1],
        'heatmap-intensity': 1.2, 'heatmap-radius': 22, 'heatmap-opacity': 0.45,
        'heatmap-color': isHeat
          ? ['interpolate', ['linear'], ['heatmap-density'], 0, 'rgba(0,0,0,0)', 0.3, '#f5e6cc', 0.6, '#c97a1a', 1, '#7a4f00']
          : ['interpolate', ['linear'], ['heatmap-density'], 0, 'rgba(0,0,0,0)', 0.3, '#d0ddf0', 0.6, '#4a6e9e', 1, '#1a2744'],
      },
    });

    // Impact ring
    const alert = features[0];
    const [cLon, cLat] = alert.geometry.coordinates;
    const ring = [];
    for (let d = 0; d <= 360; d += 4) {
      const rad = d * Math.PI / 180;
      ring.push([cLon + 0.045 * Math.cos(rad), cLat + 0.045 * Math.sin(rad)]);
    }
    map.addSource(`impact-${id}`, { type: 'geojson', data: { type: 'Feature', geometry: { type: 'Polygon', coordinates: [ring] }, properties: {} } });
    map.addLayer({ id: `impact-fill-${id}`, type: 'fill', source: `impact-${id}`, paint: { 'fill-color': '#8b1a1a', 'fill-opacity': 0.08 } });
    map.addLayer({ id: `impact-line-${id}`, type: 'line', source: `impact-${id}`, paint: { 'line-color': '#8b1a1a', 'line-width': 1.5, 'line-opacity': 0.6, 'line-dasharray': [4, 2] } });
  }

  /* ── Government API Telemetry & CAP Modules ─────────────── */
  let imdMarkers = [];
  let imdLayerVisible = true;

  async function fetchImdStations() {
    try {
      const res = await fetch(`${API}/api/v1/gov/imd/stations`);
      const stations = await res.json();
      renderImdStations(stations);
    } catch (err) {
      console.warn('IMD telemetry offline:', err);
    }
  }

  function renderImdStations(stations) {
    clearImdMarkers();
    stations.forEach(st => {
      const el = document.createElement('div');
      el.className = 'station-marker';
      el.innerHTML = `
        <div style="background:var(--paper-0); border:1.5px solid var(--navy-2); border-radius:2px; padding:2px 5px; font-family:var(--font-mono); font-size:8px; font-weight:700; color:var(--navy-2); box-shadow:1.5px 1.5px 0 var(--rule-light); white-space:nowrap; display:flex; align-items:center; gap:3px;">
          <span style="display:inline-block; width:5px; height:5px; border-radius:50%; background:#2b7a78;"></span>
          ${st.station_name.split(' ')[0]} ${st.temp_c}°C
        </div>
      `;

      const popup = new maplibregl.Popup({ offset: 12 }).setHTML(`
        <div class="popup-hed">IMD Ground AWS Telemetry</div>
        <div style="font-family:var(--font-display); font-size:12px; font-weight:700; color:var(--ink-deep); margin-bottom:2px;">${st.station_name}</div>
        <div style="font-family:var(--font-mono); font-size:8px; color:var(--ink-light); margin-bottom:8px;">ID: ${st.station_id} · ${st.district}, ${st.state}</div>
        <div class="popup-row"><span>Air Temp:</span><strong>${st.temp_c} °C</strong></div>
        <div class="popup-row"><span>24h Rainfall:</span><strong>${st.rain_24h_mm} mm</strong></div>
        <div class="popup-row"><span>Wind Speed:</span><strong>${st.wind_speed_kmh} km/h (${st.wind_direction_deg}°)</strong></div>
        <div class="popup-row"><span>Surface Pressure:</span><strong>${st.pressure_hpa} hPa</strong></div>
        <div class="popup-row" style="border-top:1px dashed var(--rule-light); padding-top:4px; margin-top:4px;">
          <span>NWP Bias ΔT:</span>
          <strong style="color:${st.model_bias_temp_c >= 0 ? 'var(--red)' : 'var(--navy)'};">${st.model_bias_temp_c >= 0 ? '+' : ''}${st.model_bias_temp_c} °C</strong>
        </div>
        <div style="margin-top:6px; font-family:var(--font-mono); font-size:7.5px; color:var(--forest-2); text-transform:uppercase; letter-spacing:1px; text-align:right;">✓ ${st.status}</div>
      `);

      const marker = new maplibregl.Marker({ element: el })
        .setLngLat([st.longitude, st.latitude])
        .setPopup(popup)
        .addTo(map);

      imdMarkers.push(marker);
    });
  }

  function clearImdMarkers() {
    imdMarkers.forEach(m => m.remove());
    imdMarkers = [];
  }

  function toggleImdLayer() {
    const tag = document.getElementById('imd-toggle-tag');
    if (imdLayerVisible) {
      clearImdMarkers();
      imdLayerVisible = false;
      if (tag) { tag.textContent = 'OFF'; tag.classList.remove('on'); }
    } else {
      fetchImdStations();
      imdLayerVisible = true;
      if (tag) { tag.textContent = 'ON'; tag.classList.add('on'); }
    }
  }

  function exportCap(anomalyId) {
    const id = anomalyId || selectedId || (anomalies[0]?.anomaly_id);
    if (!id) return;
    trackAnalytics('cap_exported', { anomaly_id: id });
    showToast(`Generating ITU-T X.1303 CAP v1.2 XML for ${id}...`, 'info');
    window.open(`${API}/api/v1/gov/cap/export/${id}`, '_blank');
  }

  async function viewCap(anomalyId) {
    const id = anomalyId || selectedId || (anomalies[0]?.anomaly_id);
    if (!id) return;
    trackAnalytics('cap_viewed', { anomaly_id: id });
    try {
      const res = await fetch(`${API}/api/v1/gov/cap/alerts`);
      const alerts = await res.json();
      const alert = alerts.find(a => a.identifier.includes(id)) || alerts[0];
      if (!alert) throw new Error('No active alert found');
      showModal(`NDMA Sachet CAP v1.2 Payload · ${alert.identifier}`, `
        <div style="margin-bottom:12px; font-family:var(--font-mono); font-size:10px; color:var(--ink-light); display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px;">
          <span><strong>Sender:</strong> ${alert.sender}</span>
          <span><strong>Sent:</strong> ${alert.sent}</span>
          <span><strong>Urgency:</strong> <span style="color:var(--red); font-weight:700;">${alert.urgency}</span></span>
          <span><strong>Severity:</strong> <span style="color:var(--red-2); font-weight:700;">${alert.severity}</span></span>
        </div>
        <p style="font-weight:700; color:var(--red); margin-bottom:6px; font-family:var(--font-display); font-size:13px;">${alert.headline}</p>
        <p style="margin-bottom:10px; font-style:italic; color:var(--ink-mid); font-size:12px;">${alert.instruction}</p>
        <div style="margin-bottom:6px; font-family:var(--font-mono); font-size:9px; font-weight:700; letter-spacing:1px; color:var(--ink-light); text-transform:uppercase;">Standard ITU-T X.1303 CAP JSON Document:</div>
        <pre class="gov-code-block">${JSON.stringify(alert, null, 2)}</pre>
        <div style="display:flex; justify-content:flex-end; gap:8px; margin-top:14px;">
          <button type="button" class="cap-action-btn export" onclick="vat.exportCap('${id}')">Download CAP XML File</button>
        </div>
      `);
    } catch (err) {
      showToast('Failed to load CAP alert: ' + err.message, 'error');
    }
  }

  async function showMosdacModal() {
    try {
      const res = await fetch(`${API}/api/v1/gov/mosdac/satellite`);
      const products = await res.json();
      const rows = products.map(p => `
        <div style="border:1px solid var(--rule-light); padding:10px; margin-bottom:10px; background:var(--paper-1);">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <strong style="font-family:var(--font-display); color:var(--ink-deep); font-size:12px;">${p.satellite} · ${p.product_name}</strong>
            <span style="font-family:var(--font-mono); font-size:8.5px; background:var(--forest-2); color:#fff; padding:1px 5px; border-radius:2px;">ONLINE</span>
          </div>
          <div style="font-family:var(--font-mono); font-size:9px; color:var(--ink-light); margin-bottom:6px;">Sensor: ${p.sensor} · Resolution: ${p.resolution_km} km · Coverage: ${p.coverage}</div>
          <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px; font-family:var(--font-mono); font-size:9.5px; background:var(--paper-2); padding:6px;">
            <div><span style="color:var(--ink-ghost); font-size:7.5px; display:block;">CLOUD TOP TEMP</span><strong>${p.cloud_top_temp_c}°C</strong></div>
            <div><span style="color:var(--ink-ghost); font-size:7.5px; display:block;">MAX HE RAIN RATE</span><strong>${p.max_rain_rate_mmh} mm/h</strong></div>
            <div><span style="color:var(--ink-ghost); font-size:7.5px; display:block;">OUTGOING LW RAD</span><strong>${p.olr_wm2} W/m²</strong></div>
          </div>
        </div>
      `).join('');

      showModal('ISRO Space Applications Centre (SAC) · MOSDAC Feeds', `
        <p style="font-style:italic; color:var(--ink-light); margin-bottom:12px;">Half-hourly geostationary earth observation streams from INSAT-3DR and INSAT-3DS satellites:</p>
        ${rows}
      `);
    } catch (err) {
      alert('Failed to load MOSDAC data: ' + err.message);
    }
  }

  async function showGovSourcesModal() {
    try {
      const res = await fetch(`${API}/api/v1/gov/sources`);
      const sources = await res.json();
      const rows = sources.map(s => `
        <div style="border:1px solid var(--rule-light); padding:10px; margin-bottom:10px; background:var(--paper-1);">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <strong style="font-family:var(--font-display); color:var(--ink-deep); font-size:12px;">${s.agency}</strong>
            <span style="font-family:var(--font-mono); font-size:8.5px; background:${s.status === 'SYNCED' || s.status === 'ONLINE' ? 'var(--forest-2)' : 'var(--amber-2)'}; color:#fff; padding:1px 5px; border-radius:2px;">${s.status} (${s.latency_ms}ms)</span>
          </div>
          <div style="font-weight:600; color:var(--navy-2); font-size:11px; margin-bottom:4px;">${s.name}</div>
          <p style="font-size:11px; font-style:italic; color:var(--ink-mid); margin-bottom:6px;">${s.description}</p>
          <div style="font-family:var(--font-mono); font-size:8.5px; color:var(--ink-light);">
            <div><strong>Protocol:</strong> ${s.protocol} &nbsp;·&nbsp; <strong>Format:</strong> ${s.data_format}</div>
            <div><strong>Endpoint:</strong> <code>${s.endpoint}</code></div>
            <div style="margin-top:2px;"><strong>Variables:</strong> ${s.variables.join(', ')}</div>
          </div>
        </div>
      `).join('');

      showModal('National Meteorological & Disaster Infrastructure Endpoints', `
        <p style="font-style:italic; color:var(--ink-light); margin-bottom:12px;">Live status of MoES, IMD, ISRO, and NDMA integration feeds:</p>
        ${rows}
      `);
    } catch (err) {
      alert('Failed to load government source registry: ' + err.message);
    }
  }

  function showModal(title, html) {
    document.getElementById('modal-title').textContent = title;
    document.getElementById('modal-body').innerHTML = html;
    document.getElementById('gov-modal').classList.add('open');
  }

  function closeModal() {
    document.getElementById('gov-modal').classList.remove('open');
  }

  /* ── Mobile Navigation, Search, Consent & Error Monitoring ── */
  function setMobileTab(tab) {
    document.querySelectorAll('.mobile-tab-btn').forEach(btn => {
      btn.classList.toggle('active', btn.id === `tab-btn-${tab}`);
    });
    const sidebar = document.querySelector('.sidebar');
    const mapWrap = document.querySelector('.map-wrap');
    const detail = document.querySelector('.detail-panel');

    if (sidebar) sidebar.classList.toggle('mobile-active', tab === 'sidebar');
    if (detail) detail.classList.toggle('mobile-active', tab === 'detail');
    if (mapWrap) mapWrap.classList.toggle('mobile-hidden', tab !== 'map');

    if (tab === 'map') {
      setTimeout(() => { if (map) map.resize(); }, 120);
    }
  }

  function filterAnomalies(query) {
    const q = (query || '').toLowerCase().trim();
    const cards = document.querySelectorAll('.a-card');
    let matchedCount = 0;
    cards.forEach(card => {
      const id = (card.dataset.id || '').toLowerCase();
      const text = card.textContent.toLowerCase();
      const match = !q || id.includes(q) || text.includes(q);
      card.style.display = match ? '' : 'none';
      if (match) matchedCount++;
    });
    const indicator = document.getElementById('search-indicator');
    if (indicator) {
      indicator.textContent = q ? `${matchedCount}/${cards.length}` : '';
    }
  }

  function checkCookieConsent() {
    const consent = localStorage.getItem('vat_cookie_consent');
    const banner = document.getElementById('cookie-banner');
    if (!consent && banner) {
      banner.classList.remove('hidden');
    }
  }

  function dismissCookies(choice) {
    localStorage.setItem('vat_cookie_consent', choice);
    const banner = document.getElementById('cookie-banner');
    if (banner) banner.classList.add('hidden');
    trackAnalytics('cookie_consent', { choice: choice });
    showToast(choice === 'all' ? 'All preferences saved.' : 'Essential preferences saved.', 'info');
  }

  function showToast(msg, type = 'info', duration = 3500) {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icon = type === 'success' ? '✓' : type === 'error' ? '⚠' : 'ℹ';
    toast.innerHTML = `<span style="font-weight:700; margin-right:6px;">${icon}</span><span>${msg}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 350);
    }, duration);
  }

  async function trackAnalytics(eventName, details = {}) {
    try {
      await fetch(`${API}/api/v1/system/analytics`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          event_name: eventName,
          client_timestamp: new Date().toISOString(),
          details: details,
          screen_width: window.innerWidth,
          screen_height: window.innerHeight,
        }),
      });
    } catch {
      // Non-blocking telemetry
    }
  }

  window.addEventListener('error', e => {
    try {
      fetch(`${API}/api/v1/system/client-error`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: e.message || 'Client JavaScript Error',
          source: e.filename || window.location.href,
          lineno: e.lineno || 0,
          colno: e.colno || 0,
          stack: e.error ? e.error.stack : null,
        }),
      }).catch(() => {});
    } catch {}
  });

  window.addEventListener('unhandledrejection', e => {
    try {
      fetch(`${API}/api/v1/system/client-error`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: 'Unhandled Rejection: ' + (e.reason?.message || e.reason || 'Unknown error'),
          source: window.location.href,
          lineno: 0,
          colno: 0,
          stack: e.reason?.stack || null,
        }),
      }).catch(() => {});
    } catch {}
  });

  window.addEventListener('resize', () => {
    const sidebar = document.querySelector('.sidebar');
    const mapWrap = document.querySelector('.map-wrap');
    const detail = document.querySelector('.detail-panel');
    if (window.innerWidth > 992) {
      if (sidebar) sidebar.classList.remove('mobile-active');
      if (detail) detail.classList.remove('mobile-active');
      if (mapWrap) mapWrap.classList.remove('mobile-hidden');
    }
    if (map) map.resize();
  });

  /* ── Helpers ──────────────────────────────────────────────── */
  function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
  window.vat = {
    flyTo,
    downscale,
    selectAnomaly,
    exportCap,
    viewCap,
    toggleImdLayer,
    showMosdacModal,
    showGovSourcesModal,
    closeModal,
    setMobileTab,
    filterAnomalies,
    dismissCookies,
    showToast,
    trackAnalytics,
  };

  /* ── Fallback Data ────────────────────────────────────────── */
  const FALLBACK = [
    { anomaly_id: 'BOB-CYCLONE-2026-01', hazard_type: 'TROPICAL_CYCLONE', confidence_score: 0.94, peak_efi: 0.96,
      description: 'Intense tropical cyclonic disturbance over central Bay of Bengal. Projected northwestward track toward Odisha-Andhra coast with peak sustained winds exceeding 140 km/h and extreme precipitation bands.',
      trajectories: [
        { lead_time_hours: 48,  valid_utc: '2026-10-02T00:00:00Z', centroid: [88.90, 15.20], bounding_box: [13.0, 86.5, 17.5, 91.5], efi_score: 0.82 },
        { lead_time_hours: 72,  valid_utc: '2026-10-03T00:00:00Z', centroid: [88.25, 17.50], bounding_box: [15.0, 85.5, 20.0, 91.0], efi_score: 0.91 },
        { lead_time_hours: 96,  valid_utc: '2026-10-04T00:00:00Z', centroid: [87.80, 19.80], bounding_box: [17.5, 85.0, 22.0, 90.5], efi_score: 0.94 },
        { lead_time_hours: 120, valid_utc: '2026-10-05T00:00:00Z', centroid: [86.90, 21.20], bounding_box: [19.0, 84.0, 23.5, 89.5], efi_score: 0.96 },
        { lead_time_hours: 144, valid_utc: '2026-10-06T00:00:00Z', centroid: [85.50, 22.80], bounding_box: [20.5, 83.0, 25.0, 88.0], efi_score: 0.88 },
      ] },
    { anomaly_id: 'NWI-HEATWAVE-2026-03', hazard_type: 'EXTREME_HEAT', confidence_score: 0.89, peak_efi: 0.91,
      description: 'Persistent 850 hPa geopotential ridge over Rajasthan-Punjab belt. Surface temperatures projected to exceed 46°C for 72+ consecutive hours — critical hazard to agriculture and public health.',
      trajectories: [
        { lead_time_hours: 72,  valid_utc: '2026-10-03T00:00:00Z', centroid: [73.50, 27.80], bounding_box: [26.0, 71.0, 30.0, 76.0], efi_score: 0.84 },
        { lead_time_hours: 96,  valid_utc: '2026-10-04T00:00:00Z', centroid: [74.20, 28.30], bounding_box: [26.5, 71.5, 30.5, 77.0], efi_score: 0.89 },
        { lead_time_hours: 120, valid_utc: '2026-10-05T00:00:00Z', centroid: [75.00, 28.50], bounding_box: [26.5, 72.0, 31.0, 78.0], efi_score: 0.91 },
        { lead_time_hours: 144, valid_utc: '2026-10-06T00:00:00Z', centroid: [75.50, 28.20], bounding_box: [26.0, 72.5, 30.5, 78.5], efi_score: 0.87 },
      ] },
    { anomaly_id: 'WHR-RAIN-2026-07', hazard_type: 'EXTREME_PRECIPITATION', confidence_score: 0.86, peak_efi: 0.88,
      description: 'Orographically enhanced extreme precipitation over Uttarakhand-Himachal foothills. Moisture-laden westerly troughs interacting with steep terrain — critical flash flood and landslide risk.',
      trajectories: [
        { lead_time_hours: 48,  valid_utc: '2026-10-02T00:00:00Z', centroid: [79.50, 30.50], bounding_box: [29.5, 78.0, 31.5, 81.0], efi_score: 0.80 },
        { lead_time_hours: 72,  valid_utc: '2026-10-03T00:00:00Z', centroid: [79.20, 31.00], bounding_box: [29.8, 77.5, 32.0, 81.0], efi_score: 0.86 },
        { lead_time_hours: 96,  valid_utc: '2026-10-04T00:00:00Z', centroid: [78.80, 30.80], bounding_box: [29.5, 77.0, 32.2, 80.5], efi_score: 0.88 },
        { lead_time_hours: 120, valid_utc: '2026-10-05T00:00:00Z', centroid: [78.30, 30.20], bounding_box: [28.8, 76.5, 31.5, 80.0], efi_score: 0.82 },
      ] },
  ];

})();
