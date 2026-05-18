/**
 * LandWatch AI — Dashboard Application
 * Leaflet map, alert management, analysis triggers, and layer controls.
 */

const API_BASE = '/api';
let map, alertLayer, changeLayer, segLayer, zoningLayer;
let activeFilters = { severities: ['CRITICAL','HIGH','MEDIUM','LOW'], minConfidence: 0.5, zone: '' };
let allAlerts = [];

// ═══ Initialize ═══
document.addEventListener('DOMContentLoaded', async () => {
    initMap();
    lucide.createIcons();
    await loadConfig();
    await refreshAlerts();
    setInterval(refreshAlerts, 30000);
});

// ═══ Map Setup ═══
function initMap() {
    map = L.map('map', {
        center: [20.745, 78.600],
        zoom: 12,
        zoomControl: true,
        attributionControl: false,
    });

    // Dark satellite basemap
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: 19,
        attribution: '&copy; Esri',
    }).addTo(map);

    // Labels overlay
    L.tileLayer('https://stamen-tiles.a.ssl.fastly.net/toner-labels/{z}/{x}/{y}.png', {
        maxZoom: 19, opacity: 0.7,
    }).addTo(map);

    // Layer groups
    alertLayer = L.layerGroup().addTo(map);
    changeLayer = L.layerGroup().addTo(map);
    segLayer = L.layerGroup();
    zoningLayer = L.layerGroup();

    // Mouse events
    map.on('mousemove', (e) => {
        document.getElementById('cursorCoords').textContent =
            `${e.latlng.lat.toFixed(4)}°N, ${e.latlng.lng.toFixed(4)}°E`;
    });
    map.on('zoomend', () => {
        document.getElementById('zoomLevel').textContent = `Zoom: ${map.getZoom()}`;
    });
}

// ═══ Config ═══
async function loadConfig() {
    try {
        const res = await fetch(`${API_BASE}/config`);
        if (res.ok) {
            const cfg = await res.json();
            map.setView(cfg.aoi.center, 12);
        }
    } catch(e) { /* offline — use defaults */ }
}

// ═══ Alerts ═══
async function refreshAlerts() {
    try {
        const res = await fetch(`${API_BASE}/alerts/geojson`);
        if (!res.ok) return;
        const geojson = await res.json();
        allAlerts = geojson.features || [];
        renderAlerts();
        updateStats();
    } catch(e) {
        // API not running — show demo data
        if (allAlerts.length === 0) loadDemoAlerts();
    }
}

function renderAlerts() {
    alertLayer.clearLayers();
    const list = document.getElementById('alertList');
    const filtered = filterAlerts(allAlerts);

    document.getElementById('alertCount').textContent = filtered.length;

    if (filtered.length === 0) {
        list.innerHTML = `<div class="empty-state">
            <i data-lucide="satellite-dish" class="empty-icon"></i>
            <p>No alerts match filters</p>
            <p class="empty-sub">Adjust filters or run an analysis</p>
        </div>`;
        lucide.createIcons();
        return;
    }

    list.innerHTML = '';
    filtered.forEach(f => {
        const p = f.properties;
        const lvl = p.severity_level.toLowerCase();

        // Map marker
        const marker = createAlertMarker(f);
        alertLayer.addLayer(marker);

        // Sidebar card
        const card = document.createElement('div');
        card.className = `alert-card ${lvl}`;
        card.innerHTML = `
            <div class="alert-card-header">
                <span class="alert-card-id">${p.alert_id}</span>
                <span class="alert-severity-badge badge-${lvl}">${p.severity_level}</span>
            </div>
            <div class="alert-card-zone">${formatZone(p.zone_type)} — ${p.violation_type.replace(/_/g, ' ')}</div>
            <div class="alert-card-coords">${p.coordinates.latitude.toFixed(4)}°N, ${p.coordinates.longitude.toFixed(4)}°E</div>
        `;
        card.onclick = () => {
            map.setView([p.coordinates.latitude, p.coordinates.longitude], 16);
            marker.openPopup();
            showDetail(p);
        };
        list.appendChild(card);
    });

    lucide.createIcons();
}

function createAlertMarker(feature) {
    const p = feature.properties;
    const lvl = p.severity_level.toLowerCase();
    const coords = [p.coordinates.latitude, p.coordinates.longitude];

    const icon = L.divIcon({
        className: '',
        html: `<div class="pulse-marker ${lvl}"></div>`,
        iconSize: [18, 18],
        iconAnchor: [9, 9],
    });

    const marker = L.marker(coords, { icon });

    const popupContent = `
        <div style="min-width:200px">
            <div style="font-weight:600;margin-bottom:6px;font-size:0.9rem">
                Alert ${p.alert_id}
            </div>
            <div style="display:grid;gap:4px;font-size:0.8rem">
                <div><strong>Severity:</strong> <span style="color:${getSeverityColor(p.severity_level)}">${p.severity_level} (${(p.severity_score*100).toFixed(0)}%)</span></div>
                <div><strong>Zone:</strong> ${formatZone(p.zone_type)}</div>
                <div><strong>Area:</strong> ${p.area_m2.toFixed(0)} m²</div>
                <div><strong>Confidence:</strong> ${(p.model_confidence*100).toFixed(0)}%</div>
                <div><strong>Period:</strong> ${p.temporal_range.before_date} → ${p.temporal_range.after_date}</div>
            </div>
            <button onclick="showDetail(${JSON.stringify(p).replace(/"/g, '&quot;')})"
                    style="margin-top:8px;padding:4px 12px;background:var(--accent);color:white;border:none;border-radius:6px;cursor:pointer;font-size:0.78rem">
                View Details
            </button>
        </div>
    `;

    marker.bindPopup(popupContent, { maxWidth: 300, className: 'dark-popup' });

    // Draw change polygon
    if (feature.geometry && feature.geometry.type !== 'Point') {
        const poly = L.geoJSON(feature.geometry, {
            style: { color: getSeverityColor(p.severity_level), weight: 2, fillOpacity: 0.15, dashArray: '5,5' },
        });
        alertLayer.addLayer(poly);
    }

    return marker;
}

// ═══ Filters ═══
function filterAlerts(alerts) {
    return alerts.filter(f => {
        const p = f.properties;
        if (!activeFilters.severities.includes(p.severity_level)) return false;
        if (p.model_confidence < activeFilters.minConfidence) return false;
        if (activeFilters.zone && p.zone_type !== activeFilters.zone) return false;
        return true;
    });
}

function toggleSeverityFilter(el) {
    el.classList.toggle('active');
    const level = el.dataset.level;
    if (el.classList.contains('active')) {
        activeFilters.severities.push(level);
    } else {
        activeFilters.severities = activeFilters.severities.filter(s => s !== level);
    }
    renderAlerts();
}

function updateConfidence(val) {
    activeFilters.minConfidence = val / 100;
    document.getElementById('confidenceValue').textContent = val + '%';
    renderAlerts();
}

function filterByZone(zone) {
    activeFilters.zone = zone;
    renderAlerts();
}

// ═══ Layers ═══
function toggleLayer(name, visible) {
    const layers = { satellite: null, change: changeLayer, segmentation: segLayer, alerts: alertLayer, zoning: zoningLayer };
    const layer = layers[name];
    if (!layer) return;
    if (visible) map.addLayer(layer);
    else map.removeLayer(layer);
}

// ═══ Stats ═══
function updateStats() {
    const total = allAlerts.length;
    const critical = allAlerts.filter(f => f.properties.severity_level === 'CRITICAL').length;
    const totalArea = allAlerts.reduce((sum, f) => sum + (f.properties.area_m2 || 0), 0);

    document.querySelector('#statTotal .stat-value').textContent = total;
    document.querySelector('#statCritical .stat-value').textContent = critical;
    document.querySelector('#statArea .stat-value').textContent = (totalArea / 1e6).toFixed(2);
}

// ═══ Detail Panel ═══
function showDetail(props) {
    const panel = document.getElementById('detailPanel');
    const body = document.getElementById('detailBody');
    document.getElementById('detailTitle').textContent = `Alert ${props.alert_id}`;

    const severityPct = (props.severity_score * 100).toFixed(0);
    const sevColor = getSeverityColor(props.severity_level);

    body.innerHTML = `
        <div class="severity-bar"><div class="severity-fill" style="width:${severityPct}%;background:${sevColor}"></div></div>
        <div style="text-align:center;margin:8px 0;font-size:0.85rem">
            <span style="color:${sevColor};font-weight:700">${props.severity_level}</span>
            <span style="color:var(--text-muted)"> — ${severityPct}% severity</span>
        </div>
        <div class="detail-row"><span class="detail-label">Coordinates</span><span class="detail-value">${props.coordinates.latitude.toFixed(5)}°N, ${props.coordinates.longitude.toFixed(5)}°E</span></div>
        <div class="detail-row"><span class="detail-label">Area</span><span class="detail-value">${props.area_m2.toFixed(0)} m²</span></div>
        <div class="detail-row"><span class="detail-label">Zone</span><span class="detail-value">${formatZone(props.zone_type)}</span></div>
        <div class="detail-row"><span class="detail-label">Land Use</span><span class="detail-value">${formatZone(props.landuse)}</span></div>
        <div class="detail-row"><span class="detail-label">Violation</span><span class="detail-value">${props.violation_type.replace(/_/g, ' ')}</span></div>
        <div class="detail-row"><span class="detail-label">Confidence</span><span class="detail-value">${(props.model_confidence*100).toFixed(0)}%</span></div>
        <div class="detail-row"><span class="detail-label">Before Date</span><span class="detail-value">${props.temporal_range.before_date}</span></div>
        <div class="detail-row"><span class="detail-label">After Date</span><span class="detail-value">${props.temporal_range.after_date}</span></div>
        <div class="detail-row"><span class="detail-label">Within Parcel</span><span class="detail-value">${props.cadastral?.within_parcel ?? 'N/A'}</span></div>
        <div class="detail-row"><span class="detail-label">Crosses Boundary</span><span class="detail-value">${props.cadastral?.crosses_boundary ?? 'N/A'}</span></div>
        <div class="detail-row"><span class="detail-label">Status</span><span class="detail-value">${props.status}</span></div>
        <div style="margin-top:16px;display:flex;gap:8px">
            <button class="btn btn-primary" style="flex:1" onclick="updateAlertStatus('${props.alert_id}','reviewed')">Mark Reviewed</button>
            <button class="btn btn-ghost" style="flex:1" onclick="updateAlertStatus('${props.alert_id}','false_positive')">False Positive</button>
        </div>
    `;

    panel.classList.add('open');
}

function closeDetail() {
    document.getElementById('detailPanel').classList.remove('open');
}

// ═══ Analysis Modal ═══
function openAnalysisModal() {
    document.getElementById('analysisModal').classList.add('active');
}
function closeAnalysisModal(e) {
    if (e && e.target !== e.currentTarget) return;
    document.getElementById('analysisModal').classList.remove('active');
}

async function runAnalysis() {
    const payload = {
        t1_start: document.getElementById('t1Start').value,
        t1_end: document.getElementById('t1End').value,
        t2_start: document.getElementById('t2Start').value,
        t2_end: document.getElementById('t2End').value,
        max_cloud_pct: parseInt(document.getElementById('cloudCover').value),
        confidence_threshold: parseInt(document.getElementById('threshold').value) / 100,
    };

    try {
        const res = await fetch(`${API_BASE}/analysis/`, {
            method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload),
        });
        const data = await res.json();
        showToast(`Analysis ${data.analysis_id} queued`, 'success');
        closeAnalysisModal();
        pollAnalysis(data.analysis_id);
    } catch(e) {
        showToast('Failed to start analysis: ' + e.message, 'error');
    }
}

async function pollAnalysis(id) {
    const poll = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/analysis/${id}`);
            const data = await res.json();
            if (data.status === 'completed') {
                clearInterval(poll);
                showToast(`Analysis complete! ${data.num_alerts} alerts found`, 'success');
                refreshAlerts();
            } else if (data.status === 'failed') {
                clearInterval(poll);
                showToast('Analysis failed: ' + data.message, 'error');
            }
        } catch(e) { clearInterval(poll); }
    }, 5000);
}

async function updateAlertStatus(alertId, status) {
    try {
        await fetch(`${API_BASE}/alerts/${alertId}/status?status=${status}`, { method: 'PATCH' });
        showToast(`Alert ${alertId} marked as ${status}`, 'info');
        closeDetail();
        refreshAlerts();
    } catch(e) { showToast('Update failed', 'error'); }
}

// ═══ Demo Data ═══
function loadDemoAlerts() {
    allAlerts = [
        { type:'Feature', geometry:{type:'Polygon',coordinates:[[[77.52,12.91],[77.525,12.91],[77.525,12.915],[77.52,12.915],[77.52,12.91]]]}, properties:{ alert_id:'demo-0001', analysis_id:'demo', timestamp:new Date().toISOString(), severity_score:0.87, severity_level:'CRITICAL', coordinates:{latitude:12.9125,longitude:77.5225}, bbox:{west:77.52,south:12.91,east:77.525,north:12.915}, area_m2:2450, zone_type:'agricultural', landuse:'farmland', violation_type:'unauthorized_development_in_agricultural', model_confidence:0.82, temporal_range:{before_date:'2024-01-15',after_date:'2024-11-20'}, cadastral:{within_parcel:false,crosses_boundary:true,parcel_id:null}, status:'new' }},
        { type:'Feature', geometry:{type:'Polygon',coordinates:[[[77.61,12.97],[77.615,12.97],[77.615,12.975],[77.61,12.975],[77.61,12.97]]]}, properties:{ alert_id:'demo-0002', analysis_id:'demo', timestamp:new Date().toISOString(), severity_score:0.72, severity_level:'HIGH', coordinates:{latitude:12.9725,longitude:77.6125}, bbox:{west:77.61,south:12.97,east:77.615,north:12.975}, area_m2:1800, zone_type:'protected', landuse:'forest', violation_type:'unauthorized_development_in_protected', model_confidence:0.78, temporal_range:{before_date:'2024-02-01',after_date:'2024-10-30'}, cadastral:{within_parcel:false,crosses_boundary:false,parcel_id:null}, status:'new' }},
        { type:'Feature', geometry:{type:'Polygon',coordinates:[[[77.55,12.93],[77.553,12.93],[77.553,12.933],[77.55,12.933],[77.55,12.93]]]}, properties:{ alert_id:'demo-0003', analysis_id:'demo', timestamp:new Date().toISOString(), severity_score:0.51, severity_level:'MEDIUM', coordinates:{latitude:12.9315,longitude:77.5515}, bbox:{west:77.55,south:12.93,east:77.553,north:12.933}, area_m2:980, zone_type:'green_space', landuse:'grass', violation_type:'unauthorized_development_in_green_space', model_confidence:0.65, temporal_range:{before_date:'2024-03-01',after_date:'2024-12-01'}, cadastral:{within_parcel:true,crosses_boundary:false,parcel_id:'BLR-4521'}, status:'new' }},
        { type:'Feature', geometry:{type:'Polygon',coordinates:[[[77.68,12.88],[77.684,12.88],[77.684,12.884],[77.68,12.884],[77.68,12.88]]]}, properties:{ alert_id:'demo-0004', analysis_id:'demo', timestamp:new Date().toISOString(), severity_score:0.23, severity_level:'LOW', coordinates:{latitude:12.882,longitude:77.682}, bbox:{west:77.68,south:12.88,east:77.684,north:12.884}, area_m2:450, zone_type:'residential', landuse:'residential', violation_type:'permitted_zone', model_confidence:0.55, temporal_range:{before_date:'2024-01-10',after_date:'2024-09-15'}, cadastral:{within_parcel:true,crosses_boundary:false,parcel_id:'BLR-7783'}, status:'new' }},
    ];
    renderAlerts();
    updateStats();
}

// ═══ Utilities ═══
function getSeverityColor(level) {
    return { CRITICAL:'#ef4444', HIGH:'#f97316', MEDIUM:'#f59e0b', LOW:'#10b981' }[level] || '#6366f1';
}

function formatZone(zone) {
    if (!zone) return 'Unknown';
    return zone.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function showToast(message, type='info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 4500);
}
