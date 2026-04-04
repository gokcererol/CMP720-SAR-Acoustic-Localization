/**
 * Map Controller — Leaflet map with toggleable layers for nodes,
 * events, confidence ellipses, hyperbolas, and range grid.
 */

// Map state
let map;
let nodeLayers = {};
let nodePositions = {};           // {id: {lat, lon}}
let nodePositionsMeters = {};     // {id: {x, y}}
let eventMarkers = [];
let ellipseLayers = [];
let hyperbolaLayers = [];
let gridLayers = [];
let selectedEventId = null;
let hyperbolasVisible = false;
let gridVisible = false;

// Node colors
const NODE_COLORS = { 1: '#6366f1', 2: '#22c55e', 3: '#eab308', 4: '#ef4444' };
const HYPERBOLA_COLORS = ['#f472b6', '#38bdf8', '#a78bfa', '#34d399', '#fb923c', '#facc15'];
const STATUS_ICONS = {
    CONFIRMED: '✅', WEAK_DATA: '⚠️', REJECTED: '❌',
    OUT_OF_BOUNDS: '🔵', FILTERED: '🚫', LOG_ONLY: '📋'
};

const ML_CLASS_NAMES = {
    0: 'Whistle', 1: 'Human Voice', 2: 'Impact', 3: 'Knocking',
    4: 'Collapse', 5: 'Machinery', 6: 'Motor', 7: 'Animal',
    8: 'Wind', 9: 'Rain', 10: 'Ambient'
};
const ML_CLASS_ICONS = {
    0: '🔔', 1: '🗣️', 2: '💥', 3: '🔨',
    4: '🏚️', 5: '⚙️', 6: '🔧', 7: '🐕',
    8: '🌬️', 9: '🌧️', 10: '🔇'
};

// Geo reference (updated from loaded node positions)
let REF_LAT = 39.867449;
let REF_LON = 32.733585;
const SPEED_OF_SOUND = 343.0;

function latlonToMeters(lat, lon) {
    const x = (lon - REF_LON) * 111320.0 * Math.cos(REF_LAT * Math.PI / 180);
    const y = (lat - REF_LAT) * 111132.0;
    return { x, y };
}

function metersToLatlon(x, y) {
    const lat = REF_LAT + y / 111132.0;
    const lon = REF_LON + x / (111320.0 * Math.cos(REF_LAT * Math.PI / 180));
    return { lat, lon };
}

function initMap() {
    map = L.map('map', {
        center: [39.867449, 32.733585],
        zoom: 17,
        zoomControl: true,
    });

    // Dark tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap &copy; CARTO',
        maxZoom: 20,
    }).addTo(map);

    // Load node positions and add markers
    loadNodePositions();
}

async function loadNodePositions() {
    try {
        const resp = await fetch('/api/nodes/positions');
        const positions = await resp.json();

        const posList = Object.values(positions || {});
        if (posList.length > 0) {
            REF_LAT = posList.reduce((s, p) => s + Number(p.lat || 0), 0) / posList.length;
            REF_LON = posList.reduce((s, p) => s + Number(p.lon || 0), 0) / posList.length;
        }

        for (const [id, pos] of Object.entries(positions)) {
            const nid = parseInt(id);
            const color = NODE_COLORS[nid] || '#ffffff';

            nodePositions[nid] = pos;
            nodePositionsMeters[nid] = latlonToMeters(pos.lat, pos.lon);

            const marker = L.circleMarker([pos.lat, pos.lon], {
                radius: 10,
                fillColor: color,
                color: '#fff',
                weight: 2,
                fillOpacity: 0.9,
            }).addTo(map);

            marker.bindTooltip(`Node ${nid}`, {
                permanent: true, direction: 'top',
                className: 'node-tooltip', offset: [0, -12]
            });

            nodeLayers[nid] = marker;
        }

        // Fit map to node bounds with padding
        const lats = Object.values(positions).map(p => p.lat);
        const lons = Object.values(positions).map(p => p.lon);
        const bounds = L.latLngBounds(
            [Math.min(...lats) - 0.001, Math.min(...lons) - 0.001],
            [Math.max(...lats) + 0.001, Math.max(...lons) + 0.001]
        );
        map.fitBounds(bounds, { padding: [50, 50] });
    } catch (e) {
        console.error('Failed to load node positions:', e);
    }
}

// ===== HYPERBOLA COMPUTATION =====

/**
 * Compute a TDoA hyperbola curve between two nodes.
 * The hyperbola is the locus of points where the difference in distance
 * to the two nodes equals c * tdoa.
 */
function computeHyperbola(node1, node2, tdoa_sec, numPoints = 150) {
    const p1 = nodePositionsMeters[node1];
    const p2 = nodePositionsMeters[node2];
    if (!p1 || !p2) return [];

    // Midpoint and rotation
    const mx = (p1.x + p2.x) / 2;
    const my = (p1.y + p2.y) / 2;
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const d = Math.sqrt(dx * dx + dy * dy);
    const angle = Math.atan2(dy, dx);

    const rangeDiff = tdoa_sec * SPEED_OF_SOUND;

    // Hyperbola parameters
    const a = Math.abs(rangeDiff) / 2;
    if (a <= 0 || a >= d / 2) return [];
    const c_hyp = d / 2;
    const b = Math.sqrt(c_hyp * c_hyp - a * a);

    // Determine which branch based on signed range difference.
    const sign = rangeDiff >= 0 ? -1 : 1;

    const points = [];
    const tRange = 3.0; // parameter range

    for (let i = 0; i < numPoints; i++) {
        const t = -tRange + (2 * tRange * i) / (numPoints - 1);

        // Hyperbola in local frame (along x-axis)
        const hx = sign * a * Math.cosh(t);
        const hy = b * Math.sinh(t);

        // Rotate and translate to global
        const gx = mx + hx * Math.cos(angle) - hy * Math.sin(angle);
        const gy = my + hx * Math.sin(angle) + hy * Math.cos(angle);

        // Convert to lat/lon and check bounds
        const ll = metersToLatlon(gx, gy);

        // Clip to reasonable extent around the node field
        const distFromCenter = Math.sqrt(gx * gx + gy * gy);
        if (distFromCenter < 500) {
            points.push([ll.lat, ll.lon]);
        }
    }

    return points;
}

/**
 * Draw hyperbolas for a specific event using its per-node timestamps.
 */
function drawHyperbolas(event) {
    clearHyperbolas();

    if (!event || !event.packets) return;

    const packets = event.packets;
    const nodeIds = Object.keys(packets).map(Number).sort();
    if (nodeIds.length < 2) return;

    // Use solver-selected reference node when available; otherwise fallback to earliest timestamp.
    let refNode = event?.solver_result?.ref_node;
    if (refNode == null || !nodeIds.includes(Number(refNode))) {
        refNode = nodeIds.reduce((best, nid) => {
            if (best == null) return nid;
            const tsBest = (packets[best] || packets[String(best)] || {}).ts_micros || Number.MAX_SAFE_INTEGER;
            const tsCur = (packets[nid] || packets[String(nid)] || {}).ts_micros || Number.MAX_SAFE_INTEGER;
            return tsCur < tsBest ? nid : best;
        }, null);
    }
    refNode = Number(refNode);
    const refPkt = packets[refNode] || packets[String(refNode)];
    if (!refPkt) return;

    let colorIdx = 0;
    for (const otherNode of nodeIds) {
        if (otherNode === refNode) continue;

        const pktOther = packets[otherNode] || packets[String(otherNode)];
        if (!pktOther) continue;

        const tdoa = (pktOther.ts_micros - refPkt.ts_micros) / 1_000_000.0;
        const points = computeHyperbola(refNode, otherNode, tdoa);

        if (points.length > 2) {
            const color = HYPERBOLA_COLORS[colorIdx % HYPERBOLA_COLORS.length];
            const line = L.polyline(points, {
                color: color,
                weight: 2.5,
                opacity: 0.75,
                dashArray: '6 4',
                className: 'hyperbola-line',
            }).addTo(map);

            line.bindTooltip(`N${refNode}↔N${otherNode}  ΔT=${(tdoa * 1e6).toFixed(0)}µs`, {
                sticky: true,
                className: 'hyperbola-tooltip',
            });

            hyperbolaLayers.push(line);
            colorIdx++;
        }
    }
}

function clearHyperbolas() {
    hyperbolaLayers.forEach(l => map.removeLayer(l));
    hyperbolaLayers = [];
}

// ===== RANGE GRID =====
function drawRangeGrid() {
    clearGrid();
    // Draw concentric range circles from center
    const center = [REF_LAT, REF_LON];
    const radii = [25, 50, 75, 100, 150, 200];
    radii.forEach(r => {
        const circ = L.circle(center, {
            radius: r,
            fill: false,
            color: 'rgba(100, 116, 139, 0.25)',
            weight: 1,
            dashArray: '2 4',
        }).addTo(map);

        // Label
        const lbl = metersToLatlon(0, r);
        L.marker([lbl.lat, lbl.lon], {
            icon: L.divIcon({
                className: 'grid-label',
                html: `<span>${r}m</span>`,
                iconSize: [30, 14],
            })
        }).addTo(map);

        gridLayers.push(circ);
    });
}

function clearGrid() {
    gridLayers.forEach(l => map.removeLayer(l));
    gridLayers = [];
}

// ===== EVENT MARKERS =====
function addEventToMap(event) {
    const sr = event.solver_result;
    if (!sr || !sr.success) return;

    const lat = sr.lat;
    const lon = sr.lon;
    const status = event.filter_status || 'UNKNOWN';
    const eventId = event.event_id || eventMarkers.length + 1;

    // Status-based styling
    let color, icon;
    switch (status) {
        case 'CONFIRMED': color = '#22c55e'; icon = '✅'; break;
        case 'WEAK_DATA': color = '#eab308'; icon = '⚠️'; break;
        case 'REJECTED': color = '#ef4444'; icon = '❌'; break;
        case 'OUT_OF_BOUNDS': color = '#3b82f6'; icon = '🔵'; break;
        default: color = '#94a3b8'; icon = '❓';
    }

    // Dominant ML class from packets
    let className = '', classIcon = '';
    if (event.packets) {
        const pkts = Object.values(event.packets);
        if (pkts.length > 0) {
            const cls = pkts[0].ml_class ?? 0;
            className = ML_CLASS_NAMES[cls] || 'Unknown';
            classIcon = ML_CLASS_ICONS[cls] || '❓';
        }
    }

    // Event marker with pulse animation on creation
    const marker = L.circleMarker([lat, lon], {
        radius: 8,
        fillColor: color,
        color: '#fff',
        weight: 2,
        fillOpacity: 0.9,
    }).addTo(map);

    // Build rich popup
    const errorStr = event.position_error_m ? `${event.position_error_m.toFixed(1)}m` : 'N/A';
    const popupHtml = `
        <div class="map-popup">
            <div class="popup-header">
                <span class="popup-icon">${classIcon}</span>
                <strong>${icon} Event #${eventId}</strong>
            </div>
            <div class="popup-grid">
                <span class="popup-label">Status</span><span class="popup-value">${status}</span>
                <span class="popup-label">Class</span><span class="popup-value">${className}</span>
                <span class="popup-label">GDOP</span><span class="popup-value">${(event.gdop || 0).toFixed(1)} (${event.gdop_label || ''})</span>
                <span class="popup-label">Error</span><span class="popup-value">${errorStr}</span>
                <span class="popup-label">Method</span><span class="popup-value">${sr.method || 'n/a'}</span>
                <span class="popup-label">Compute</span><span class="popup-value">${(sr.compute_time_ms || 0).toFixed(1)}ms</span>
                <span class="popup-label">Packets</span><span class="popup-value">${event.num_packets || 0}/4</span>
            </div>
            <div class="popup-actions">
                <button onclick="selectEventOnMap(${eventId})">🔎 Hyperbolas</button>
                <button onclick="showEventDetail(${eventId})">📊 Full Detail</button>
            </div>
        </div>
    `;
    marker.bindPopup(popupHtml, { maxWidth: 280, className: 'dark-popup' });

    // Ground truth target line
    let gtMarker = null;
    let errLine = null;
    if (event.ground_truth) {
        const gtLat = event.ground_truth.lat;
        const gtLon = event.ground_truth.lon;
        
        errLine = L.polyline([[gtLat, gtLon], [lat, lon]], {
            color: '#facc15',
            weight: 1.5,
            dashArray: '3 4',
            opacity: 0.6
        }).addTo(map);

        gtMarker = L.circleMarker([gtLat, gtLon], {
            radius: 3,
            fillColor: '#ffffff',
            color: '#facc15',
            weight: 1,
            fillOpacity: 1
        }).addTo(map);
        gtMarker.bindTooltip("Ground Truth Source", {className: 'hyperbola-tooltip', direction: 'top', offset: [0, -3]});
    }

    eventMarkers.push({ marker, event, id: eventId, gtMarker, errLine });

    // Confidence ellipse
    if (event.confidence_ellipse && event.confidence_ellipse.semi_major_90) {
        const ellipse = event.confidence_ellipse;
        const semi90 = Math.min(ellipse.semi_major_90, 200);
        const circ = L.circle([lat, lon], {
            radius: semi90,
            fillColor: color,
            fillOpacity: 0.08,
            color: color,
            weight: 1,
            dashArray: '4 4',
        }).addTo(map);
        ellipseLayers.push(circ);
    }

    // Auto-draw hyperbolas if visible and this is the selected event
    if (hyperbolasVisible && selectedEventId === eventId) {
        drawHyperbolas(event);
    }
}

// ===== EVENT SELECTION & FOCUS =====

function selectEventOnMap(eventId) {
    selectedEventId = eventId;

    // Highlight the selected event card in the sidebar
    document.querySelectorAll('.event-card').forEach(c => c.classList.remove('selected'));
    const cards = document.querySelectorAll('.event-card');
    cards.forEach(c => {
        const idSpan = c.querySelector('.event-id');
        if (idSpan && idSpan.textContent === `#${eventId}`) {
            c.classList.add('selected');
        }
    });

    // Highlight marker
    eventMarkers.forEach(e => {
        if (e.id === eventId) {
            e.marker.setStyle({ weight: 3, radius: 11, color: '#f0abfc' });
        } else {
            e.marker.setStyle({ weight: 2, radius: 8, color: '#fff' });
        }
    });

    // Draw hyperbolas for this event
    const entry = eventMarkers.find(e => e.id === eventId);
    if (entry) {
        drawHyperbolas(entry.event);
        hyperbolasVisible = true;
        const cb = document.getElementById('layer-hyperbolas');
        if (cb) cb.checked = true;
    }
}

function focusOnEvent(eventId) {
    const entry = eventMarkers.find(e => e.id === eventId);
    if (!entry) return;

    const sr = entry.event.solver_result;
    if (sr && sr.lat && sr.lon) {
        map.setView([sr.lat, sr.lon], 18, { animate: true, duration: 0.6 });
        entry.marker.openPopup();
        selectEventOnMap(eventId);
    }
}

function clearMap() {
    eventMarkers.forEach(e => {
        map.removeLayer(e.marker);
        if (e.gtMarker) map.removeLayer(e.gtMarker);
        if (e.errLine) map.removeLayer(e.errLine);
    });
    ellipseLayers.forEach(l => map.removeLayer(l));
    clearHyperbolas();
    clearGrid();
    eventMarkers = [];
    ellipseLayers = [];
    selectedEventId = null;
}

// Toggle layers
function toggleNodeLayer(visible) {
    Object.values(nodeLayers).forEach(m => {
        if (visible) m.addTo(map);
        else map.removeLayer(m);
    });
}

function toggleEventLayer(visible) {
    eventMarkers.forEach(e => {
        if (visible) {
            e.marker.addTo(map);
            if (e.gtMarker) e.gtMarker.addTo(map);
            if (e.errLine) e.errLine.addTo(map);
        } else {
            map.removeLayer(e.marker);
            if (e.gtMarker) map.removeLayer(e.gtMarker);
            if (e.errLine) map.removeLayer(e.errLine);
        }
    });
}

function toggleEllipseLayer(visible) {
    ellipseLayers.forEach(l => {
        if (visible) l.addTo(map);
        else map.removeLayer(l);
    });
}

function toggleHyperbolaLayer(visible) {
    hyperbolasVisible = visible;
    if (visible && selectedEventId) {
        const entry = eventMarkers.find(e => e.id === selectedEventId);
        if (entry) drawHyperbolas(entry.event);
    } else {
        clearHyperbolas();
    }
}

function toggleGridLayer(visible) {
    gridVisible = visible;
    if (visible) drawRangeGrid();
    else clearGrid();
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    initMap();

    document.getElementById('layer-hyperbolas')?.addEventListener('change', e => toggleHyperbolaLayer(e.target.checked));
    document.getElementById('layer-grid')?.addEventListener('change', e => toggleGridLayer(e.target.checked));
});
