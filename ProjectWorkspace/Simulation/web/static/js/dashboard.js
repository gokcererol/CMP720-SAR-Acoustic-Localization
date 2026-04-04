/**
 * Dashboard — Event history, scenario runner, node status, and polling loop.
 * Enhanced with rich event detail modals and event selection.
 */

let pollingInterval = null;
let startTime = Date.now();
let knownEventCount = 0;
let allEvents = [];  // Store full event data

const DETAIL_TOOLTIPS = {
    "Position": "Estimated GPS coordinates (Lat/Lon) of the sound source found by the solver.",
    "Position Error": "Distance between the estimated position and the known ground truth location.",
    "GDOP": "Geometric Dilution of Precision. Measures how sensor geometry affects accuracy. Lower is better (< 5.0).",
    "Solver Method": "The specific mathematical algorithm used to triangulate the location (e.g., Chan-Ho, Taylor Series, LM).",
    "Reference Node": "Node selected as the TDoA reference for this solve (earliest capture policy).",
    "Compute Time": "Total time taken by the Jetson central unit to solve the TDoA equations for this event.",
    "Residual": "The root-mean-square error of the TDoA equations. Lower values indicate a more mathematically consistent solution.",
    "Packets Received": "Number of sensor nodes that successfully detected this sound and transmitted data via LoRa/UDP.",
    "Sound Classification": "The type of sound identified by the node-side TinyML model running on the ESP32-S3.",
    "Node": "Sensor node identifier (N1-N4).",
    "Timestamp (µs)": "The precise microsecond timestamp when the sound wave front was detected by this node.",
    "ΔT from Ref": "Time Difference of Arrival relative to the solver-selected reference node.",
    "Magnitude": "Log-scaled peak amplitude of the detected sound wave.",
    "SNR": "Signal-to-Noise Ratio. Measures signal strength vs background noise. Higher is better.",
    "ML Class": "The specific sound category identified locally on the node using TinyML.",
    "Conf": "The probability confidence (%) assigned by the TinyML model on the node.",
    "Bat": "Current battery level of the remote sensor node.",
    "Freq": "The dominant frequency component (Hz) detected in the audio snippet.",
    "Semi-Major": "The longest radius of the 90% confidence uncertainty ellipse.",
    "Semi-Minor": "The shortest radius of the 90% confidence uncertainty ellipse.",
    "Azimuth": "The orientation angle of the uncertainty area relative to true North.",
    "CEP (50%)": "Circular Error Probable. The radius within which 50% of estimations are expected to fall.",
    "Status": "The outcome of the solver (OK or ERROR/FAILED)."
};

document.addEventListener('DOMContentLoaded', () => {
    loadScenarios();
    startPolling();
    bindTestButtons();
    bindEventPanelButtons();
});

// ===== SCENARIO LOADER =====
async function loadScenarios() {
    try {
        const resp = await fetch('/api/scenarios');
        const scenarios = await resp.json();
        const select = document.getElementById('scenario-select');
        if (!select) return;

        scenarios.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s.name;
            opt.textContent = `${s.name} [${s.expected_status}]`;
            select.appendChild(opt);
        });
    } catch (e) {
        console.error('Failed to load scenarios:', e);
    }
}

// ===== TEST BUTTONS =====
function bindTestButtons() {
    // Fire selected scenario
    document.getElementById('btn-fire-scenario')?.addEventListener('click', () => {
        const name = document.getElementById('scenario-select')?.value;
        if (!name) return alert('Select a scenario first!');

        fetch('/api/scenarios/fire', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
        }).then(r => r.json())
          .then(d => {
              console.log('🔥 Scenario fired:', d);
              if (!d.success) {
                  alert(`Scenario fire failed: ${d.error || 'Unknown error'}`);
                  return;
              }
              if (typeof injectWaveform === 'function' && d.info) {
                  injectWaveform(d.info.sound_type || 'whistle', d.info.amplitude || 0.8);
              }
          })
          .catch(console.error);
    });

    // Random scenario
    document.getElementById('btn-random')?.addEventListener('click', () => {
        const select = document.getElementById('scenario-select');
        const opts = select.querySelectorAll('option');
        if (opts.length <= 1) return;
        const idx = Math.floor(Math.random() * (opts.length - 1)) + 1;
        select.selectedIndex = idx;
        document.getElementById('btn-fire-scenario')?.click();
    });

    // Run all
    document.getElementById('btn-run-all')?.addEventListener('click', () => {
        fetch('/api/scenarios/run_batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ category: 'all', delay_sec: 3.0 }),
        }).then(r => r.json())
          .then(d => console.log('▶▶ Batch started:', d))
          .catch(console.error);
    });

    // Category buttons
    document.querySelectorAll('.category-btns .btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const cat = btn.dataset.cat;
            fetch('/api/scenarios/run_batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ category: cat, delay_sec: 2.5 }),
            }).then(r => r.json())
              .then(d => console.log(`▶▶ Category ${cat} started:`, d))
              .catch(console.error);
        });
    });
}

function bindEventPanelButtons() {
    document.getElementById('btn-clear-events')?.addEventListener('click', () => {
        clearEventHistoryUI();
    });
}

function clearEventHistoryUI() {
    allEvents = [];
    knownEventCount = 0;

    const list = document.getElementById('event-list');
    if (list) {
        list.innerHTML = '<p class="empty-state">No events yet. Fire a test scenario!</p>';
    }

    const badge = document.getElementById('event-count-badge');
    if (badge) badge.textContent = '0';

    const passEl = document.getElementById('test-pass');
    const failEl = document.getElementById('test-fail');
    if (passEl) passEl.textContent = '✅ 0';
    if (failEl) failEl.textContent = '❌ 0';

    if (typeof clearMap === 'function') {
        clearMap();
    }

    // Keep polling in sync: do not repopulate old server-side events after clearing UI.
    fetch('/api/events', { cache: 'no-store' })
        .then(r => r.json())
        .then(events => {
            knownEventCount = Array.isArray(events) ? events.length : 0;
        })
        .catch(() => {
            knownEventCount = 0;
        });
}

// ===== POLLING LOOP =====
function startPolling() {
    pollingInterval = setInterval(async () => {
        try {
            // Poll status
            const statusResp = await fetch('/api/status', { cache: 'no-store' });
            const status = await statusResp.json();
            updateNodeStatus(status.nodes);
            updateHeaderStats(status);

            // Poll events
            const eventsResp = await fetch('/api/events', { cache: 'no-store' });
            const events = await eventsResp.json();

            if (events.length > knownEventCount) {
                // New events arrived
                for (let i = knownEventCount; i < events.length; i++) {
                    allEvents.push(events[i]);
                    addEventToHistory(events[i]);
                    if (typeof addEventToMap === 'function') {
                        addEventToMap(events[i]);
                    }
                }
                knownEventCount = events.length;
                document.getElementById('event-count-badge').textContent = knownEventCount;
            }

        } catch (e) {
            // Server not reachable
        }
    }, 1500);
}

// ===== UPDATE DISPLAY =====
function updateHeaderStats(status) {
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const mins = Math.floor(elapsed / 60);
    const secs = elapsed % 60;
    document.getElementById('stat-uptime').innerHTML = `Uptime: <strong>${mins}m ${secs}s</strong>`;

    const totalEvents = status.solver?.total_events || 0;
    document.getElementById('stat-events').innerHTML = `Events: <strong>${totalEvents}</strong>`;

    const totalPackets = status.lora?.packets_received || 0;
    document.getElementById('stat-packets').innerHTML = `Packets: <strong>${totalPackets}</strong>`;
}

function updateNodeStatus(nodes) {
    if (!nodes) return;
    for (const [nid, info] of Object.entries(nodes)) {
        const card = document.querySelector(`.node-card[data-node="${nid}"]`);
        if (!card) continue;

        // Battery
        const bat = info.battery_pct || 100;
        const fill = card.querySelector('.battery-fill');
        fill.style.width = bat + '%';
        fill.style.background = bat > 30 ? '#22c55e' : bat > 10 ? '#eab308' : '#ef4444';

        // GPS
        const gps = card.querySelector('.node-gps');
        gps.textContent = info.gps_locked ? '🛰️' : '⏳';

        // Stats
        card.querySelector('.node-events').textContent = `${info.events_transmitted || 0} events`;
        const drift = info.clock_drift_us || 0;
        card.querySelector('.node-drift').textContent = `${drift.toFixed(1)}µs`;
    }
}

// ===== ML CLASS LABELS =====
// ML_CLASS_NAMES and ML_CLASS_ICONS are defined in map.js

function addEventToHistory(event) {
    const list = document.getElementById('event-list');
    if (!list) return;

    // Remove empty state
    const empty = list.querySelector('.empty-state');
    if (empty) empty.remove();

    const sr = event.solver_result || {};
    const status = event.filter_status || 'UNKNOWN';
    const eventId = event.event_id || knownEventCount;

    let statusClass = '';
    let statusIcon = '';
    switch (status) {
        case 'CONFIRMED': statusClass = 'confirmed'; statusIcon = '✅'; break;
        case 'WEAK_DATA': statusClass = 'weak'; statusIcon = '⚠️'; break;
        case 'REJECTED': statusClass = 'rejected'; statusIcon = '❌'; break;
        case 'OUT_OF_BOUNDS': statusClass = 'oob'; statusIcon = '🔵'; break;
        case 'FILTERED': statusClass = 'filtered'; statusIcon = '🚫'; break;
        default: statusClass = ''; statusIcon = '❓';
    }

    // Determine dominant ML class from packets
    let className = '', classIcon = '';
    let avgConf = 0;
    if (event.packets) {
        const pkts = Object.values(event.packets);
        if (pkts.length > 0) {
            const cls = pkts[0].ml_class ?? 0;
            className = ML_CLASS_NAMES[cls] || 'Unknown';
            classIcon = ML_CLASS_ICONS[cls] || '❓';
            avgConf = Math.round(pkts.reduce((s, p) => s + (p.ml_confidence || 0), 0) / pkts.length);
        }
    }

    const errorStr = event.position_error_m != null ? `${event.position_error_m.toFixed(1)}m` : '—';
    const gdopStr = event.gdop ? event.gdop.toFixed(1) : '—';
    const methodStr = sr.method ? sr.method.replace('pipeline(', '').replace(')', '') : '—';
    const packets = event.num_packets || 0;

    const card = document.createElement('div');
    card.className = `event-card ${statusClass}`;
    card.dataset.eventId = eventId;
    card.innerHTML = `
        <div class="event-header">
            <span class="event-id">#${eventId}</span>
            <span class="event-status ${statusClass}">${statusIcon} ${status}</span>
        </div>
        <div class="event-summary">
            <span class="event-class">${classIcon} ${className}</span>
            <span class="event-conf">${avgConf}%</span>
        </div>
        <div class="event-detail">
            📦 ${packets}/4 &nbsp;|&nbsp; GDOP: ${gdopStr} &nbsp;|&nbsp; Err: ${errorStr} &nbsp;|&nbsp; ${methodStr}
        </div>
        <div class="event-actions">
            <button onclick="focusOnEvent(${eventId}); event.stopPropagation();">📍 Map</button>
            <button onclick="showEventDetail(${eventId}); event.stopPropagation();">📊 Detail</button>
            <button onclick="selectEventOnMap(${eventId}); event.stopPropagation();">📐 Hyperbolas</button>
        </div>
    `;

    // Click on card to focus
    card.addEventListener('click', () => focusOnEvent(eventId));

    list.prepend(card);

    // Update pass/fail counters
    updateTestResults();
}

function updateTestResults() {
    const cards = document.querySelectorAll('.event-card');
    let pass = 0, fail = 0;
    cards.forEach(c => {
        if (c.classList.contains('confirmed') || c.classList.contains('weak')) pass++;
        else if (c.classList.contains('rejected') || c.classList.contains('oob') || c.classList.contains('filtered')) fail++;
    });
    document.getElementById('test-pass').textContent = `✅ ${pass}`;
    document.getElementById('test-fail').textContent = `❌ ${fail}`;
}

// ===== EVENT DETAIL MODAL (ENHANCED) =====
function showEventDetail(eventId) {
    fetch('/api/events', { cache: 'no-store' }).then(r => r.json()).then(events => {
        const event = events.find(e => e.event_id === eventId) || events[eventId - 1];
        if (!event) return;

        const modal = document.getElementById('event-modal');
        const title = document.getElementById('modal-title');
        const body = document.getElementById('modal-body');

        const sr = event.solver_result || {};
        const status = event.filter_status || 'UNKNOWN';
        const packetsForRef = event.packets || {};
        const packetNodeIds = Object.keys(packetsForRef).sort();
        const refNodeId = sr.ref_node != null
            ? Number(sr.ref_node)
            : Number(packetNodeIds.reduce((best, nid) => {
                if (!best) return nid;
                const tsBest = packetsForRef[best]?.ts_micros || Number.MAX_SAFE_INTEGER;
                const tsCur = packetsForRef[nid]?.ts_micros || Number.MAX_SAFE_INTEGER;
                return tsCur < tsBest ? nid : best;
            }, null));

        // Determine ML class
        let className = 'Unknown', classIcon = '❓', classId = -1;
        if (event.packets) {
            const pkts = Object.values(event.packets);
            if (pkts.length > 0) {
                classId = pkts[0].ml_class ?? 0;
                className = ML_CLASS_NAMES[classId] || 'Unknown';
                classIcon = ML_CLASS_ICONS[classId] || '❓';
            }
        }

        let statusIcon = '';
        switch (status) {
            case 'CONFIRMED': statusIcon = '✅'; break;
            case 'WEAK_DATA': statusIcon = '⚠️'; break;
            case 'REJECTED': statusIcon = '❌'; break;
            case 'OUT_OF_BOUNDS': statusIcon = '🔵'; break;
            default: statusIcon = '❓';
        }

        title.innerHTML = `${statusIcon} Event #${eventId} <span class="modal-subtitle">${classIcon} ${className} — ${status}</span>`;

        let html = '';

        // ===== SECTION 1: OVERVIEW =====
        html += '<div class="detail-section">';
        html += '<h4>📋 Overview</h4>';
        html += '<div class="detail-grid">';

        if (sr.success) {
            html += `
                <div class="detail-item">
                    <span class="detail-label tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['Position']}">Position</span>
                    <span class="detail-value mono">${sr.lat?.toFixed(6)}, ${sr.lon?.toFixed(6)}</span>
                </div>`;
            if (event.position_error_m != null) {
                const errColor = event.position_error_m < 5 ? '#22c55e' : event.position_error_m < 15 ? '#eab308' : '#ef4444';
                html += `
                <div class="detail-item">
                    <span class="detail-label tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['Position Error']}">Position Error</span>
                    <span class="detail-value" style="color:${errColor};font-weight:600;">${event.position_error_m.toFixed(2)}m</span>
                </div>`;
            }
        }

        html += `
            <div class="detail-item">
                <span class="detail-label tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['GDOP']}">GDOP</span>
                <span class="detail-value"><span class="gdop-badge" style="background:${event.gdop_color || '#64748b'}">${(event.gdop || 0).toFixed(1)}</span> ${event.gdop_label || ''}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['Solver Method']}">Solver Method</span>
                <span class="detail-value mono">${sr.method || 'n/a'}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['Reference Node']}">Reference Node</span>
                <span class="detail-value mono">${Number.isFinite(refNodeId) ? `N${refNodeId}` : 'n/a'}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['Compute Time']}">Compute Time</span>
                <span class="detail-value mono">${(sr.compute_time_ms || 0).toFixed(2)}ms</span>
            </div>
            <div class="detail-item">
                <span class="detail-label tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['Residual']}">Residual</span>
                <span class="detail-value mono">${sr.residual != null ? sr.residual.toFixed(6) : 'n/a'}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['Packets Received']}">Packets Received</span>
                <span class="detail-value">${event.num_packets || 0} / 4</span>
            </div>
            <div class="detail-item">
                <span class="detail-label tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['Sound Classification']}">Sound Classification</span>
                <span class="detail-value">${classIcon} ${className} (ID: ${classId})</span>
            </div>
        `;
        html += '</div></div>';

        // ===== SECTION 2: PER-NODE TELEMETRY =====
        if (event.packets) {
            html += '<div class="detail-section">';
            html += '<h4>📡 Per-Node Telemetry</h4>';
            html += '<table class="detail-table">';
            html += `<thead><tr>
                <th class="tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['Node']}">Node</th>
                <th class="tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['Timestamp (µs)']}">Timestamp (µs)</th>
                <th class="tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['ΔT from Ref']}">ΔT from Ref</th>
                <th class="tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['Magnitude']}">Magnitude</th>
                <th class="tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['SNR']}">SNR</th>
                <th class="tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['ML Class']}">ML Class</th>
                <th class="tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['Conf']}">Conf</th>
                <th class="tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['Bat']}">Bat</th>
                <th class="tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['Freq']}">Freq</th>
            </tr></thead><tbody>`;

            const packets = event.packets;
            const nodeIds = Object.keys(packets).sort();
            const solverRef = event?.solver_result?.ref_node;
            const refId = (solverRef != null && packets[String(solverRef)])
                ? String(solverRef)
                : nodeIds.reduce((best, nid) => {
                    if (!best) return nid;
                    const tsBest = packets[best]?.ts_micros || Number.MAX_SAFE_INTEGER;
                    const tsCur = packets[nid]?.ts_micros || Number.MAX_SAFE_INTEGER;
                    return tsCur < tsBest ? nid : best;
                }, null);
            const refTs = packets[refId]?.ts_micros || 0;

            for (const nid of nodeIds) {
                const pkt = packets[nid];
                const dt = (pkt.ts_micros || 0) - refTs;
                const dtStr = dt === 0 ? '<span style="color:#64748b">ref</span>' : `<span style="color:#38bdf8">${dt > 0 ? '+' : ''}${dt}µs</span>`;
                const cls = pkt.ml_class ?? 0;
                const clsName = ML_CLASS_NAMES[cls] || '?';
                const clsIcon = ML_CLASS_ICONS[cls] || '';
                const snrColor = (pkt.snr_db || 0) > 15 ? '#22c55e' : (pkt.snr_db || 0) > 5 ? '#eab308' : '#ef4444';

                html += `<tr>
                    <td><span style="color:${NODE_COLORS[parseInt(nid)] || '#fff'};font-weight:700;">N${nid}</span></td>
                    <td class="mono">${pkt.ts_micros || 'n/a'}</td>
                    <td>${dtStr}</td>
                    <td class="mono">${pkt.magnitude || 0}</td>
                    <td><span style="color:${snrColor}" class="mono">${pkt.snr_db || 0}dB</span></td>
                    <td>${clsIcon} ${clsName}</td>
                    <td class="mono">${pkt.ml_confidence || 0}%</td>
                    <td class="mono">${pkt.battery_pct || 0}%</td>
                    <td class="mono">${pkt.peak_freq_hz || 0}Hz</td>
                </tr>`;
            }
            html += '</tbody></table></div>';

            // ===== SECTION 2b: TDOA MATRIX =====
            if (nodeIds.length >= 2) {
                html += '<div class="detail-section">';
                html += '<h4>📐 TDoA Difference Matrix</h4>';
                html += '<table class="detail-table tdoa-matrix">';
                html += '<thead><tr><th></th>';
                nodeIds.forEach(n => html += `<th>N${n}</th>`);
                html += '</tr></thead><tbody>';

                for (const n1 of nodeIds) {
                    html += `<tr><td><strong>N${n1}</strong></td>`;
                    for (const n2 of nodeIds) {
                        if (n1 === n2) {
                            html += '<td class="matrix-diag">—</td>';
                        } else {
                            const dt = ((packets[n2]?.ts_micros || 0) - (packets[n1]?.ts_micros || 0));
                            const distDiff = (dt / 1e6 * 343).toFixed(2);
                            html += `<td class="mono"><span style="font-size:11px;color:#94a3b8">${dt}µs</span><br><span style="font-size:10px;color:#64748b">${distDiff}m</span></td>`;
                        }
                    }
                    html += '</tr>';
                }
                html += '</tbody></table></div>';
            }
        }

        // ===== SECTION 3: SOLVER COMPARISON =====
        if (event.all_methods && Object.keys(event.all_methods).length > 0) {
            html += '<div class="detail-section">';
            html += '<h4>🧮 Solver Method Comparison</h4>';
            html += '<table class="detail-table">';
            html += `<thead><tr>
                <th class="tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['Solver Method']}">Method</th>
                <th class="tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['Position']}">Position (m)</th>
                <th class="tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['Residual']}">Residual</th>
                <th class="tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['Compute Time']}">Time</th>
                <th class="tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['Status']}">Status</th>
            </tr></thead><tbody>`;
            for (const [method, result] of Object.entries(event.all_methods)) {
                if (result.success) {
                    const isBest = sr.method && sr.method.includes(method.replace('_', ''));
                    html += `<tr${isBest ? ' class="row-highlight"' : ''}>
                        <td>${method}${isBest ? ' ⭐' : ''}</td>
                        <td class="mono">(${result.x?.toFixed(2)}, ${result.y?.toFixed(2)})</td>
                        <td class="mono">${result.residual?.toFixed(6)}</td>
                        <td class="mono">${(result.compute_time_ms || 0).toFixed(2)}ms</td>
                        <td style="color:#22c55e">✅ OK</td>
                    </tr>`;
                } else {
                    html += `<tr style="opacity:0.5">
                        <td>${method}</td>
                        <td colspan="3">${result.reason || 'failed'}</td>
                        <td style="color:#ef4444">❌</td>
                    </tr>`;
                }
            }
            html += '</tbody></table></div>';
        }

        // ===== SECTION 4: CONFIDENCE ELLIPSE =====
        if (event.confidence_ellipse && event.confidence_ellipse.semi_major_90) {
            const ell = event.confidence_ellipse;
            html += '<div class="detail-section">';
            html += '<h4>🎯 Confidence Ellipse (90%)</h4>';
            html += '<div class="detail-grid">';
            html += `
                <div class="detail-item">
                    <span class="detail-label tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['Semi-Major']}">Semi-Major</span>
                    <span class="detail-value mono">${ell.semi_major_90?.toFixed(2)}m</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['Semi-Minor']}">Semi-Minor</span>
                    <span class="detail-value mono">${ell.semi_minor_90?.toFixed(2) || 'n/a'}m</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['Azimuth']}">Azimuth</span>
                    <span class="detail-value mono">${ell.azimuth_deg?.toFixed(1) || 'n/a'}°</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label tooltip-label" data-tooltip="${DETAIL_TOOLTIPS['CEP (50%)']}">CEP (50%)</span>
                    <span class="detail-value mono">${ell.cep_50?.toFixed(2) || 'n/a'}m</span>
                </div>
            `;
            html += '</div></div>';
        }

        // ===== SECTION 5: FILTER PIPELINE =====
        if ((event.filter_reasons && event.filter_reasons.length) ||
            (event.filter_warnings && event.filter_warnings.length)) {
            html += '<div class="detail-section">';
            html += '<h4>🔍 Filter Pipeline Results</h4>';

            if (event.filter_reasons?.length) {
                event.filter_reasons.forEach(r => {
                    html += `<div class="filter-item danger">🚫 ${r}</div>`;
                });
            }
            if (event.filter_warnings?.length) {
                event.filter_warnings.forEach(w => {
                    html += `<div class="filter-item warning">⚠️ ${w}</div>`;
                });
            }
            html += '</div>';
        }

        // ===== FOOTER ACTIONS =====
        html += `<div class="modal-footer-actions">
            <button class="btn btn-primary" onclick="focusOnEvent(${eventId}); document.getElementById('event-modal').classList.add('hidden');">📍 Focus on Map</button>
            <button class="btn btn-secondary" onclick="selectEventOnMap(${eventId}); document.getElementById('event-modal').classList.add('hidden');">📐 Show Hyperbolas</button>
        </div>`;

        body.innerHTML = html;
        modal.classList.remove('hidden');
    }).catch(console.error);
}

// Node colors (must match map.js)
// NODE_COLORS is defined in map.js

// Close modal
document.getElementById('modal-close')?.addEventListener('click', () => {
    document.getElementById('event-modal').classList.add('hidden');
});
document.querySelector('.modal-backdrop')?.addEventListener('click', () => {
    document.getElementById('event-modal').classList.add('hidden');
});
