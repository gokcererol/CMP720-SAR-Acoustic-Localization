/**
 * Controls — Parameter sliders and buttons for real-time configuration.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ===== SLIDER BINDINGS =====
    const sliders = [
        { id: 'slider-rmse', valId: 'val-rmse', key: 'rmse_threshold_m', fmt: v => v },
        { id: 'slider-reliability', valId: 'val-reliability', key: 'base_reliability', fmt: v => (v / 100).toFixed(2), display: v => v },
        { id: 'slider-ml-conf', valId: 'val-ml-conf', key: 'confidence_threshold', fmt: v => parseFloat(v).toFixed(2) },
        { id: 'slider-sta-lta', valId: 'val-sta-lta', key: 'sta_lta_threshold', fmt: v => parseFloat(v).toFixed(1) },
        { id: 'slider-noise', valId: 'val-noise', key: 'ambient_noise_level', fmt: v => v },
        { id: 'slider-sos', valId: 'val-sos', key: 'speed_of_sound_ms', fmt: v => v },
    ];

    sliders.forEach(s => {
        const slider = document.getElementById(s.id);
        const valEl = document.getElementById(s.valId);
        if (!slider || !valEl) return;

        slider.addEventListener('input', () => {
            const displayVal = s.display ? s.display(slider.value) : s.fmt(slider.value);
            valEl.textContent = displayVal;
        });

        slider.addEventListener('change', () => {
            const val = s.key === 'base_reliability' ? parseFloat(slider.value) / 100 : parseFloat(slider.value);
            const update = {};
            if (s.key === 'base_reliability') {
                update.lora = { base_reliability: val };
            } else if (s.key === 'rmse_threshold_m') {
                update.solver = { rmse_threshold_m: val };
            } else if (s.key === 'confidence_threshold') {
                update.ml_classifier = { confidence_threshold: val };
            } else {
                update[s.key] = val;
            }

            fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(update),
            }).catch(console.error);
        });
    });

    // ===== SOLVER METHOD =====
    document.querySelectorAll('input[name="solver"]').forEach(radio => {
        radio.addEventListener('change', () => {
            fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ solver: { method: radio.value } }),
            }).catch(console.error);
        });
    });

    // ===== MAP LAYER TOGGLES =====
    document.getElementById('layer-nodes')?.addEventListener('change', e => toggleNodeLayer(e.target.checked));
    document.getElementById('layer-events')?.addEventListener('change', e => toggleEventLayer(e.target.checked));
    document.getElementById('layer-confidence')?.addEventListener('change', e => toggleEllipseLayer(e.target.checked));

    // ===== MANUAL FIRE =====
    document.getElementById('btn-fire-manual')?.addEventListener('click', () => {
        const data = {
            sound_type: document.getElementById('manual-sound').value,
            lat: parseFloat(document.getElementById('manual-lat').value),
            lon: parseFloat(document.getElementById('manual-lon').value),
            amplitude: parseFloat(document.getElementById('manual-amp').value),
        };

        fetch('/api/fire', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        }).then(r => r.json())
          .then(d => console.log('🔥 Event fired:', d))
          .catch(console.error);
    });

    // Manual amplitude slider display
    const manualAmp = document.getElementById('manual-amp');
    const valManualAmp = document.getElementById('val-manual-amp');
    if (manualAmp && valManualAmp) {
        manualAmp.addEventListener('input', () => {
            valManualAmp.textContent = parseFloat(manualAmp.value).toFixed(2);
        });
    }

    // ===== SPEAKER CONTROLS =====
    let speakerMuted = false;
    const btnMute = document.getElementById('btn-mute');
    btnMute?.addEventListener('click', () => {
        speakerMuted = !speakerMuted;
        btnMute.textContent = speakerMuted ? '🔇' : '🔊';
        fetch('/api/speaker', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: !speakerMuted }),
        }).catch(console.error);
    });

    const volumeSlider = document.getElementById('slider-volume');
    const valVolume = document.getElementById('val-volume');
    volumeSlider?.addEventListener('input', () => {
        valVolume.textContent = volumeSlider.value + '%';
    });
    volumeSlider?.addEventListener('change', () => {
        fetch('/api/speaker', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ volume: parseFloat(volumeSlider.value) / 100 }),
        }).catch(console.error);
    });
});
