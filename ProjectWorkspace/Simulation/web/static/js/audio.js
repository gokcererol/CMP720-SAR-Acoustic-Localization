/**
 * Audio Visualizer — Waveform oscilloscope display for the audio monitor panel.
 * Shows source audio and per-node delayed/attenuated waveforms.
 */

let audioCtx;
const waveformData = new Float32Array(260);
const nodeWaveforms = { 1: new Float32Array(180), 2: new Float32Array(180), 3: new Float32Array(180), 4: new Float32Array(180) };

document.addEventListener('DOMContentLoaded', () => {
    const mainCanvas = document.getElementById('waveform-canvas');
    if (!mainCanvas) return;

    const ctx = mainCanvas.getContext('2d');
    const width = mainCanvas.width;
    const height = mainCanvas.height;

    // Animation loop for waveform
    function drawWaveform() {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.fillRect(0, 0, width, height);

        // Draw grid lines
        ctx.strokeStyle = 'rgba(71, 85, 105, 0.3)';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();

        // Draw waveform
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 1.5;
        ctx.shadowBlur = 4;
        ctx.shadowColor = '#6366f1';
        ctx.beginPath();

        for (let i = 0; i < width; i++) {
            const v = waveformData[i] || 0;
            const y = height / 2 + v * (height / 2 - 4);
            if (i === 0) ctx.moveTo(i, y);
            else ctx.lineTo(i, y);
        }
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Decay waveform
        for (let i = 0; i < waveformData.length; i++) {
            waveformData[i] *= 0.97;
        }

        requestAnimationFrame(drawWaveform);
    }

    drawWaveform();

    // Draw node waveforms
    const nodeCanvases = document.querySelectorAll('.node-canvas');
    nodeCanvases.forEach(canvas => {
        const nctx = canvas.getContext('2d');
        const nid = parseInt(canvas.dataset.node);
        const nw = canvas.width;
        const nh = canvas.height;
        const colors = { 1: '#6366f1', 2: '#22c55e', 3: '#eab308', 4: '#ef4444' };

        function drawNodeWave() {
            nctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
            nctx.fillRect(0, 0, nw, nh);

            const data = nodeWaveforms[nid];
            nctx.strokeStyle = colors[nid] || '#94a3b8';
            nctx.lineWidth = 1;
            nctx.beginPath();

            for (let i = 0; i < nw; i++) {
                const v = data[i] || 0;
                const y = nh / 2 + v * (nh / 2 - 2);
                if (i === 0) nctx.moveTo(i, y);
                else nctx.lineTo(i, y);
            }
            nctx.stroke();

            // Decay
            for (let i = 0; i < data.length; i++) {
                data[i] *= 0.95;
            }

            requestAnimationFrame(drawNodeWave);
        }

        drawNodeWave();
    });
});

// Simulate waveform injection (called when event fires)
function injectWaveform(soundType, amplitude) {
    const freq = soundType === 'whistle' ? 0.15 : soundType === 'impact' ? 0.5 : 0.1;
    const decay = soundType === 'impact' ? 0.92 : 0.99;
    let amp = amplitude || 0.8;

    for (let i = 0; i < waveformData.length; i++) {
        waveformData[i] += amp * Math.sin(i * freq * Math.PI * 2);
        amp *= decay;
    }

    // Also inject into node waveforms with delays
    const delays = { 1: 0, 2: 15, 3: 30, 4: 10 };
    const attens = { 1: 1.0, 2: 0.75, 3: 0.55, 4: 0.85 };

    for (let nid = 1; nid <= 4; nid++) {
        const data = nodeWaveforms[nid];
        let nodeAmp = (amplitude || 0.8) * attens[nid];
        const d = delays[nid];
        for (let i = d; i < data.length; i++) {
            data[i] += nodeAmp * Math.sin((i - d) * freq * Math.PI * 2);
            nodeAmp *= decay;
        }
    }

    // Update info display
    const info = document.getElementById('audio-now-playing');
    if (info) {
        info.textContent = `Playing: ${soundType} (amp=${amplitude?.toFixed(2) || '0.80'})`;
        setTimeout(() => { info.textContent = 'Idle'; }, 3000);
    }
}
