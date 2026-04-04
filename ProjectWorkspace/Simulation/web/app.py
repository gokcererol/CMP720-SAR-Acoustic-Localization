"""
Flask Web Application — Main web server providing the control panel,
API endpoints, and WebSocket for real-time updates.
"""

import os
import json
import time
import threading
from typing import Dict, Optional

from flask import Flask, render_template, jsonify, request

SCENARIOS_AVAILABLE = True
SCENARIO_IMPORT_ERROR = ""
try:
    from testing.scenarios import ALL_SCENARIOS, get_scenario, get_scenario_list, CATEGORIES
except Exception as e:
    SCENARIOS_AVAILABLE = False
    SCENARIO_IMPORT_ERROR = str(e)
    ALL_SCENARIOS = []
    CATEGORIES = {}

    def get_scenario(name: str):
        raise ValueError("Scenario support unavailable: testing.scenarios module not found")

    def get_scenario_list():
        return []


def create_app(source_engine=None, nodes=None, lora_channel=None,
               solver_process=None, config=None):
    """Create and configure the Flask application."""

    app = Flask(__name__,
                template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                static_folder=os.path.join(os.path.dirname(__file__), 'static'))
    app.config['SECRET_KEY'] = 'sar-tdoa-sim'

    web_cfg = (config or {}).get("web", {})
    socketio_enabled = bool(web_cfg.get("socketio_enabled", False))
    socketio = None
    if socketio_enabled:
        try:
            from flask_socketio import SocketIO
            socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
            print("🌐 [WEB] Socket.IO enabled.")
        except Exception as e:
            print(f"⚠️ [WEB] Socket.IO disabled due to import/init error: {e}")
            socketio_enabled = False

    app.socketio_enabled = bool(socketio_enabled and socketio is not None)

    if not SCENARIOS_AVAILABLE:
        print(
            f"⚠️ [WEB] Scenario APIs disabled: testing.scenarios is unavailable ({SCENARIO_IMPORT_ERROR})"
        )

    # Store component references
    app.source_engine = source_engine
    app.nodes = nodes or {}
    app.lora_channel = lora_channel
    app.solver_process = solver_process
    app.sim_config = config or {}

    if socketio and app.source_engine:
        def _send_audio_ws(data):
            socketio.emit('audio_event', data)
        app.source_engine.set_web_callback(_send_audio_ws)

    # ===== PAGE ROUTES =====
    @app.route('/')
    def index():
        return render_template('index.html', socketio_enabled=app.socketio_enabled)

    # ===== API: STATUS =====
    @app.route('/api/status')
    def api_status():
        """Get full system status."""
        status = {
            "timestamp": time.time(),
            "nodes": {},
            "lora": {},
            "solver": {},
            "source": {},
        }
        # Node status
        if app.nodes:
            for nid, node in app.nodes.items():
                status["nodes"][nid] = node.get_status()
        # LoRa status
        if app.lora_channel:
            status["lora"] = app.lora_channel.get_status()
        # Solver status
        if app.solver_process:
            collector_status = app.solver_process.collector.get_status()
            status["solver"]["collector"] = collector_status
            status["solver"]["method"] = app.solver_process.solver_method
            status["solver"]["total_events"] = len(app.solver_process.event_results)
        return jsonify(status)

    # ===== API: EVENTS =====
    @app.route('/api/events')
    def api_events():
        """Get all solved events."""
        if app.solver_process:
            results = app.solver_process.get_results()
            # Serialize for JSON (remove numpy arrays)
            clean = []
            for r in results:
                event = _clean_event(r)
                clean.append(event)
            return jsonify(clean)
        return jsonify([])

    @app.route('/api/events/latest')
    def api_latest_event():
        """Get latest event result."""
        if app.solver_process:
            result = app.solver_process.get_latest_result()
            if result:
                return jsonify(_clean_event(result))
        return jsonify(None)

    # ===== API: FIRE EVENT =====
    @app.route('/api/fire', methods=['POST'])
    def api_fire_event():
        """Fire a sound event at specified coordinates."""
        data = request.json
        sound_type = data.get("sound_type", "whistle")
        lat = data.get("lat", 39.867449)
        lon = data.get("lon", 32.733585)
        amplitude = data.get("amplitude", 0.8)
        duration = data.get("duration", None)

        if app.source_engine:
            if hasattr(app, "solver_process") and app.solver_process:
                app.solver_process.set_ground_truth(lat, lon)
            info = app.source_engine.fire_event(
                sound_type, lat, lon, amplitude, duration,
                scenario_name=data.get("scenario", "manual")
            )
            return jsonify({"success": True, "info": _clean_dict(info)})
        return jsonify({"success": False, "error": "Source engine not running"})

    # ===== API: SCENARIOS =====
    @app.route('/api/scenarios')
    def api_scenarios():
        """Get list of all test scenarios."""
        return jsonify(get_scenario_list())

    @app.route('/api/scenarios/fire', methods=['POST'])
    def api_fire_scenario():
        """Fire a specific test scenario."""
        if not SCENARIOS_AVAILABLE:
            return jsonify({
                "success": False,
                "error": "Scenario support unavailable (missing testing.scenarios module)",
            }), 503

        data = request.json
        scenario_name = data.get("name", "A1")
        try:
            scenario = get_scenario(scenario_name)
        except ValueError:
            return jsonify({"success": False, "error": f"Unknown scenario: {scenario_name}"})

        if app.source_engine:
            try:
                print(f"🔥 [WEB] Scenario request: {scenario['name']}")
                if hasattr(app, "solver_process") and app.solver_process:
                    app.solver_process.set_ground_truth(scenario["lat"], scenario["lon"])
                info = app.source_engine.fire_event(
                    scenario["sound_type"],
                    scenario["lat"], scenario["lon"],
                    scenario["amplitude"],
                    scenario_name=scenario["name"],
                    **scenario.get("synth_kwargs", {})
                )
                return jsonify({
                    "success": True,
                    "scenario": scenario["name"],
                    "expected_status": scenario["expected_status"],
                    "info": _clean_dict(info),
                })
            except Exception as e:
                print(f"❌ [WEB] Scenario fire failed: {e}")
                return jsonify({"success": False, "error": str(e)}), 500
        return jsonify({"success": False, "error": "Source engine not running"})

    @app.route('/api/scenarios/run_batch', methods=['POST'])
    def api_run_batch():
        """Run a batch of scenarios."""
        if not SCENARIOS_AVAILABLE:
            return jsonify({
                "success": False,
                "error": "Scenario support unavailable (missing testing.scenarios module)",
            }), 503

        data = request.json
        category = data.get("category", "all")
        delay = data.get("delay_sec", 3.0)

        if category == "all":
            scenarios = ALL_SCENARIOS
        elif category in CATEGORIES:
            scenarios = CATEGORIES[category]
        else:
            return jsonify({"success": False, "error": f"Unknown category: {category}"})

        # Run in background thread
        def _run():
            for i, s in enumerate(scenarios):
                if app.source_engine:
                    if hasattr(app, "solver_process") and app.solver_process:
                        app.solver_process.set_ground_truth(s["lat"], s["lon"])
                    app.source_engine.fire_event(
                        s["sound_type"], s["lat"], s["lon"],
                        s["amplitude"], scenario_name=s["name"],
                        **s.get("synth_kwargs", {})
                    )
                time.sleep(delay)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return jsonify({"success": True, "count": len(scenarios)})

    # ===== API: CONFIG UPDATE =====
    @app.route('/api/config', methods=['GET'])
    def api_get_config():
        """Get current configuration."""
        return jsonify(app.sim_config)

    @app.route('/api/config', methods=['POST'])
    def api_update_config():
        """Update configuration parameters in real-time."""
        data = request.json

        # Update source engine
        if app.source_engine:
            app.source_engine._apply_config_update(data)

        # Update LoRa channel
        if app.lora_channel and "lora" in data:
            app.lora_channel.update_config(data["lora"])

        # Update solver
        if app.solver_process and "solver" in data:
            app.solver_process.update_config(data["solver"])

        # Update nodes
        if app.nodes:
            for nid, node in app.nodes.items():
                node._apply_config_update(data)

        return jsonify({"success": True})

    # ===== API: SPEAKER =====
    @app.route('/api/speaker', methods=['POST'])
    def api_speaker():
        """Control speaker settings."""
        data = request.json
        if app.source_engine:
            if "enabled" in data:
                app.source_engine.speaker.set_enabled(data["enabled"])
            if "volume" in data:
                app.source_engine.speaker.set_volume(data["volume"])
            return jsonify({"success": True})
        return jsonify({"success": False})

    # ===== API: NODE POSITIONS =====
    @app.route('/api/nodes/positions')
    def api_node_positions():
        """Get node positions for map display."""
        positions = app.sim_config.get("nodes", {}).get("positions", {})
        return jsonify(positions)

    # ===== HELPER FUNCTIONS =====
    def _clean_event(event: Dict) -> Dict:
        """Remove non-serializable fields from an event."""
        clean = {}
        for key, val in event.items():
            if key == "packets":
                clean[key] = {str(k): _clean_dict(v) for k, v in val.items()}
            elif key in ("all_methods",):
                clean[key] = {k: _clean_dict(v) for k, v in val.items()}
            elif isinstance(val, dict):
                clean[key] = _clean_dict(val)
            elif isinstance(val, (int, float, str, bool, type(None))):
                clean[key] = val
            elif isinstance(val, list):
                clean[key] = [_clean_dict(v) if isinstance(v, dict) else v for v in val]
        return clean

    def _clean_dict(d) -> Dict:
        """Recursively clean a dict for JSON serialization."""
        if not isinstance(d, dict):
            return d
        clean = {}
        for k, v in d.items():
            if isinstance(v, dict):
                clean[k] = _clean_dict(v)
            elif isinstance(v, (int, float, str, bool, type(None))):
                clean[k] = v
            elif isinstance(v, list):
                clean[k] = v
        return clean

    return app, socketio
