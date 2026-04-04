#!/usr/bin/env python3
"""
SAR-TDoA Realistic Simulation (with ML) — Unified Launcher
Starts all components in the correct order and manages lifecycle.
"""

import os
import sys
import time
import argparse
import webbrowser
import signal
import yaml
import threading

# Prevent OpenBLAS/MKL deadlocks by forcing single-threaded execution on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Force unbuffered output for Windows CMD/PowerShell
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def load_config(path: str = None) -> dict:
    """Load configuration from YAML file."""
    if path is None:
        path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="SAR-TDoA Realistic Simulation")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--test", type=str, default=None, help="Run a single test scenario (e.g., A1)")
    parser.add_argument("--verify", action="store_true", help="Run all verification tests")
    parser.add_argument("--monte-carlo", action="store_true", help="Run Monte Carlo batch")
    parser.add_argument("--iters", type=int, default=100, help="Monte Carlo iterations")
    parser.add_argument("--train", action="store_true", help="Train ML models before running")
    parser.add_argument("--train-backend", type=str, default=None, choices=["numpy", "sklearn"],
                        help="Override classifier training backend (numpy or sklearn)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    parser.add_argument("--no-speaker", action="store_true", help="Disable speaker playback")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.no_speaker:
        config.setdefault("audio", {}).setdefault("speaker", {})["enabled"] = False

    # Auto-detect trained ML model
    model_dir = os.path.join(PROJECT_ROOT, "models")
    sklearn_model_path = os.path.join(model_dir, "sound_classifier.joblib")
    esp32_model_path = os.path.join(model_dir, "sound_classifier_esp32.joblib")
    solver_model_path = os.path.join(model_dir, "position_regressor.joblib")
    ml_cfg = config.get("nodes", {}).get("ml_classifier", {})
    configured_backend = str(ml_cfg.get("model_type", "numpy_rescue")).strip().lower()
    train_backend = args.train_backend or ("sklearn" if configured_backend == "sklearn" else "numpy")

    if args.train:
        print("🧠 Training ML models...")
        from models.train_classifier import train_sound_classifier, train_sound_classifier_numpy
        from models.train_solver import train_position_regressor

        if train_backend == "sklearn":
            train_sound_classifier(output_path=sklearn_model_path)
        else:
            train_sound_classifier_numpy(output_path=esp32_model_path)

        train_position_regressor(config, output_path=solver_model_path)
        print("✅ ML models trained.\n")

    if configured_backend == "sklearn":
        preferred_model = sklearn_model_path
        fallback_model = esp32_model_path
    else:
        preferred_model = esp32_model_path
        fallback_model = sklearn_model_path

    if os.path.exists(preferred_model):
        config["_model_path"] = preferred_model
        print(f"🧠 [ML] Using trained classifier ({configured_backend}): {preferred_model}")
    elif os.path.exists(fallback_model):
        config["_model_path"] = fallback_model
        print(f"⚠️  [ML] Preferred model missing; using fallback: {fallback_model}")
    else:
        print("⚠️  [ML] No trained classifier found — using rule-based fallback.")
        print("   Run with --train to generate models first.\n")

    print("=" * 60)
    print("  SAR-TDoA Realistic Simulation (with ML)")
    print("=" * 60)
    print()

    # ===== STARTUP ORDER =====
    print("📦 Loading Simulation Engine Components...", flush=True)
    
    # Mathematical libraries are now deferred to methods to prevent startup hangs
    print("   > Importing SolverProcess...", flush=True)
    from solver.solver_process import SolverProcess
    
    print("   > Importing LoRaChannel...", flush=True)
    from network.lora_channel import LoRaChannel
    
    print("   > Importing NodeProcess...", flush=True)
    from node.node_process import NodeProcess
    
    print("   > Importing SourceEngine...", flush=True)
    from source.source_engine import SourceEngine
    
    print("✅ Components Loaded.\n", flush=True)
    
    components = []
    nodes = {}

    # 1. Solver process (needs to be ready before anything sends packets)
    print("⏳ Starting Solver Process...")
    solver = SolverProcess(config)
    solver.start()
    components.append(("Solver", solver.stop))
    time.sleep(0.3)

    # 2. LoRa channel
    print("⏳ Starting LoRa Channel (E22-900T22D)...")
    lora = LoRaChannel(config)
    lora.start()
    components.append(("LoRa", lora.stop))
    time.sleep(0.3)

    # 3. ESP32 Nodes (4 independent processes)
    print("⏳ Starting ESP32 Nodes (1-4)...")
    for nid in range(1, 5):
        node = NodeProcess(nid, config)
        node.start()
        nodes[nid] = node
        components.append((f"Node {nid}", node.stop))
    time.sleep(0.5)

    # 4. Source engine
    print("⏳ Starting Source Engine...")
    source = SourceEngine(config)
    source.start()
    components.append(("Source", source.stop))
    time.sleep(0.3)

    # 5. Flask web server
    print("⏳ Starting Web Server...")
    from web.app import create_app
    app, socketio = create_app(source, nodes, lora, solver, config)

    web_cfg = config.get("web", {})
    host = web_cfg.get("host", "127.0.0.1")
    port = web_cfg.get("port", 8080)

    # Run Flask in background thread
    def run_flask():
        if socketio:
            socketio.run(app, host=host, port=port,
                         allow_unsafe_werkzeug=True, use_reloader=False)
        else:
            app.run(host=host, port=port, debug=False, use_reloader=False)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    time.sleep(1.0)

    print()
    print("=" * 60)
    print(f"  ✅ All systems running!")
    print(f"  🌐 Web UI: http://{host}:{port}")
    print(f"  📡 Nodes: 4 active on ports 5011-5014")
    print(f"  📻 LoRa: port 5020 (reliability={config.get('lora', {}).get('base_reliability', 0.95):.0%})")
    print(f"  🧮 Solver: {config.get('solver', {}).get('method', 'auto_pipeline')}")
    print(f"  🔊 Speaker: {'ON' if config.get('audio', {}).get('speaker', {}).get('enabled', True) else 'OFF'}")
    print("=" * 60)
    print()
    print("Press Ctrl+C to stop all components.")
    print()

    # Open browser
    if not args.no_browser and web_cfg.get("auto_open_browser", True):
        webbrowser.open(f"http://{host}:{port}")

    # Handle CLI scenarios
    if args.test:
        time.sleep(2.0)
        try:
            from testing.scenarios import get_scenario
        except Exception as e:
            print(f"⚠️  Scenario support unavailable: {e}")
            print("   Continue running without scenario execution.")
            get_scenario = None

        if get_scenario is None:
            scenario = None
        else:
            scenario = get_scenario(args.test)

        if scenario is None:
            pass
        else:
            print(f"\n🧪 Running scenario: {scenario['name']}")
            source.fire_event(
                scenario["sound_type"], scenario["lat"], scenario["lon"],
                scenario["amplitude"], scenario_name=scenario["name"],
                **scenario.get("synth_kwargs", {})
            )

    elif args.verify:
        time.sleep(2.0)
        try:
            from testing.scenarios import ALL_SCENARIOS
        except Exception as e:
            print(f"⚠️  Scenario support unavailable: {e}")
            print("   Continue running without verification batch.")
            ALL_SCENARIOS = []

        if ALL_SCENARIOS:
            print(f"\n🧪 Running all {len(ALL_SCENARIOS)} verification scenarios...")
            for i, s in enumerate(ALL_SCENARIOS):
                print(f"\n--- Scenario {i+1}/{len(ALL_SCENARIOS)}: {s['name']} ---")
                source.fire_event(
                    s["sound_type"], s["lat"], s["lon"],
                    s["amplitude"], scenario_name=s["name"],
                    **s.get("synth_kwargs", {})
                )
                time.sleep(3.0)

    # Keep running until Ctrl+C
    def shutdown(sig, frame):
        print("\n\n🛑 Shutting down...")
        for name, stop_fn in reversed(components):
            print(f"   Stopping {name}...")
            try:
                stop_fn()
            except Exception:
                pass
        print("   Bye! 👋")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
