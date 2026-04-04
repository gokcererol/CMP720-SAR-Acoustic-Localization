"""
Solver Process — Main Jetson solver loop.
Receives assembled events from the collector, solves TDoA,
applies filters, and pushes results to the web UI.
"""
print("      - solver_process.py: top-level reached", flush=True)

import threading
import time
from typing import Dict, List, Optional, Any

print("      - solver_process.py: sub-imports starting...", flush=True)
from solver.collector import PacketCollector
from solver.tdoa_solver import TDoASolver, latlon_to_meters
from solver.gdop import compute_gdop, compute_confidence_ellipse, gdop_color, gdop_label
from solver.filters import FilterPipeline
print("      - solver_process.py: initialization complete.", flush=True)


class SolverProcess:
    """
    Main solver that assembles events, solves TDoA positions,
    computes GDOP, applies filters, and stores results.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.solver_method = config.get("solver", {}).get("method", "auto_pipeline")
        self.enable_all_method_compare = bool(
            config.get("solver", {}).get("compare_all_methods", False)
        )

        # Initialize components
        node_positions = config.get("nodes", {}).get("positions", {})
        speed = config.get("environment", {}).get("speed_of_sound_ms", 343.0)
        solver_cfg = config.get("solver", {})

        self.tdoa_solver = TDoASolver(
            node_positions,
            speed,
            max_range_m=solver_cfg.get("max_range_m"),
            max_residual_m=solver_cfg.get("max_residual_m", 20.0),
            max_physical_slack_m=solver_cfg.get("max_physical_slack_m", 40.0),
        )
        self.filter_pipeline = FilterPipeline(config)

        # Node positions in meters for GDOP
        self.node_positions_m = {}
        for nid, pos in node_positions.items():
            x, y = latlon_to_meters(
                pos["lat"], pos["lon"],
                self.tdoa_solver.ref_lat, self.tdoa_solver.ref_lon
            )
            self.node_positions_m[nid] = (x, y)

        # Collector
        self.collector = PacketCollector(
            listen_port=5000,
            packet_timeout_sec=solver_cfg.get("packet_timeout_sec", 1.5),
            min_packets=solver_cfg.get("min_packets", 3),
        )
        self.collector.set_event_callback(self._on_event)

        # Results storage
        self.event_results: List[Dict] = []
        self._lock = threading.Lock()
        self._web_callback = None
        self._active_ground_truth = None

    def set_ground_truth(self, lat: float, lon: float):
        """Set the physical source coordinate for error calculation."""
        with self._lock:
            self._active_ground_truth = {"lat": lat, "lon": lon}

    def clear_ground_truth(self):
        with self._lock:
            self._active_ground_truth = None

    def set_web_callback(self, callback):
        """Set callback for pushing results to web UI."""
        self._web_callback = callback

    def _on_event(self, event: Dict):
        """Called by collector when an event is assembled."""
        with self._lock:
            if self._active_ground_truth:
                event["ground_truth"] = self._active_ground_truth
                
        packets = event["packets"]
        event_id = event["event_id"]

        print(f"\n🧮 [SOLVER] Processing Event #{event_id}...")

        # Solve TDoA
        solver_result = self.tdoa_solver.solve(packets, self.solver_method)

        # Optionally run all methods for comparison (can be expensive on some setups)
        all_methods = {}
        if self.enable_all_method_compare and len(packets) >= 3:
            try:
                all_methods = self.tdoa_solver.solve_all_methods(packets)
            except Exception as e:
                print(f"   ⚠️ all-method comparison skipped: {e}")

        # Compute GDOP and confidence ellipse
        gdop_val = 99.0
        ellipse = {}
        if solver_result.get("success", False):
            gdop_val = compute_gdop(
                self.node_positions_m,
                solver_result["x"], solver_result["y"]
            )
            ellipse = compute_confidence_ellipse(
                self.node_positions_m,
                solver_result["x"], solver_result["y"]
            )

        # Apply filters
        result = self.filter_pipeline.apply(event, solver_result, gdop_val, packets)

        # Build complete result
        result.update({
            "solver_result": solver_result,
            "all_methods": all_methods,
            "gdop": gdop_val,
            "gdop_color": gdop_color(gdop_val),
            "gdop_label": gdop_label(gdop_val),
            "confidence_ellipse": ellipse,
        })

        # Compute position error if ground truth is available
        if "ground_truth" in event:
            gt = event["ground_truth"]
            if solver_result.get("success", False):
                import math
                error_m = math.sqrt(
                    ((solver_result["lat"] - gt["lat"]) * 111132.0) ** 2 +
                    ((solver_result["lon"] - gt["lon"]) * 111320.0 *
                     math.cos(math.radians(gt["lat"]))) ** 2
                )
                result["position_error_m"] = error_m

        # Store result
        with self._lock:
            self.event_results.append(result)

        # Print summary
        status = result.get("filter_status", "UNKNOWN")
        if solver_result.get("success", False):
            lat = solver_result.get("lat", 0)
            lon = solver_result.get("lon", 0)
            method = solver_result.get("method", "")
            err_str = ""
            if "position_error_m" in result:
                err_str = f" | Error: {result['position_error_m']:.1f}m"
            print(f"   📍 [{status}] ({lat:.6f}, {lon:.6f}) | "
                  f"GDOP: {gdop_val:.1f} ({gdop_label(gdop_val)}) | "
                  f"Method: {method}{err_str}")
        else:
            reason = solver_result.get("reason", "unknown")
            print(f"   ❌ [{status}] Solver failed: {reason}")

        if result.get("filter_warnings"):
            for w in result["filter_warnings"]:
                print(f"   ⚠️ {w}")
        if result.get("filter_reasons"):
            for r in result["filter_reasons"]:
                print(f"   🚫 {r}")

        # Push to web UI
        if self._web_callback:
            try:
                self._web_callback(result)
            except Exception as e:
                print(f"[Solver] Web callback error: {e}")

    def get_results(self) -> List[Dict]:
        """Get all event results."""
        with self._lock:
            return list(self.event_results)

    def get_latest_result(self) -> Optional[Dict]:
        """Get the most recent result."""
        with self._lock:
            return self.event_results[-1] if self.event_results else None

    def set_method(self, method: str):
        """Set the solver method."""
        self.solver_method = method

    def update_config(self, update: Dict):
        """Update solver configuration at runtime."""
        if "method" in update:
            self.solver_method = update["method"]
        if "rmse_threshold_m" in update:
            self.filter_pipeline.update_config(update)
        if "packet_timeout_sec" in update:
            self.collector.update_config(update)

    def start(self):
        """Start the solver (starts collector)."""
        self.collector.start()
        print(f"✅ [SOLVER] Started (method={self.solver_method})")

    def stop(self):
        """Stop the solver."""
        self.collector.stop()
        print(f"🛑 [SOLVER] Stopped.")
