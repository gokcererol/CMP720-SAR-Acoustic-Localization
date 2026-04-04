"""
Rejection Filters — Pipeline of filters for accepting/rejecting TDoA solutions.
Each filter checks a different quality criterion.
"""
print("      - filters.py: top-level reached", flush=True)

from typing import Dict, List, Optional, Tuple
import math


class FilterPipeline:
    """
    Runs a sequence of quality filters on solved TDoA events.
    Returns the event with a status: CONFIRMED, WEAK_DATA, OUT_OF_BOUNDS, or REJECTED.
    """

    def __init__(self, config: Dict):
        solver_cfg = config.get("solver", {})
        self.rmse_threshold = solver_cfg.get("rmse_threshold_m", 8.5)
        self.max_range_m = solver_cfg.get("max_range_m", 2000)
        self.min_packets = solver_cfg.get("min_packets", 3)
        self.bounds_padding = solver_cfg.get("bounds_padding_m", 15.0)
        self.gdop_warning = solver_cfg.get("gdop_warning_threshold", 5.0)
        self.gdop_reject = solver_cfg.get("gdop_reject_threshold", 10.0)

        dscore_cfg = solver_cfg.get("decision_score", {})
        self.decision_score_enabled = dscore_cfg.get("enabled", True)
        self.decision_score_accept = dscore_cfg.get("accept_threshold", 0.58)
        self.decision_score_weights = {
            "ml": float(dscore_cfg.get("weights", {}).get("ml", 0.45)),
            "residual": float(dscore_cfg.get("weights", {}).get("residual", 0.25)),
            "gdop": float(dscore_cfg.get("weights", {}).get("gdop", 0.20)),
            "packets": float(dscore_cfg.get("weights", {}).get("packets", 0.10)),
        }

        ag_cfg = solver_cfg.get("anti_ghost", {})
        self.anti_ghost_enabled = ag_cfg.get("enabled", True)
        self.anti_ghost_dist = ag_cfg.get("distance_threshold_m", 150)
        self.anti_ghost_min_mag = ag_cfg.get("min_magnitude", 45)

        # Node positions for bounds checking
        node_cfg = config.get("nodes", {}).get("positions", {})
        self.node_lats = [p["lat"] for p in node_cfg.values()]
        self.node_lons = [p["lon"] for p in node_cfg.values()]

    def apply(self, event: Dict, solver_result: Dict, gdop: float,
              packets: Dict[int, Dict]) -> Dict:
        """
        Apply all filters to a solved event.
        Returns event with status and filter results.
        """
        reasons = []
        warnings = []
        status = "CONFIRMED"

        # Filter 1: Minimum packets
        num_packets = len(packets)
        if num_packets < self.min_packets:
            status = "REJECTED"
            reasons.append(f"insufficient_packets ({num_packets}/{self.min_packets})")
            return self._build_result(event, status, reasons, warnings)

        if not solver_result.get("success", False):
            status = "REJECTED"
            reasons.append(f"solver_failed: {solver_result.get('reason', 'unknown')}")
            return self._build_result(event, status, reasons, warnings)

        est_lat = solver_result.get("lat", 0)
        est_lon = solver_result.get("lon", 0)
        est_x = solver_result.get("x", 0)
        est_y = solver_result.get("y", 0)
        dist = None

        # Filter 2: RMSE threshold
        residual = solver_result.get("residual", 0)
        # Solver residual is already expressed in meters.
        rmse = float(residual)
        if rmse > self.rmse_threshold:
            status = "REJECTED"
            reasons.append(f"rmse={rmse:.1f}m > {self.rmse_threshold}m")

        # Filter 3: Range bounds
        if self.node_lats and est_lat != 0:
            center_lat = sum(self.node_lats) / len(self.node_lats)
            center_lon = sum(self.node_lons) / len(self.node_lons)
            dist = math.sqrt(
                ((est_lat - center_lat) * 111132.0) ** 2 +
                ((est_lon - center_lon) * 111320.0 * math.cos(math.radians(center_lat))) ** 2
            )
            if dist > self.max_range_m:
                status = "OUT_OF_BOUNDS"
                reasons.append(f"distance={dist:.0f}m > {self.max_range_m}m")

        # Filter 4: ML consensus check
        ml_classes = {nid: p.get("ml_class", -1) for nid, p in packets.items()}
        unique_classes = set(ml_classes.values())
        if len(unique_classes) > 1 and len(unique_classes) > len(ml_classes) / 2:
            warnings.append(f"ml_disagreement: {dict(ml_classes)}")

        # Filter 5: GDOP quality
        if gdop >= self.gdop_warning:
            if gdop >= self.gdop_reject:
                status = "REJECTED" if status not in ["OUT_OF_BOUNDS"] else status
                reasons.append(f"extreme_gdop={gdop:.1f}")
            else:
                warnings.append(f"high_gdop={gdop:.1f}")

        # Filter 6: Anti-ghost
        if self.anti_ghost_enabled:
            avg_mag = sum(p.get("magnitude", 0) for p in packets.values()) / len(packets)
            if avg_mag < self.anti_ghost_min_mag:
                # Weak signal — might be a ghost
                if dist is not None and dist > self.anti_ghost_dist:
                    status = "REJECTED"
                    reasons.append(f"anti_ghost: weak_mag={avg_mag:.0f}")

        decision_score = self._compute_decision_score(packets, rmse, gdop, num_packets)
        decision_recommendation = "accept" if decision_score >= self.decision_score_accept else "review"
        if self.decision_score_enabled and status == "CONFIRMED" and decision_recommendation == "review":
            status = "WEAK_DATA"
            warnings.append(f"decision_score_low={decision_score:.2f}")

        # Filter 7: Battery cross-check
        low_battery_nodes = [
            nid for nid, p in packets.items()
            if p.get("battery_pct", 100) < 10
        ]
        if len(low_battery_nodes) > 0:
            warnings.append(f"low_battery_nodes: {low_battery_nodes}")

        # Downgrade to WEAK_DATA if partially successful
        if status == "CONFIRMED" and num_packets < 4:
            status = "WEAK_DATA"
            warnings.append(f"only_{num_packets}/4_packets")

        if status == "CONFIRMED" and len(warnings) > 2:
            status = "WEAK_DATA"

        return self._build_result(event, status, reasons, warnings, decision_score, decision_recommendation)

    def _build_result(self, event: Dict, status: str,
                      reasons: List[str], warnings: List[str],
                      decision_score: Optional[float] = None,
                      decision_recommendation: Optional[str] = None) -> Dict:
        """Build the filter result dict."""
        event["filter_status"] = status
        event["filter_reasons"] = reasons
        event["filter_warnings"] = warnings
        if decision_score is not None:
            event["decision_score"] = float(decision_score)
        if decision_recommendation is not None:
            event["decision_recommendation"] = decision_recommendation
        return event

    def _compute_decision_score(self, packets: Dict[int, Dict], rmse: float, gdop: float, num_packets: int) -> float:
        """Compute a calibrated [0,1] score from ML confidence and solver quality indicators."""
        ml_conf_values = [float(p.get("ml_confidence", 0.0)) / 100.0 for p in packets.values()]
        ml_score = (sum(ml_conf_values) / len(ml_conf_values)) if ml_conf_values else 0.0

        residual_score = max(0.0, min(1.0, 1.0 - (rmse / max(self.rmse_threshold, 1e-6))))
        gdop_score = max(0.0, min(1.0, 1.0 - (gdop / max(self.gdop_reject, 1e-6))))
        packet_score = max(0.0, min(1.0, (num_packets - self.min_packets + 1) / 2.0))

        w = self.decision_score_weights
        total = w["ml"] + w["residual"] + w["gdop"] + w["packets"]
        if total <= 0.0:
            return ml_score
        return (
            w["ml"] * ml_score
            + w["residual"] * residual_score
            + w["gdop"] * gdop_score
            + w["packets"] * packet_score
        ) / total

    def update_config(self, update: Dict):
        """Update filter parameters at runtime."""
        if "rmse_threshold_m" in update:
            self.rmse_threshold = update["rmse_threshold_m"]
        if "max_range_m" in update:
            self.max_range_m = update["max_range_m"]
        if "gdop_warning_threshold" in update:
            self.gdop_warning = update["gdop_warning_threshold"]
