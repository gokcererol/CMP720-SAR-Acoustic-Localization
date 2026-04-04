print("      - tdoa_solver.py: top-level reached", flush=True)
import numpy as np
import math
import subprocess
import sys
from typing import Dict, List, Tuple, Optional
# scipy.optimize is deferred to prevent startup hangs


def latlon_to_meters(lat: float, lon: float, ref_lat: float = 0, ref_lon: float = 0) -> Tuple[float, float]:
    """Convert lat/lon to local meter coordinates relative to reference point."""
    x = (lon - ref_lon) * 111320.0 * math.cos(math.radians(ref_lat))
    y = (lat - ref_lat) * 111132.0
    return x, y


def meters_to_latlon(x: float, y: float, ref_lat: float = 0, ref_lon: float = 0) -> Tuple[float, float]:
    """Convert local meter coordinates back to lat/lon."""
    lat = ref_lat + y / 111132.0
    lon = ref_lon + x / (111320.0 * math.cos(math.radians(ref_lat)))
    return lat, lon


class TDoASolver:
    """
    Solves TDoA localization using three methods.
    Nodes must have known positions. At least 3 packets required.
    """

    def __init__(self, node_positions: Dict[int, Dict],
                 speed_of_sound: float = 343.0,
                 max_range_m: Optional[float] = None,
                 max_residual_m: float = 20.0,
                 max_physical_slack_m: float = 40.0):
        self.speed_of_sound = speed_of_sound

        # Convert node positions to meters relative to centroid
        lats = [p["lat"] for p in node_positions.values()]
        lons = [p["lon"] for p in node_positions.values()]
        self.ref_lat = sum(lats) / len(lats)
        self.ref_lon = sum(lons) / len(lons)

        self.node_positions = {}  # node_id -> (x_m, y_m)
        for nid, pos in node_positions.items():
            x, y = latlon_to_meters(pos["lat"], pos["lon"], self.ref_lat, self.ref_lon)
            self.node_positions[nid] = (x, y)

        self.node_pair_dist_m = {}
        node_ids = list(self.node_positions.keys())
        for i in range(len(node_ids)):
            n1 = node_ids[i]
            x1, y1 = self.node_positions[n1]
            for j in range(i + 1, len(node_ids)):
                n2 = node_ids[j]
                x2, y2 = self.node_positions[n2]
                self.node_pair_dist_m[(n1, n2)] = math.hypot(x2 - x1, y2 - y1)
            
        node_xy = list(self.node_positions.values())
        max_pair = 0.0
        for i in range(len(node_xy)):
            xi, yi = node_xy[i]
            for j in range(i + 1, len(node_xy)):
                xj, yj = node_xy[j]
                max_pair = max(max_pair, math.hypot(xi - xj, yi - yj))
        self.array_aperture_m = max_pair

        default_range_cap = max(180.0, 3.6 * self.array_aperture_m)
        self.max_range_m = float(max_range_m) if max_range_m is not None else float(min(2500.0, default_range_cap))
        self.max_residual_m = float(max(2.0, max_residual_m))
        self.max_physical_slack_m = float(max(1.0, max_physical_slack_m))
        self.max_geometry_condition = 35.0
        self.near_field_rescue_radius_m = float(max(120.0, 1.8 * self.array_aperture_m))
        self.scipy_enabled = self._probe_scipy_import(timeout_sec=8)

    def _probe_scipy_import(self, timeout_sec: int = 8) -> bool:
        """Probe SciPy import in a child process so main solver cannot deadlock."""
        cmd = [sys.executable, "-c", "from scipy.optimize import least_squares, minimize; print('ok')"]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=max(1, int(timeout_sec)),
            )
            if proc.returncode == 0:
                print("      - tdoa_solver.py: scipy probe OK", flush=True)
                return True

            print("      - tdoa_solver.py: scipy probe failed; disabling scipy solvers", flush=True)
            return False
        except subprocess.TimeoutExpired:
            print("      - tdoa_solver.py: scipy probe timeout; disabling scipy solvers", flush=True)
            return False
        except Exception:
            print("      - tdoa_solver.py: scipy probe error; disabling scipy solvers", flush=True)
            return False

    def solve(self, packets: Dict[int, Dict], method: str = "auto_pipeline") -> Dict:
        """
        Solve TDoA for a set of packets from different nodes.
        Returns dict with position estimate, error metrics, and method used.
        """
        if len(packets) < 3:
            return {"success": False, "reason": "insufficient_packets",
                    "num_packets": len(packets)}

        if not self._passes_physical_tdoa_consistency(packets):
            return {"success": False, "reason": "inconsistent_tdoa"}

        # Use a single reference node (reference-vs-others TDoA).
        all_ids = sorted(packets.keys())
        candidate_refs = [self._choose_reference_node(packets, all_ids)]
        selected_ref_node = int(candidate_refs[0]) if candidate_refs else None

        best_result = None
        best_positions = None
        best_tdoa = None
        best_weights = None
        best_score = float("inf")
        best_failure = None

        for ref_node in candidate_refs:
            node_ids = [ref_node] + [nid for nid in all_ids if nid != ref_node]
            positions, tdoa, weights = self._prepare_arrays(packets, node_ids)

            if method == "auto_pipeline":
                result = self._pipeline(positions, tdoa, weights, node_ids)
            
                # --- RANSAC Outlier Rejection Fallback ---
                # Residual is in meters; try subsets when fit quality is poor.
                if len(positions) > 3 and result.get("success", False) and result.get("residual", 0) > 8.0:
                    import itertools
                    best_sub_result = result
                
                    # Check all (N-1) subsets
                    for combo in itertools.combinations(range(len(positions)), len(positions) - 1):
                        idx = list(combo)
                        sub_pos = positions[idx]
                        # Recenter TDoA relative to the first node in the subset
                        sub_tdoa = tdoa[idx] - tdoa[idx[0]]
                        sub_w = weights[idx]
                        sub_w = sub_w / np.sum(sub_w)
                        sub_nids = [node_ids[i] for i in idx]

                        sub_res = self._pipeline(sub_pos, sub_tdoa, sub_w, sub_nids)
                        if sub_res.get("success", False):
                            if sub_res.get("residual", float('inf')) < best_sub_result.get("residual", float('inf')):
                                best_sub_result = sub_res

                    # If outlier rejection improved the residual significantly, accept it.
                    if best_sub_result != result and best_sub_result.get("residual", float('inf')) < result.get("residual", float('inf')) - 1.0:
                        best_sub_result["method"] += " (ransac)"
                        result = best_sub_result

            elif method == "chan_ho":
                result = self._chan_ho(positions, tdoa, weights, node_ids)
            elif method == "lm":
                result = self._levenberg_marquardt(positions, tdoa, weights, node_ids)
            elif method == "nelder_mead":
                result = self._nelder_mead(positions, tdoa, weights, node_ids)
            else:
                result = self._pipeline(positions, tdoa, weights, node_ids)

            if result.get("success", False):
                residual = float(result.get("residual", float("inf")))
                dist = float(np.hypot(result.get("x", 0.0), result.get("y", 0.0)))
                cond = self._jacobian_condition(positions, tdoa, result.get("x", 0.0), result.get("y", 0.0))

                # Candidate score: prioritize low residual but penalize unstable geometry
                # and implausible far-branch picks that commonly appear in corner scenarios.
                score = residual
                if np.isfinite(cond):
                    score += 0.35 * max(0.0, cond - 12.0)
                else:
                    score += 20.0

                if dist > self.near_field_rescue_radius_m:
                    score += 0.015 * (dist - self.near_field_rescue_radius_m)
                if dist > self.max_range_m:
                    score += 100.0 + 0.05 * (dist - self.max_range_m)

                if (
                    best_result is None
                    or score < best_score
                    or (
                        abs(score - best_score) < 0.2
                        and residual < float(best_result.get("residual", float("inf")))
                    )
                ):
                    best_result = result
                    best_positions = positions
                    best_tdoa = tdoa
                    best_weights = weights
                    best_score = score
            else:
                best_failure = result

        result = best_result if best_result is not None else (best_failure or {"success": False, "reason": "no_solution"})

        # Convert result back to lat/lon
        if result.get("success", False):
            def _with_latlon(sol: Dict) -> Dict:
                lat, lon = meters_to_latlon(sol["x"], sol["y"], self.ref_lat, self.ref_lon)
                sol["lat"] = lat
                sol["lon"] = lon
                return sol

            # Residual gate
            if result.get("residual", float("inf")) > self.max_residual_m:
                return {
                    "success": False,
                    "reason": "high_residual",
                    "residual": float(result.get("residual", float("inf"))),
                    "residual_limit": float(self.max_residual_m),
                    "ref_node": selected_ref_node,
                }

            geom_cond = float("inf")
            if best_positions is not None and best_tdoa is not None:
                geom_cond = self._jacobian_condition(best_positions, best_tdoa, result["x"], result["y"])

            dist_from_ref = float(np.sqrt(result["x"]**2 + result["y"]**2))
            residual_now = float(result.get("residual", float("inf")))
            far_branch_suspect = dist_from_ref > max(130.0, 1.25 * self.array_aperture_m)

            # Rescue branch: corner scenarios can drift to ambiguous/far branches.
            # We evaluate a bounded near-field candidate more often and accept it
            # only when it is similarly consistent but clearly more plausible.
            rescue_candidate_ok = False
            rescue_candidate = None
            consider_rescue = (
                best_positions is not None
                and best_tdoa is not None
                and (
                    (np.isfinite(geom_cond) and geom_cond > self.max_geometry_condition)
                    or dist_from_ref > self.max_range_m
                    or (far_branch_suspect and residual_now > 4.0)
                    or residual_now > 10.0
                )
            )
            if consider_rescue:
                hint = np.array([result["x"], result["y"]], dtype=np.float64)
                rw = best_weights if best_weights is not None else np.ones(len(best_positions), dtype=np.float64)
                rescue = self._near_field_rescue(best_positions, best_tdoa, rw, x_hint=hint)
                if rescue.get("success", False):
                    rescue_dist = float(np.sqrt(rescue["x"]**2 + rescue["y"]**2))
                    rescue_residual = float(rescue.get("residual", float("inf")))
                    rescue_residual_limit = min(float(self.max_residual_m), 14.0)
                    rescue_candidate_ok = (
                        rescue_residual <= rescue_residual_limit
                        and rescue_dist <= self.max_range_m
                    )
                    if rescue_candidate_ok:
                        rescue_candidate = rescue

            if rescue_candidate_ok and rescue_candidate is not None:
                rescue_dist = float(np.sqrt(rescue_candidate["x"]**2 + rescue_candidate["y"]**2))
                rescue_residual = float(rescue_candidate.get("residual", float("inf")))

                accept_rescue = (
                    rescue_residual <= 0.92 * residual_now
                    or (
                        rescue_residual <= residual_now + 0.8
                        and rescue_dist <= (dist_from_ref - 16.0)
                    )
                )

                if accept_rescue:
                    return _with_latlon(rescue_candidate)

            if np.isfinite(geom_cond) and geom_cond > self.max_geometry_condition:
                return {
                    "success": False,
                    "reason": "unstable_geometry",
                    "condition": float(geom_cond),
                    "condition_limit": float(self.max_geometry_condition),
                    "ref_node": selected_ref_node,
                }

            if dist_from_ref > self.max_range_m:
                return {
                    "success": False,
                    "reason": "out_of_range",
                    "dist": float(dist_from_ref),
                    "ref_node": selected_ref_node,
                }

            result = _with_latlon(result)

        if selected_ref_node is not None and isinstance(result, dict):
            result["ref_node"] = selected_ref_node

        return result

    def _passes_physical_tdoa_consistency(self, packets: Dict[int, Dict]) -> bool:
        """Reject packet sets whose pairwise time deltas exceed physical limits."""
        ids = sorted(packets.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                n1, n2 = ids[i], ids[j]
                d_pair = self.node_pair_dist_m.get((n1, n2), self.node_pair_dist_m.get((n2, n1), None))
                if d_pair is None:
                    continue
                dt_s = abs(packets[n1]["ts_micros"] - packets[n2]["ts_micros"]) / 1_000_000.0
                measured_diff_m = dt_s * self.speed_of_sound
                if measured_diff_m > (d_pair + self.max_physical_slack_m):
                    return False
        return True

    def _jacobian_condition(self, positions: np.ndarray, tdoa: np.ndarray,
                            x: float, y: float) -> float:
        """Condition number of linearized TDoA Jacobian; high means unstable geometry."""
        try:
            n = len(positions)
            if n < 3:
                return float("inf")

            d = np.sqrt((x - positions[:, 0]) ** 2 + (y - positions[:, 1]) ** 2)
            d0 = d[0]
            if d0 < 1e-6:
                return float("inf")

            A = np.zeros((n - 1, 2), dtype=np.float64)
            for i in range(1, n):
                if d[i] < 1e-6:
                    return float("inf")
                A[i - 1, 0] = (x - positions[i, 0]) / d[i] - (x - positions[0, 0]) / d0
                A[i - 1, 1] = (y - positions[i, 1]) / d[i] - (y - positions[0, 1]) / d0

            s = np.linalg.svd(A, compute_uv=False)
            if len(s) < 2 or s[-1] < 1e-9:
                return float("inf")
            return float(s[0] / s[-1])
        except Exception:
            return float("inf")

    def _prepare_arrays(self, packets: Dict[int, Dict], node_ids: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build solver arrays given an explicit reference-first node ordering."""
        positions = []
        tdoa_values = []
        weights = []
        ref_ts = packets[node_ids[0]]["ts_micros"]

        for nid in node_ids:
            x, y = self.node_positions[nid]
            positions.append((x, y))
            dt_us = packets[nid]["ts_micros"] - ref_ts
            tdoa_values.append(dt_us / 1_000_000.0)
            snr = float(np.clip(packets[nid].get("snr_db", 10), 3.0, 35.0))
            conf = float(np.clip(packets[nid].get("ml_confidence", 50), 20.0, 100.0))
            weights.append(math.sqrt((snr * conf) / 100.0))

        positions = np.array(positions)
        tdoa = np.array(tdoa_values)
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        return positions, tdoa, weights

    def _choose_reference_node(self, packets: Dict[int, Dict], node_ids: List[int]) -> int:
        """Choose a deterministic TDoA reference node.

        Use the first-capture node (earliest timestamp) as the TDoA reference.
        If there is a tie, prefer the strongest packet quality (snr * confidence).
        """
        def _quality(nid: int) -> float:
            return (
                max(1.0, float(packets[nid].get("snr_db", 10.0)))
                * max(1.0, float(packets[nid].get("ml_confidence", 50.0)))
            )

        return min(
            node_ids,
            key=lambda nid: (
                int(packets[nid].get("ts_micros", 0)),
                -_quality(nid),
                nid,
            ),
        )

    def _pipeline(self, positions: np.ndarray, tdoa: np.ndarray,
                  weights: np.ndarray, node_ids: List[int]) -> Dict:
        """Auto pipeline: Chan-Ho → LM refinement → NM fallback."""
        # Stage 1: Chan-Ho (now with Stage 2 refinement internally)
        result = self._chan_ho(positions, tdoa, weights, node_ids)
        if result["success"]:
            # Stage 2: Refine with Taylor Series (Iterative)
            taylor_result = self._taylor_series(
                positions, tdoa, weights, node_ids,
                x0=np.array([result["x"], result["y"]])
            )
            if taylor_result["success"] and taylor_result["residual"] < result["residual"]:
                result = taylor_result
                result["method"] = "pipeline(chan_ho+taylor)"

            # Stage 2b: Multi-start Taylor to recover from difficult geometry/outliers.
            ms_result = self._multi_start_taylor(positions, tdoa, weights, node_ids, result)
            if ms_result.get("success", False) and ms_result.get("residual", float("inf")) < result.get("residual", float("inf")):
                result = ms_result

            # Stage 2c: robust nonlinear refinement with soft-L1 loss.
            if result.get("residual", 0.0) > 2.0:
                robust_result = self._robust_least_squares(
                    positions,
                    tdoa,
                    weights,
                    node_ids,
                    x0=np.array([result["x"], result["y"]]),
                )
                if robust_result.get("success", False):
                    old_xy = np.array([result["x"], result["y"]], dtype=np.float64)
                    new_xy = np.array([robust_result["x"], robust_result["y"]], dtype=np.float64)
                    jump_m = float(np.linalg.norm(new_xy - old_xy))
                    # Accept only meaningful residual gain with controlled spatial jump.
                    if (
                        robust_result.get("residual", float("inf")) < 0.90 * result.get("residual", float("inf"))
                        and jump_m < 180.0
                    ):
                        robust_result["method"] = "pipeline(robust_lsq)"
                        result = robust_result

            # Stage 3: Final polish with LM
            lm_result = self._levenberg_marquardt(
                positions, tdoa, weights, node_ids,
                x0=np.array([result["x"], result["y"]])
            )
            if lm_result["success"] and lm_result["residual"] < result.get("residual", float('inf')):
                lm_result["method"] = "pipeline(refined_lm)"
                return lm_result
            
            return result

        # Stage 3: Fallback to multi-start Nelder-Mead
        nm_result = self._nelder_mead(positions, tdoa, weights, node_ids)
        if nm_result["success"]:
            nm_result["method"] = "pipeline(nelder_mead_fallback)"
            robust_from_nm = self._robust_least_squares(
                positions,
                tdoa,
                weights,
                node_ids,
                x0=np.array([nm_result["x"], nm_result["y"]]),
            )
            if robust_from_nm.get("success", False) and robust_from_nm.get("residual", float("inf")) < nm_result.get("residual", float("inf")):
                robust_from_nm["method"] = "pipeline(nm+robust_lsq)"
                return robust_from_nm
        return nm_result

    def _robust_least_squares(self, positions: np.ndarray, tdoa: np.ndarray,
                              weights: np.ndarray, node_ids: List[int],
                              x0: Optional[np.ndarray] = None) -> Dict:
        """Robust nonlinear least-squares using soft-L1 loss against TDoA outliers."""
        import time as _time
        t_start = _time.time()

        if not self.scipy_enabled:
            return {"success": False, "reason": "scipy_disabled"}

        try:
            from scipy.optimize import least_squares

            c = self.speed_of_sound
            if x0 is None:
                x0 = np.mean(positions, axis=0)

            cx = float(np.mean(positions[:, 0]))
            cy = float(np.mean(positions[:, 1]))
            sx = float(max(1.0, np.ptp(positions[:, 0])))
            sy = float(max(1.0, np.ptp(positions[:, 1])))
            max_dt = float(np.max(np.abs(tdoa[1:]))) if len(tdoa) > 1 else 0.0
            dyn_extent = max(500.0, 1.8 * c * max_dt)

            starts = [
                np.array([x0[0], x0[1]], dtype=np.float64),
                np.array([cx, cy], dtype=np.float64),
                np.array([cx + 0.18 * sx, cy], dtype=np.float64),
                np.array([cx - 0.18 * sx, cy], dtype=np.float64),
                np.array([cx, cy + 0.18 * sy], dtype=np.float64),
                np.array([cx, cy - 0.18 * sy], dtype=np.float64),
            ]

            min_x = float(np.min(positions[:, 0]) - dyn_extent)
            max_x = float(np.max(positions[:, 0]) + dyn_extent)
            min_y = float(np.min(positions[:, 1]) - dyn_extent)
            max_y = float(np.max(positions[:, 1]) + dyn_extent)

            def residuals(xy: np.ndarray) -> np.ndarray:
                x, y = float(xy[0]), float(xy[1])
                d_ref = math.sqrt((x - positions[0][0]) ** 2 + (y - positions[0][1]) ** 2)
                res = []
                for i in range(1, len(positions)):
                    d_i = math.sqrt((x - positions[i][0]) ** 2 + (y - positions[i][1]) ** 2)
                    pred_m = d_i - d_ref
                    meas_m = tdoa[i] * c
                    w = float(weights[i] if i < len(weights) else 1.0)
                    res.append(math.sqrt(max(1e-6, w)) * (pred_m - meas_m))
                return np.array(res, dtype=np.float64)

            best = None
            best_cost = float("inf")
            for start in starts:
                try:
                    lsq = least_squares(
                        residuals,
                        start,
                        method="trf",
                        loss="soft_l1",
                        f_scale=3.0,
                        bounds=([min_x, min_y], [max_x, max_y]),
                        max_nfev=320,
                    )
                except Exception:
                    continue

                if lsq.cost < best_cost:
                    best = lsq
                    best_cost = float(lsq.cost)

            if best is None:
                return {"success": False, "reason": "robust_lsq_failed"}

            x_est, y_est = best.x
            residual = self._compute_residual(positions, tdoa, x_est, y_est)
            return {
                "success": True,
                "x": float(x_est),
                "y": float(y_est),
                "residual": float(residual),
                "method": "robust_lsq",
                "iterations": int(getattr(best, "nfev", 0)),
                "compute_time_ms": (_time.time() - t_start) * 1000,
            }
        except Exception as e:
            return {"success": False, "reason": f"robust_lsq_error: {e}"}

    def _near_field_rescue(self, positions: np.ndarray, tdoa: np.ndarray,
                           weights: np.ndarray, x_hint: Optional[np.ndarray] = None) -> Dict:
        """Bounded robust fit around the array footprint to recover corner scenarios."""
        if not self.scipy_enabled:
            return {"success": False, "reason": "scipy_disabled"}

        try:
            from scipy.optimize import least_squares

            c = self.speed_of_sound
            cx = float(np.mean(positions[:, 0]))
            cy = float(np.mean(positions[:, 1]))
            r = float(self.near_field_rescue_radius_m)

            min_x, max_x = cx - r, cx + r
            min_y, max_y = cy - r, cy + r

            starts = [np.array([cx, cy], dtype=np.float64)]
            if x_hint is not None and len(x_hint) == 2:
                xh = float(np.clip(x_hint[0], min_x, max_x))
                yh = float(np.clip(x_hint[1], min_y, max_y))
                starts.append(np.array([xh, yh], dtype=np.float64))

            def residuals(xy: np.ndarray) -> np.ndarray:
                x, y = float(xy[0]), float(xy[1])
                d_ref = math.sqrt((x - positions[0][0]) ** 2 + (y - positions[0][1]) ** 2)
                out = []
                for i in range(1, len(positions)):
                    d_i = math.sqrt((x - positions[i][0]) ** 2 + (y - positions[i][1]) ** 2)
                    pred_m = d_i - d_ref
                    meas_m = tdoa[i] * c
                    w = float(weights[i] if i < len(weights) else 1.0)
                    out.append(math.sqrt(max(1e-6, w)) * (pred_m - meas_m))
                return np.array(out, dtype=np.float64)

            best = None
            best_cost = float("inf")
            for start in starts:
                try:
                    lsq = least_squares(
                        residuals,
                        start,
                        method="trf",
                        loss="soft_l1",
                        f_scale=3.0,
                        bounds=([min_x, min_y], [max_x, max_y]),
                        max_nfev=240,
                    )
                except Exception:
                    continue

                if lsq.cost < best_cost:
                    best = lsq
                    best_cost = float(lsq.cost)

            if best is None:
                return {"success": False, "reason": "near_field_rescue_failed"}

            x_est, y_est = best.x
            residual = self._compute_residual(positions, tdoa, x_est, y_est)
            return {
                "success": True,
                "x": float(x_est),
                "y": float(y_est),
                "residual": float(residual),
                "method": "near_field_rescue",
            }
        except Exception as e:
            return {"success": False, "reason": f"near_field_rescue_error: {e}"}

    def _multi_start_taylor(self, positions: np.ndarray, tdoa: np.ndarray,
                            weights: np.ndarray, node_ids: List[int],
                            base_result: Dict) -> Dict:
        """Run several Taylor initializations and keep the best residual solution."""
        try:
            if not base_result.get("success", False):
                return {"success": False}

            cx, cy = np.mean(positions[:, 0]), np.mean(positions[:, 1])
            sx = max(1.0, np.ptp(positions[:, 0]))
            sy = max(1.0, np.ptp(positions[:, 1]))

            starts = [
                np.array([base_result["x"], base_result["y"]], dtype=np.float64),
                np.array([cx, cy], dtype=np.float64),
                np.array([cx + 0.20 * sx, cy], dtype=np.float64),
                np.array([cx - 0.20 * sx, cy], dtype=np.float64),
                np.array([cx, cy + 0.20 * sy], dtype=np.float64),
                np.array([cx, cy - 0.20 * sy], dtype=np.float64),
            ]

            best = dict(base_result)
            for x0 in starts:
                cand = self._taylor_series(positions, tdoa, weights, node_ids, x0=x0, max_iters=12)
                if cand.get("success", False) and cand.get("residual", float("inf")) < best.get("residual", float("inf")):
                    cand["method"] = "pipeline(chan_ho+taylor_ms)"
                    best = cand
            return best
        except Exception:
            return {"success": False}

    def _chan_ho(self, positions: np.ndarray, tdoa: np.ndarray,
                 weights: np.ndarray, node_ids: List[int]) -> Dict:
        """
        Chan-Ho closed-form TDoA solver with Stage 1 & 2 WLS.
        Based on Y.T. Chan & K.C. Ho (1994).
        """
        import time as _time
        t_start = _time.time()

        try:
            n = len(positions)
            if n < 3:
                return {"success": False, "reason": "need_3_positions"}

            c = self.speed_of_sound
            x0, y0 = positions[0]

            # --- STAGE 1 ---
            # Build the system Ga = h
            G = np.zeros((n - 1, 3))
            h = np.zeros(n - 1)

            for i in range(1, n):
                xi, yi = positions[i]
                ri0 = tdoa[i] * c
                G[i - 1, 0] = -(xi - x0)
                G[i - 1, 1] = -(yi - y0)
                G[i - 1, 2] = -ri0
                h[i - 1] = 0.5 * (ri0**2 - xi**2 + x0**2 - yi**2 + y0**2)

            # Weighting matrix for Stage 1 (Simplified diagonal if noise unknown)
            # True Chan-Ho uses a covariance matrix, we use node weights + distance scaling
            W1 = np.diag(weights[1:])
            
            try:
                # Weighted Least Squares: (G^T W G)^-1 G^T W h
                GtW = G.T @ W1
                theta = np.linalg.solve(GtW @ G, GtW @ h)
            except np.linalg.LinAlgError:
                # Fallback to standard lstsq if singular
                theta = np.linalg.lstsq(G, h, rcond=None)[0]

            x_est = theta[0]
            y_est = theta[1]
            r0_est = theta[2]

            # --- STAGE 2 (Refinement) ---
            # Refine (x,y) by considering the quadratic constraint r0^2 = (x-x0)^2 + (y-y0)^2
            # only if n > 3 or if r0_est is physically plausible
            if n >= 3:
                # G2 matrix for Stage 2
                G2 = np.array([[1, 0], [0, 1], [1, 1]])
                h2 = np.array([
                    (x_est - x0)**2,
                    (y_est - y0)**2,
                    r0_est**2
                ])
                
                # Weighting for Stage 2
                # We reuse Stage 1 estimate to scale variances
                B = np.diag([abs(x_est - x0), abs(y_est - y0), abs(r0_est)])
                W2 = np.linalg.pinv(B) @ np.linalg.pinv(B)
                
                try:
                    theta2 = np.linalg.solve(G2.T @ W2 @ G2, G2.T @ W2 @ h2)
                    # Correct signs from Stage 1
                    x_final = np.sign(x_est - x0) * np.sqrt(abs(theta2[0])) + x0
                    y_final = np.sign(y_est - y0) * np.sqrt(abs(theta2[1])) + y0
                    
                    # Update estimate if residual improves
                    res1 = self._compute_residual(positions, tdoa, x_est, y_est)
                    res2 = self._compute_residual(positions, tdoa, x_final, y_final)
                    
                    if res2 < res1:
                        x_est, y_est = x_final, y_final
                except Exception:
                    pass # Keep Stage 1 if Stage 2 fails

            residual = self._compute_residual(positions, tdoa, x_est, y_est)
            elapsed = _time.time() - t_start
            
            return {
                "success": True,
                "x": float(x_est), "y": float(y_est),
                "residual": float(residual),
                "method": "chan_ho",
                "compute_time_ms": elapsed * 1000,
            }

        except Exception as e:
            return {"success": False, "reason": f"chan_ho_error: {e}"}

    def _taylor_series(self, positions: np.ndarray, tdoa: np.ndarray,
                      weights: np.ndarray, node_ids: List[int],
                      x0: np.ndarray, max_iters: int = 10) -> Dict:
        """Taylor Series iterative refinement (Linearization of TDoA equations)."""
        import time as _time
        t_start = _time.time()
        c = self.speed_of_sound
        x, y = x0
        
        try:
            for _ in range(max_iters):
                # Distances to nodes
                d = np.sqrt((x - positions[:, 0])**2 + (y - positions[:, 1])**2)
                # Reference node is index 0
                d0 = d[0]
                
                # Jacobian matrix A and residual vector b
                A = np.zeros((len(positions) - 1, 2))
                b = np.zeros(len(positions) - 1)
                
                for i in range(1, len(positions)):
                    # Preprocess derivatives
                    if d[i] < 0.1 or d0 < 0.1: continue
                    
                    A[i-1, 0] = (x - positions[i, 0]) / d[i] - (x - positions[0, 0]) / d0
                    A[i-1, 1] = (y - positions[i, 1]) / d[i] - (y - positions[0, 1]) / d0
                    
                    # Range difference (measured - predicted)
                    measured_ri0 = tdoa[i] * c
                    predicted_ri0 = d[i] - d0
                    b[i-1] = measured_ri0 - predicted_ri0
                
                # Weighted solution
                W = np.diag(weights[1:])
                try:
                    delta = np.linalg.solve(A.T @ W @ A, A.T @ W @ b)
                    x += delta[0]
                    y += delta[1]
                    if np.linalg.norm(delta) < 0.01: break
                except np.linalg.LinAlgError:
                    break
            
            residual = self._compute_residual(positions, tdoa, x, y)
            return {
                "success": True,
                "x": float(x), "y": float(y),
                "residual": float(residual),
                "method": "taylor_series",
                "compute_time_ms": (_time.time() - t_start) * 1000
            }
        except Exception:
            return {"success": False}

    def _levenberg_marquardt(self, positions: np.ndarray, tdoa: np.ndarray,
                             weights: np.ndarray, node_ids: List[int],
                             x0: Optional[np.ndarray] = None) -> Dict:
        """Levenberg-Marquardt iterative nonlinear solver."""
        import time as _time
        t_start = _time.time()

        if not self.scipy_enabled:
            return {"success": False, "reason": "scipy_disabled"}

        try:
            c = self.speed_of_sound
            if x0 is None:
                # Use centroid as initial guess
                x0 = np.mean(positions, axis=0)

            def residuals(xy):
                x, y = xy
                res = []
                # Reference node
                d_ref = np.sqrt((x - positions[0][0]) ** 2 + (y - positions[0][1]) ** 2)
                for i in range(1, len(positions)):
                    d_i = np.sqrt((x - positions[i][0]) ** 2 + (y - positions[i][1]) ** 2)
                    predicted_dt = (d_i - d_ref) / c
                    measured_dt = tdoa[i]
                    w = weights[i] if i < len(weights) else 1.0
                    res.append(w * (predicted_dt - measured_dt))
                return np.array(res)

            try:
                from scipy.optimize import least_squares
                result = least_squares(residuals, x0, method='lm', max_nfev=200, 
                                       ftol=1e-8, xtol=1e-8)
            except (ImportError, Exception):
                # Graceful fallback if scipy is broken or hangs
                return {"success": False, "reason": "scipy_unavailable_fallback"}

            if result.success or result.cost < 1.0:
                x_est, y_est = result.x
                residual = self._compute_residual(positions, tdoa, x_est, y_est)
                elapsed = _time.time() - t_start
                return {
                    "success": True,
                    "x": float(x_est), "y": float(y_est),
                    "residual": float(residual),
                    "method": "levenberg_marquardt",
                    "iterations": result.nfev,
                    "compute_time_ms": elapsed * 1000,
                }
            else:
                return {"success": False, "reason": "lm_no_convergence"}

        except Exception as e:
            return {"success": False, "reason": f"lm_error: {e}"}

    def _nelder_mead(self, positions: np.ndarray, tdoa: np.ndarray,
                     weights: np.ndarray, node_ids: List[int]) -> Dict:
        """Multi-start Nelder-Mead solver (fallback)."""
        import time as _time
        t_start = _time.time()

        if not self.scipy_enabled:
            return {"success": False, "reason": "scipy_disabled"}

        try:
            c = self.speed_of_sound

            def objective(xy):
                x, y = xy
                d_ref = np.sqrt((x - positions[0][0]) ** 2 + (y - positions[0][1]) ** 2)
                total = 0.0
                for i in range(1, len(positions)):
                    d_i = np.sqrt((x - positions[i][0]) ** 2 + (y - positions[i][1]) ** 2)
                    predicted_dt = (d_i - d_ref) / c
                    measured_dt = tdoa[i]
                    w = weights[i] if i < len(weights) else 1.0
                    total += w * (predicted_dt - measured_dt) ** 2
                return total

            # Multi-start: centroid + grid points
            cx = np.mean(positions[:, 0])
            cy = np.mean(positions[:, 1])
            spread_x = np.ptp(positions[:, 0])
            spread_y = np.ptp(positions[:, 1])

            starts = [(cx, cy)]
            for dx in [-0.3, 0, 0.3]:
                for dy in [-0.3, 0, 0.3]:
                    starts.append((cx + dx * spread_x, cy + dy * spread_y))

            best_result = None
            best_cost = float('inf')

            for x0 in starts:
                try:
                    from scipy.optimize import minimize
                    result = minimize(objective, x0, method='Nelder-Mead',
                                      options={'maxiter': 500, 'xatol': 0.01, 'fatol': 1e-12})
                except (ImportError, Exception):
                    return {"success": False, "reason": "scipy_unavailable_fallback"}
                if result.fun < best_cost:
                    best_cost = result.fun
                    best_result = result

            if best_result is not None and best_result.success:
                x_est, y_est = best_result.x
                residual = self._compute_residual(positions, tdoa, x_est, y_est)
                elapsed = _time.time() - t_start
                return {
                    "success": True,
                    "x": float(x_est), "y": float(y_est),
                    "residual": float(residual),
                    "method": "nelder_mead",
                    "iterations": best_result.nit,
                    "compute_time_ms": elapsed * 1000,
                }
            else:
                return {"success": False, "reason": "nm_no_convergence"}

        except Exception as e:
            return {"success": False, "reason": f"nm_error: {e}"}

    def solve_all_methods(self, packets: Dict[int, Dict]) -> Dict[str, Dict]:
        """Run all solver methods and return results for comparison."""
        node_ids = sorted(packets.keys())
        positions, tdoa, weights = self._extract_data(packets, node_ids)

        results = {}
        results["chan_ho"] = self._chan_ho(positions, tdoa, weights, node_ids)

        # LM with Chan-Ho initialization
        x0 = None
        if results["chan_ho"]["success"]:
            x0 = np.array([results["chan_ho"]["x"], results["chan_ho"]["y"]])
        results["lm"] = self._levenberg_marquardt(positions, tdoa, weights, node_ids, x0)
        results["nelder_mead"] = self._nelder_mead(positions, tdoa, weights, node_ids)

        return results

    def _extract_data(self, packets: Dict[int, Dict],
                      node_ids: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract positions, TDoA, and weights from packets."""
        # Keep method comparison consistent with solve() by using the same reference policy.
        ref_node = self._choose_reference_node(packets, node_ids)
        node_ids = [ref_node] + [nid for nid in node_ids if nid != ref_node]

        positions = []
        tdoa_values = []
        weights = []
        ref_ts = packets[node_ids[0]]["ts_micros"]

        for nid in node_ids:
            x, y = self.node_positions[nid]
            positions.append((x, y))
            dt_us = packets[nid]["ts_micros"] - ref_ts
            tdoa_values.append(dt_us / 1_000_000.0)
            snr = float(np.clip(packets[nid].get("snr_db", 10), 3.0, 35.0))
            conf = float(np.clip(packets[nid].get("ml_confidence", 50), 20.0, 100.0))
            weights.append(math.sqrt((snr * conf) / 100.0))

        positions = np.array(positions)
        tdoa = np.array(tdoa_values)
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        return positions, tdoa, weights

    def _compute_residual(self, positions: np.ndarray, tdoa: np.ndarray,
                          x: float, y: float) -> float:
        """Compute RMSE of range-difference mismatch in meters."""
        c = self.speed_of_sound
        d_ref = np.sqrt((x - positions[0][0]) ** 2 + (y - positions[0][1]) ** 2)
        errors = []
        for i in range(1, len(positions)):
            d_i = np.sqrt((x - positions[i][0]) ** 2 + (y - positions[i][1]) ** 2)
            predicted_dt = (d_i - d_ref) / c
            measured_dt = tdoa[i]
            # Convert time mismatch to equivalent range mismatch (meters).
            errors.append(((predicted_dt - measured_dt) * c) ** 2)
        return float(np.sqrt(np.mean(errors))) if errors else 0.0
