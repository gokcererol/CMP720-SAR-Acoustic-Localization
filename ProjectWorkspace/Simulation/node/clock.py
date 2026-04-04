"""
Clock Model — Simulates ESP32 crystal oscillator drift and GPS PPS correction.
Each node gets a unique, persistent clock offset.
"""

import random
import time
from typing import Dict


class ClockModel:
    """
    Simulates ESP32 crystal drift and GPS PPS time correction.
    - Crystal: 40 MHz with ±10-25 ppm tolerance
    - GPS PPS: corrects once per second with ±50-500ns jitter
    """

    def __init__(self, node_id: int, ppm_sigma: float = 10.0,
                 pps_jitter_ns: float = 100.0, cold_start_sec: float = 60.0):
        self.node_id = node_id
        self.ppm_sigma = ppm_sigma
        self.pps_jitter_ns = pps_jitter_ns
        self.cold_start_sec = cold_start_sec

        # Each node gets a random but persistent PPM offset
        random.seed(node_id * 12345 + 6789)
        self.ppm_offset = random.gauss(0, ppm_sigma)
        random.seed()  # Re-randomize for other uses

        # State tracking
        # Set boot time earlier so GPS locks quickly in simulation
        self.boot_time = time.time() - cold_start_sec - 1.0
        self.last_pps_time = self.boot_time
        self.accumulated_drift = 0.0
        self.gps_locked = False
        self.total_drift_applied = 0.0

    def get_timestamp(self, true_time: float) -> float:
        """
        Returns a drifted timestamp given the true wall-clock time.
        Simulates crystal drift accumulation between GPS PPS corrections.
        """
        elapsed_since_boot = true_time - self.boot_time

        # GPS cold start: no PPS lock for the first N seconds
        if elapsed_since_boot < self.cold_start_sec:
            self.gps_locked = False
        else:
            self.gps_locked = True

        # Crystal drift since last PPS
        elapsed_since_pps = true_time - self.last_pps_time
        crystal_drift = self.ppm_offset * 1e-6 * elapsed_since_pps

        if self.gps_locked:
            # PPS corrects once per second
            if elapsed_since_pps >= 1.0:
                # PPS pulse fires — reset drift, add PPS jitter
                pps_jitter = random.gauss(0, self.pps_jitter_ns * 1e-9)
                self.accumulated_drift = pps_jitter
                self.last_pps_time = true_time
                crystal_drift = 0.0
        else:
            # No GPS lock — drift accumulates freely
            self.accumulated_drift = crystal_drift

        drifted_time = true_time + self.accumulated_drift + crystal_drift
        self.total_drift_applied = self.accumulated_drift + crystal_drift
        return drifted_time

    def get_timestamp_micros(self, true_time: float) -> int:
        """Returns drifted timestamp in microseconds (matching ESP32 format)."""
        return int(self.get_timestamp(true_time) * 1_000_000)

    def is_gps_locked(self) -> bool:
        """Check if GPS has achieved PPS lock."""
        return self.gps_locked

    def get_drift_info(self) -> Dict:
        """Get current clock drift information."""
        return {
            "node_id": self.node_id,
            "ppm_offset": self.ppm_offset,
            "total_drift_us": self.total_drift_applied * 1e6,
            "gps_locked": self.gps_locked,
        }

    def update_config(self, config: Dict):
        """Update clock parameters at runtime."""
        if "crystal_ppm_sigma" in config:
            self.ppm_sigma = config["crystal_ppm_sigma"]
        if "gps_pps_jitter_ns" in config:
            self.pps_jitter_ns = config["gps_pps_jitter_ns"]
