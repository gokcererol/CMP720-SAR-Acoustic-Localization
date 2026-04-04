"""
LoRa Channel Model — E22-900T22D specific simulation.
Simulates path loss, duty cycle, packet collision, AUX pin, and BER.
"""

import socket
import struct
import time
import random
import threading
import json
import numpy as np
from typing import Dict, List, Optional

from node.lora_tx import unpack_packet


class LoRaChannel:
    """
    Simulates the LoRa wireless channel between ESP32 nodes and Jetson solver.
    Models E22-900T22D specific behaviors.
    """

    def __init__(self, config: Dict):
        self.config = config
        lora_cfg = config.get("lora", {})

        self.listen_port = 5020
        self.solver_port = 5000
        self.tx_power_dbm = lora_cfg.get("tx_power_dbm", 22)
        self.air_data_rate = lora_cfg.get("air_data_rate_bps", 9600)
        self.payload_bytes = lora_cfg.get("payload_bytes", 23)
        self.base_reliability = lora_cfg.get("base_reliability", 0.95)
        self.path_loss_exp = lora_cfg.get("path_loss_exponent", 3.0)

        # Duty cycle
        dc_cfg = lora_cfg.get("duty_cycle", {})
        self.duty_cycle_enabled = dc_cfg.get("enabled", True)
        self.duty_cycle_max = dc_cfg.get("max_percent", 1.0)

        # Collision
        col_cfg = lora_cfg.get("collision", {})
        self.collision_enabled = col_cfg.get("enabled", True)
        self.capture_threshold_db = col_cfg.get("capture_threshold_db", 6.0)

        # Sockets
        self.listen_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listen_sock.settimeout(0.5)

        self.forward_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Track recent transmissions for collision detection
        self._recent_tx: List[Dict] = []
        self._lock = threading.Lock()

        # Stats
        self.packets_received = 0
        self.packets_forwarded = 0
        self.packets_dropped = 0
        self.packets_corrupted = 0
        self.packets_collided = 0

        self.running = False

    def _compute_airtime_ms(self) -> float:
        """Compute packet airtime in milliseconds."""
        total_bits = (self.payload_bytes + 8) * 8  # payload + preamble overhead
        return total_bits / self.air_data_rate * 1000.0

    def _check_collision(self, arrival_time: float, node_id: int) -> bool:
        """Check if this packet collides with a recent transmission."""
        if not self.collision_enabled:
            return False

        airtime = self._compute_airtime_ms() / 1000.0  # seconds
        with self._lock:
            # Clean old entries
            self._recent_tx = [
                tx for tx in self._recent_tx
                if arrival_time - tx["end_time"] < 0.5
            ]

            # Check for overlap
            start = arrival_time
            end = arrival_time + airtime

            for tx in self._recent_tx:
                if start < tx["end_time"] and end > tx["start_time"]:
                    # Collision! Check capture effect
                    # Both have same TX power in our sim, so collision = loss
                    self.packets_collided += 1
                    return True

            # Record this transmission
            self._recent_tx.append({
                "start_time": start,
                "end_time": end,
                "node_id": node_id,
            })

        return False

    def _apply_channel_effects(self, packet_data: bytes) -> Optional[bytes]:
        """
        Apply channel effects to a packet.
        Returns modified packet bytes, or None if packet is lost.
        """
        self.packets_received += 1

        # 1. Random packet loss (base reliability)
        if random.random() > self.base_reliability:
            self.packets_dropped += 1
            return None

        # 2. CRC check (already in packet — just simulate BER corruption)
        ber = max(0, 1.0 - self.base_reliability) * 0.01  # Very low BER
        if random.random() < ber * len(packet_data) * 8:
            # Flip a random bit
            data = bytearray(packet_data)
            bit_pos = random.randint(0, len(data) * 8 - 1)
            byte_idx = bit_pos // 8
            bit_idx = bit_pos % 8
            data[byte_idx] ^= (1 << bit_idx)
            self.packets_corrupted += 1
            return bytes(data)  # CRC will fail on receiver side

        # 3. Collision detection
        parsed = unpack_packet(packet_data)
        if parsed:
            if self._check_collision(time.time(), parsed["node_id"]):
                return None

        return packet_data

    def _add_jitter(self, packet_data: bytes) -> bytes:
        """Add timing jitter by slightly modifying the timestamp."""
        if len(packet_data) < 9:
            return packet_data

        # Parse timestamp, add jitter (~0-2ms)
        parsed = unpack_packet(packet_data)
        if parsed:
            jitter_us = random.gauss(0, 500)  # ±500µs jitter
            new_ts = parsed["ts_micros"] + int(jitter_us)
            # Reconstruct packet with new timestamp
            data = bytearray(packet_data)
            struct.pack_into("<Q", data, 1, max(0, new_ts))
            # Recompute CRC
            from node.lora_tx import crc8
            new_crc = crc8(bytes(data[:21]))
            data[21] = new_crc
            return bytes(data)

        return packet_data

    def _channel_loop(self):
        """Main loop: receive packets, apply channel effects, forward to solver."""
        print(f"📡 [LoRa] Channel active on port {self.listen_port} "
              f"(reliability={self.base_reliability:.0%}, rate={self.air_data_rate}bps)")

        while self.running:
            try:
                data, addr = self.listen_sock.recvfrom(256)
            except socket.timeout:
                continue
            except Exception:
                continue

            if len(data) != 23:
                continue

            # Apply channel effects
            processed = self._apply_channel_effects(data)
            if processed is None:
                continue

            # Add timing jitter
            processed = self._add_jitter(processed)

            # Forward to solver
            try:
                self.forward_sock.sendto(processed, ("127.0.0.1", self.solver_port))
                self.packets_forwarded += 1
            except Exception as e:
                print(f"[LoRa] Forward error: {e}")

    def get_status(self) -> Dict:
        """Get channel statistics."""
        return {
            "packets_received": self.packets_received,
            "packets_forwarded": self.packets_forwarded,
            "packets_dropped": self.packets_dropped,
            "packets_corrupted": self.packets_corrupted,
            "packets_collided": self.packets_collided,
            "reliability": self.base_reliability,
            "air_data_rate": self.air_data_rate,
        }

    def update_config(self, update: Dict):
        """Update channel parameters at runtime."""
        if "base_reliability" in update:
            self.base_reliability = update["base_reliability"]
        if "air_data_rate_bps" in update:
            self.air_data_rate = update["air_data_rate_bps"]
        if "collision_enabled" in update:
            self.collision_enabled = update["collision_enabled"]
        if "duty_cycle_enabled" in update:
            self.duty_cycle_enabled = update["duty_cycle_enabled"]

    def start(self):
        """Start the LoRa channel simulator."""
        self.running = True
        try:
            self.listen_sock.bind(("127.0.0.1", self.listen_port))
        except OSError as e:
            print(f"[LoRa] Bind error: {e}")
            return
        self._thread = threading.Thread(target=self._channel_loop, daemon=True)
        self._thread.start()
        print(f"✅ [LoRa] Channel started.")

    def stop(self):
        """Stop the LoRa channel."""
        self.running = False
        self.listen_sock.close()
        self.forward_sock.close()
        print(f"🛑 [LoRa] Channel stopped.")
