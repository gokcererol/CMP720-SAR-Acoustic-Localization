"""
Packet Collector — Receives LoRa packets from the channel simulator,
groups them by event using timestamp proximity, and assembles event records.
"""

import socket
import time
import threading
from typing import Dict, List, Optional

from node.lora_tx import unpack_packet


class PacketCollector:
    """
    Collects TDoA packets from the LoRa channel and groups them by event.
    When enough packets arrive within the timeout window, an event is triggered.
    """

    def __init__(self, listen_port: int = 5000, packet_timeout_sec: float = 1.8,
                 min_packets: int = 3):
        self.listen_port = listen_port
        self.packet_timeout = packet_timeout_sec
        self.min_packets = min_packets

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(0.5)

        # Buffer: groups packets by approximate event time
        self._event_buffer: Dict[str, Dict] = {}
        self._lock = threading.Lock()

        # Callback when an event is ready
        self._event_callback = None

        self.running = False
        self.total_packets = 0
        self.total_events = 0

    def set_event_callback(self, callback):
        """Set callback function called when an event is assembled."""
        self._event_callback = callback

    def _find_or_create_group(self, packet: Dict) -> str:
        """
        Finds an active event group where the packet's timestamp is within 500ms
        of the group's reference timestamp. If none found, creates a new one.
        """
        ts_micros = packet["ts_micros"]
        now = time.time()
        
        with self._lock:
            # Search for an existing group within 0.5 seconds
            for key, buf in self._event_buffer.items():
                ref_ts = buf["ref_ts_micros"]
                if abs(ts_micros - ref_ts) <= 500_000:
                    buf["packets"][packet["node_id"]] = packet
                    buf["last_arrival"] = now
                    return key
            
            # Create a new group using the current packet's timestamp as the key
            key = f"evt_{ts_micros}"
            self._event_buffer[key] = {
                "packets": {packet["node_id"]: packet},
                "first_arrival": now,
                "last_arrival": now,
                "ref_ts_micros": ts_micros,
            }
            return key

    def _collect_loop(self):
        """Main loop: receive packets and group them."""
        print(f"📥 [Collector] Listening on port {self.listen_port}...")

        while self.running:
            # Receive packets
            try:
                data, addr = self.sock.recvfrom(256)
            except socket.timeout:
                self._check_timeouts()
                continue
            except Exception:
                continue

            packet = unpack_packet(data)
            if packet is None:
                continue  # CRC failed

            self.total_packets += 1
            key = self._find_or_create_group(packet)
            
            # Print status safely with lock
            with self._lock:
                if key in self._event_buffer:
                    buf = self._event_buffer[key]
                    print(f"📨 [Collector] Packet from Node {packet['node_id']} | "
                          f"TS: {packet['ts_micros']} | Group: {key} | "
                          f"Total in group: {len(buf['packets'])}")

            self._check_timeouts()

    def _check_timeouts(self):
        """Check if any event groups have timed out and should be processed."""
        now = time.time()
        to_process = []

        with self._lock:
            expired_keys = []
            for key, buf in self._event_buffer.items():
                elapsed = now - buf["first_arrival"]
                if elapsed >= self.packet_timeout:
                    if len(buf["packets"]) >= self.min_packets:
                        to_process.append((key, buf))
                    expired_keys.append(key)

            for key in expired_keys:
                del self._event_buffer[key]

        # Process completed events
        for key, buf in to_process:
            self.total_events += 1
            event = {
                "event_id": self.total_events,
                "group_key": key,
                "packets": buf["packets"],
                "num_packets": len(buf["packets"]),
                "collection_time": now,
            }

            print(f"📦 [Collector] Event #{event['event_id']} assembled: "
                  f"{event['num_packets']}/4 packets")

            if self._event_callback:
                try:
                    self._event_callback(event)
                except Exception as e:
                    print(f"[Collector] Callback error: {e}")

    def get_status(self) -> Dict:
        """Get collector statistics."""
        return {
            "total_packets": self.total_packets,
            "total_events": self.total_events,
            "pending_groups": len(self._event_buffer),
        }

    def update_config(self, update: Dict):
        """Update collector parameters."""
        if "packet_timeout_sec" in update:
            self.packet_timeout = update["packet_timeout_sec"]
        if "min_packets" in update:
            self.min_packets = update["min_packets"]

    def start(self):
        """Start the collector."""
        self.running = True
        try:
            self.sock.bind(("127.0.0.1", self.listen_port))
        except OSError as e:
            print(f"[Collector] Bind error: {e}")
            return
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
        print(f"✅ [Collector] Started.")

    def stop(self):
        """Stop the collector."""
        self.running = False
        self.sock.close()
        print(f"🛑 [Collector] Stopped.")
