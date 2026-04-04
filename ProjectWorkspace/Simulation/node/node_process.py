"""
Node Process — Main ESP32 node emulation loop.
Receives continuous PCM audio stream, processes through STA/LTA → FFT → ML pipeline,
and transmits 23-byte LoRa packets when target sounds are detected.
"""

import socket
import struct
import time
import json
import threading
import numpy as np
from typing import Dict

from node.stream_processor import StreamProcessor
from node.fft_filter import FFTFilter
from node.ml_classifier import SoundClassifier
from node.clock import ClockModel
from node.lora_tx import LoRaTX


class NodeProcess:
    """
    Emulates one ESP32 sensor node.
    Receives continuous audio via UDP, processes it, and sends TDoA packets.
    """

    def __init__(self, node_id: int, config: Dict):
        self.node_id = node_id
        self.config = config
        self.running = False

        # Port assignments
        self.listen_port = 5010 + node_id  # 5011-5014
        self.lora_port = 5020

        audio_cfg = config.get("audio", {})
        node_cfg = config.get("nodes", {})

        # Stream processor
        sp_cfg = node_cfg.get("stream_processor", {})
        self.stream_proc = StreamProcessor(
            sample_rate=audio_cfg.get("sample_rate", 16000),
            chunk_size=audio_cfg.get("chunk_size", 1024),
            sta_window_ms=sp_cfg.get("sta_window_ms", 50),
            lta_window_ms=sp_cfg.get("lta_window_ms", 5000),
            sta_lta_threshold=sp_cfg.get("sta_lta_threshold", 5.0),
            cooldown_ms=sp_cfg.get("cooldown_ms", 500),
        )

        # FFT filter
        fft_cfg = node_cfg.get("fft_prefilter", {})
        self.fft_filter = FFTFilter(
            sample_rate=audio_cfg.get("sample_rate", 16000),
            freq_min=fft_cfg.get("whistle_freq_min", 2500),
            freq_max=fft_cfg.get("whistle_freq_max", 4000),
            enabled=fft_cfg.get("enabled", True),
        )

        # ML classifier
        ml_cfg = node_cfg.get("ml_classifier", {})
        model_path = config.get("_model_path", None)
        self.classifier = SoundClassifier(
            model_path=model_path,
            confidence_threshold=ml_cfg.get("confidence_threshold", 0.7),
            target_classes=ml_cfg.get("target_classes",
                                      ["whistle", "human_voice", "impact", "knocking"]),
        )
        self.whistle_rescue_enabled = bool(ml_cfg.get("whistle_rescue_enabled", True))
        self.whistle_rescue_min_peak = int(ml_cfg.get("whistle_rescue_min_peak", 400))

        # Clock model
        clk_cfg = node_cfg.get("clock", {})
        self.clock = ClockModel(
            node_id=node_id,
            ppm_sigma=clk_cfg.get("crystal_ppm_sigma", 10.0),
            pps_jitter_ns=clk_cfg.get("gps_pps_jitter_ns", 100.0),
            cold_start_sec=clk_cfg.get("gps_cold_start_sec", 60.0),
        )

        # LoRa transmitter
        bat_cfg = node_cfg.get("battery", {})
        self.lora_tx = LoRaTX(
            node_id=node_id,
            lora_channel_port=self.lora_port,
            battery_pct=bat_cfg.get("initial_percent", 100),
        )
        self.battery_drain = bat_cfg.get("drain_rate_per_event", 0.1)

        # UDP listener socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(0.5)

        # Config update socket
        self.config_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.config_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.config_sock.setblocking(False)

        # Stats
        self.chunks_processed = 0
        self.events_detected = 0
        self.events_transmitted = 0
        self.events_rejected = 0
        self.event_log = []

    def _process_detection(self, detection: Dict) -> bool:
        """
        Process a detected event through the FFT → ML pipeline.
        Returns True if a TDoA packet was transmitted.
        """
        audio = detection["audio_segment"]
        trigger_chunk = detection.get("trigger_chunk", audio[-1024:])

        # Stage 1: FFT pre-filter
        fft_result = self.fft_filter.analyze(trigger_chunk)

        # If FFT filter is enabled and says "not in target band", allow ML to still check
        # (FFT is a pre-filter, not a hard gate — ML has the final say)
        fft_passed = fft_result["passed"]

        # Stage 2: ML classification
        ml_result = self.classifier.classify(trigger_chunk, fft_result)

        # Rescue rule for whistle-like events that ML may misclassify as animal/ambient.
        # This keeps survivor-like high-frequency signals from being dropped too early.
        peak_amp = int(detection.get("peak_amplitude", 0))
        fft_peak_hz = float(fft_result.get("peak_freq_hz", 0.0))
        ml_conf = float(ml_result.get("confidence", 0.0))
        low_amp_floor = max(120, int(self.whistle_rescue_min_peak * 0.55))
        freq_plausible_whistle = 2200.0 <= fft_peak_hz <= 4200.0
        rescue_amp_ok = (
            peak_amp >= self.whistle_rescue_min_peak
            or (peak_amp >= low_amp_floor and freq_plausible_whistle and ml_conf < 0.78)
        )

        if (
            self.whistle_rescue_enabled
            and ml_result.get("action") == "reject"
            and fft_passed
            and rescue_amp_ok
            and ml_result.get("class_name") in {"animal", "ambient", "wind", "rain"}
        ):
            ml_result = {
                "class_id": 0,
                "class_name": "whistle",
                "confidence": max(0.51, float(ml_result.get("confidence", 0.0))),
                "is_target": True,
                "action": "target",
            }
            print(
                f"🛟 [Node {self.node_id}] Whistle rescue applied "
                f"(fft_pass={fft_passed}, peak={peak_amp}, f0={fft_peak_hz:.0f}Hz)"
            )

        # Decision
        if ml_result["action"] == "target" and ml_result["is_target"]:
            # Get drifted timestamp
            ts_micros = self.clock.get_timestamp_micros(time.time())

            # Build and transmit packet
            success = self.lora_tx.send_event(
                ts_micros=ts_micros,
                magnitude=detection["peak_amplitude"],
                peak_freq_hz=fft_result.get("peak_freq_hz", 0),
                ml_class=ml_result["class_id"],
                ml_confidence=int(ml_result["confidence"] * 100),
                snr_db=detection.get("snr_db", 0),
            )

            if success:
                self.events_transmitted += 1
                self.lora_tx.drain_battery(self.battery_drain)

            # Log the event
            self.event_log.append({
                "time": time.time(),
                "action": "transmitted",
                "class": ml_result["class_name"],
                "confidence": ml_result["confidence"],
                "amplitude": detection["peak_amplitude"],
                "fft_passed": fft_passed,
            })
            return success

        elif ml_result["action"] == "log_only":
            self.event_log.append({
                "time": time.time(),
                "action": "logged",
                "class": ml_result["class_name"],
                "confidence": ml_result["confidence"],
            })
            print(f"📋 [Node {self.node_id}] Logged: {ml_result['class_name']} "
                  f"(conf={ml_result['confidence']:.0%})")
            return False
        else:
            self.events_rejected += 1
            self.event_log.append({
                "time": time.time(),
                "action": "rejected",
                "class": ml_result["class_name"],
                "confidence": ml_result["confidence"],
            })
            print(f"🚫 [Node {self.node_id}] REJECTED: {ml_result['class_name']} "
                  f"(conf={ml_result['confidence']:.0%}, action={ml_result['action']}) "
                  f"| amp={detection['peak_amplitude']} | fft_pass={fft_passed}")
            return False

    def _listen_loop(self):
        """Main loop: receive PCM chunks and process them."""
        print(f"👂 [Node {self.node_id}] Listening on port {self.listen_port}...")

        while self.running:
            try:
                data, _ = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except Exception:
                continue

            if len(data) < 8:
                continue

            # Unpack header: seq(uint32) + timestamp(float32)
            header = data[:8]
            pcm_data = data[8:]

            try:
                chunk_seq, true_ts = struct.unpack("<If", header)
            except struct.error:
                continue

            # Convert bytes to int16 PCM samples
            chunk = np.frombuffer(pcm_data, dtype=np.int16)
            if len(chunk) == 0:
                continue

            self.chunks_processed += 1

            # Process through STA/LTA detector
            detection = self.stream_proc.process_chunk(chunk)

            if detection is not None:
                self.events_detected += 1
                try:
                    self._process_detection(detection)
                except Exception as e:
                    import traceback
                    print(f"💥 [Node {self.node_id}] CRASH in process_detection: {e}")
                    traceback.print_exc()

            # Check for config updates (non-blocking)
            try:
                cfg_data, _ = self.config_sock.recvfrom(4096)
                update = json.loads(cfg_data.decode("utf-8"))
                self._apply_config_update(update)
            except (BlockingIOError, OSError):
                pass
            except Exception:
                pass

    def _apply_config_update(self, update: Dict):
        """Apply runtime configuration updates."""
        if "sta_lta_threshold" in update:
            self.stream_proc.update_config(update)
        if "fft_prefilter" in update:
            self.fft_filter.update_config(update["fft_prefilter"])
        if "ml_classifier" in update:
            self.classifier.update_config(update["ml_classifier"])
        if "clock" in update:
            self.clock.update_config(update["clock"])

    def get_status(self) -> Dict:
        """Get current node status."""
        return {
            "node_id": self.node_id,
            "running": self.running,
            "chunks_processed": self.chunks_processed,
            "events_detected": self.events_detected,
            "events_transmitted": self.events_transmitted,
            "events_rejected": self.events_rejected,
            "battery_pct": self.lora_tx.battery_pct,
            "gps_locked": self.clock.is_gps_locked(),
            "clock_drift_us": self.clock.total_drift_applied * 1e6,
        }

    def start(self):
        """Start the node process."""
        self.running = True
        try:
            self.sock.bind(("127.0.0.1", self.listen_port))
        except OSError as e:
            print(f"[Node {self.node_id}] Bind error: {e}")
            return
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print(f"✅ [Node {self.node_id}] Started on port {self.listen_port}")

    def stop(self):
        """Stop the node process."""
        self.running = False
        self.sock.close()
        self.lora_tx.close()
        print(f"🛑 [Node {self.node_id}] Stopped.")
