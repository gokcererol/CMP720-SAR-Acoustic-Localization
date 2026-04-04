"""
Source Engine — Main process that generates continuous audio streams
for each node and injects sound events with propagation effects.
Runs as an independent process, communicates via UDP.
"""

import socket
import struct
import time
import json
import threading
import numpy as np
from typing import Dict, List, Optional, Any

from source.synthesizer import synthesize_sound, SAMPLE_RATE, SOUND_CLASSES
from source.propagation import propagate_to_node, inject_into_stream, get_distance_meters
from source.speaker import Speaker


class SourceEngine:
    """
    Generates continuous 16kHz PCM audio streams for each of 4 nodes.
    When events fire, synthesized waveforms are injected into each node's
    stream with correct propagation delay and attenuation.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.sample_rate = config.get("audio", {}).get("sample_rate", SAMPLE_RATE)
        self.chunk_size = config.get("audio", {}).get("chunk_size", 1024)
        self.speed_of_sound = config.get("environment", {}).get("speed_of_sound_ms", 343.0)
        self.terrain = config.get("environment", {}).get("terrain", "flat")
        self.multipath_config = config.get("environment", {}).get("multipath", {})
        self.ambient_level = config.get("environment", {}).get("ambient_noise_level", 15)

        # Node positions
        self.nodes = config.get("nodes", {}).get("positions", {})
        self.node_ports = {1: 5011, 2: 5012, 3: 5013, 4: 5014}

        # Speaker
        speaker_cfg = config.get("audio", {}).get("speaker", {})
        self.speaker = Speaker(
            sample_rate=self.sample_rate,
            enabled=speaker_cfg.get("enabled", True),
            volume=speaker_cfg.get("volume", 0.8),
            fallback_beep=speaker_cfg.get("fallback_beep", True),
            backend=speaker_cfg.get("backend", "auto"),
        )

        # UDP sockets for streaming to nodes
        self.stream_socks = {}
        for nid in self.nodes:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.stream_socks[nid] = s

        # Config update listener
        self.config_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.config_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.config_sock.setblocking(False)

        # Event injection buffers (per node)
        # Each buffer holds waveform samples to be mixed into the next stream chunks
        self._injection_buffers: Dict[int, np.ndarray] = {}
        for nid in self.nodes:
            self._injection_buffers[nid] = np.zeros(0, dtype=np.int16)

        self._lock = threading.Lock()
        self.running = False
        self.chunk_seq = 0
        self.event_log: List[Dict] = []

        # Web callback for waveform visualization
        self._web_callback = None

    def set_web_callback(self, callback):
        """Set callback for sending waveform data to web UI."""
        self._web_callback = callback

    def fire_event(self, sound_type: str, lat: float, lon: float,
                   amplitude: float = 0.8, duration: Optional[float] = None,
                   scenario_name: str = "", **synth_kwargs) -> Dict:
        """
        Fire an acoustic event at the given coordinates.
        The synthesized waveform is:
        1. Played on PC speakers
        2. Propagated to each node with delay + attenuation
        3. Injected into each node's continuous audio stream
        """
        # 1. Synthesize the source waveform
        waveform = synthesize_sound(sound_type, duration=duration,
                                    amplitude=amplitude, **synth_kwargs)

        # 2. Play on speakers
        self.speaker.play(waveform)

        # 3. Send waveform to web UI for visualization
        if self._web_callback:
            try:
                self._web_callback({
                    "type": "event_audio",
                    "sound_type": sound_type,
                    "lat": lat, "lon": lon,
                    "amplitude": amplitude,
                    "duration": len(waveform) / self.sample_rate,
                    "scenario": scenario_name,
                })
            except Exception:
                pass

        # 4. Propagate to each node
        event_info = {
            "sound_type": sound_type,
            "sound_class": SOUND_CLASSES.get(sound_type, -1),
            "lat": lat, "lon": lon,
            "amplitude": amplitude,
            "scenario": scenario_name,
            "timestamp": time.time(),
            "node_arrivals": {},
        }

        with self._lock:
            for nid, pos in self.nodes.items():
                node_lat = pos["lat"]
                node_lon = pos["lon"]

                prop = propagate_to_node(
                    waveform, lat, lon, node_lat, node_lon,
                    sample_rate=self.sample_rate,
                    speed_of_sound=self.speed_of_sound,
                    terrain=self.terrain,
                    multipath_config=self.multipath_config,
                )

                # Queue the attenuated waveform for injection
                delay = prop["delay_samples"]
                arrived_wave = prop["waveform"]

                # Create injection buffer: zeros for delay, then waveform
                total_len = delay + len(arrived_wave)
                injection = np.zeros(total_len, dtype=np.int16)
                injection[delay:delay + len(arrived_wave)] = arrived_wave

                # Add multipath reflections
                for mp_wave, mp_delay in prop["multipath"]:
                    mp_offset = delay + mp_delay
                    mp_end = min(mp_offset + len(mp_wave), total_len)
                    if mp_offset < total_len:
                        actual = mp_end - mp_offset
                        injection[mp_offset:mp_end] = np.clip(
                            injection[mp_offset:mp_end].astype(np.float64) +
                            mp_wave[:actual].astype(np.float64),
                            -32768, 32767
                        ).astype(np.int16)

                # Append to node's injection buffer
                existing = self._injection_buffers[nid]
                if len(existing) > 0:
                    # Mix with existing buffer
                    max_len = max(len(existing), len(injection))
                    merged = np.zeros(max_len, dtype=np.float64)
                    merged[:len(existing)] += existing.astype(np.float64)
                    merged[:len(injection)] += injection.astype(np.float64)
                    self._injection_buffers[nid] = np.clip(merged, -32768, 32767).astype(np.int16)
                else:
                    self._injection_buffers[nid] = injection

                event_info["node_arrivals"][nid] = {
                    "distance": prop["distance"],
                    "delay_ms": prop["delay_seconds"] * 1000,
                    "amplitude_factor": prop["amplitude_factor"],
                    "multipath_count": len(prop["multipath"]),
                }

        self.event_log.append(event_info)
        print(f"🔊 [SOURCE] Event fired: {sound_type} at ({lat:.6f}, {lon:.6f}) | {scenario_name}")
        for nid, info in event_info["node_arrivals"].items():
            print(f"   Node {nid}: {info['distance']:.1f}m, delay {info['delay_ms']:.1f}ms, "
                  f"atten {info['amplitude_factor']:.3f}")

        return event_info

    def _generate_ambient_chunk(self) -> np.ndarray:
        """Generate one chunk of ambient noise."""
        noise_level = self.ambient_level / 100.0 * 0.05  # Scale to reasonable amplitude
        chunk = (np.random.normal(0, noise_level, self.chunk_size) * 32767).astype(np.int16)
        return chunk

    def _get_next_chunk_for_node(self, node_id: int) -> np.ndarray:
        """Get the next audio chunk for a specific node (ambient + any pending injections)."""
        ambient = self._generate_ambient_chunk()

        with self._lock:
            injection = self._injection_buffers.get(node_id, np.zeros(0, dtype=np.int16))
            if len(injection) > 0:
                # Take first chunk_size samples from injection buffer
                take = min(self.chunk_size, len(injection))
                to_mix = injection[:take]
                # Remove consumed samples
                self._injection_buffers[node_id] = injection[take:]

                # Mix ambient + injection
                mixed = ambient.astype(np.float64)
                mixed[:take] += to_mix.astype(np.float64)
                return np.clip(mixed, -32768, 32767).astype(np.int16)
            else:
                return ambient

    def _stream_loop(self):
        """Main loop: continuously stream audio chunks to all nodes."""
        chunk_duration = self.chunk_size / self.sample_rate
        print(f"🎙️ [SOURCE] Streaming {self.sample_rate}Hz audio, "
              f"{self.chunk_size} samples/chunk ({chunk_duration*1000:.0f}ms)")

        while self.running:
            t_start = time.time()
            self.chunk_seq += 1

            for nid in self.nodes:
                chunk = self._get_next_chunk_for_node(nid)

                # Pack: seq(uint32) + timestamp(float32) + PCM data (1024 x int16)
                header = struct.pack("<If", self.chunk_seq, time.time())
                payload = header + chunk.tobytes()

                try:
                    self.stream_socks[nid].sendto(
                        payload, ("127.0.0.1", self.node_ports[nid])
                    )
                except Exception:
                    pass

            # Listen for config updates
            try:
                data, _ = self.config_sock.recvfrom(4096)
                update = json.loads(data.decode("utf-8"))
                self._apply_config_update(update)
            except BlockingIOError:
                pass
            except Exception:
                pass

            # Sleep to maintain real-time rate
            elapsed = time.time() - t_start
            sleep_time = chunk_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _apply_config_update(self, update: Dict):
        """Apply a runtime configuration update."""
        if "ambient_noise_level" in update:
            self.ambient_level = update["ambient_noise_level"]
        if "terrain" in update:
            self.terrain = update["terrain"]
        if "multipath" in update:
            self.multipath_config.update(update["multipath"])
        if "speed_of_sound_ms" in update:
            self.speed_of_sound = update["speed_of_sound_ms"]
        if "speaker_enabled" in update:
            self.speaker.set_enabled(update["speaker_enabled"])
        if "speaker_volume" in update:
            self.speaker.set_volume(update["speaker_volume"])

    def start(self):
        """Start the source engine streaming loop."""
        self.running = True
        try:
            self.config_sock.bind(("127.0.0.1", 5055))  # Source-specific config port
        except OSError:
            pass
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        print("✅ [SOURCE] Engine started.")

    def stop(self):
        """Stop the source engine."""
        self.running = False
        self.speaker.stop()
        for s in self.stream_socks.values():
            s.close()
        self.config_sock.close()
        print("🛑 [SOURCE] Engine stopped.")
