"""
Stream Processor — Continuous audio buffer with STA/LTA energy detection.
Processes incoming PCM chunks from the source engine and detects acoustic events.
Mimics the real ESP32 I2S DMA buffer processing loop.
"""

import numpy as np
from typing import Optional, Tuple, Dict


class StreamProcessor:
    """
    Processes a continuous stream of audio chunks using STA/LTA
    (Short-Term Average / Long-Term Average) energy detection.
    """

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024,
                 sta_window_ms: float = 50.0, lta_window_ms: float = 5000.0,
                 sta_lta_threshold: float = 5.0, cooldown_ms: float = 500.0):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        # STA/LTA parameters
        sta_samples = int(sta_window_ms / 1000.0 * sample_rate)
        lta_samples = int(lta_window_ms / 1000.0 * sample_rate)
        self.alpha_sta = 2.0 / (sta_samples + 1) if sta_samples > 0 else 0.05
        self.alpha_lta = 2.0 / (lta_samples + 1) if lta_samples > 0 else 0.0005
        self.sta_lta_threshold = sta_lta_threshold
        self.cooldown_samples = int(cooldown_ms / 1000.0 * sample_rate)

        # Running state
        self.current_sta = 0.0
        self.current_lta = 1.0  # Start non-zero to avoid division by zero
        self.samples_since_last_trigger = self.cooldown_samples  # Allow immediate first trigger
        self.total_samples_processed = 0

        # Buffer for keeping recent audio around trigger point
        self._ring_buffer = np.zeros(chunk_size * 4, dtype=np.int16)
        self._ring_pos = 0

    def process_chunk(self, chunk: np.ndarray) -> Optional[Dict]:
        """
        Process one chunk of audio (1024 samples).
        Returns detection info dict if event detected, None otherwise.

        The detection dict contains:
          - trigger_sample: global sample index where trigger occurred
          - audio_segment: the audio around the trigger (for feature extraction)
          - peak_amplitude: peak amplitude in the trigger region
          - rms_energy: RMS energy of the trigger region
          - sta_lta_ratio: the STA/LTA ratio at trigger time
        """
        if len(chunk) == 0:
            return None

        # Update ring buffer
        buf_len = len(self._ring_buffer)
        chunk_len = min(len(chunk), buf_len)
        # Shift old data and append new
        if self._ring_pos + chunk_len <= buf_len:
            self._ring_buffer[self._ring_pos:self._ring_pos + chunk_len] = chunk[:chunk_len]
            self._ring_pos += chunk_len
        else:
            # Wrap around: shift buffer left and append
            shift = chunk_len
            self._ring_buffer[:-shift] = self._ring_buffer[shift:]
            self._ring_buffer[-shift:] = chunk[:shift]
            self._ring_pos = buf_len

        detection = None
        samples = chunk.astype(np.float64)

        for i, sample in enumerate(samples):
            sample_mag = abs(sample)
            self.total_samples_processed += 1
            self.samples_since_last_trigger += 1

            # Exponential moving average for STA and LTA
            self.current_sta = self.alpha_sta * sample_mag + (1.0 - self.alpha_sta) * self.current_sta
            self.current_lta = self.alpha_lta * sample_mag + (1.0 - self.alpha_lta) * self.current_lta

            # Prevent LTA from being too small
            if self.current_lta < 1.0:
                self.current_lta = 1.0

            ratio = self.current_sta / self.current_lta

            # Check for trigger
            if ratio > self.sta_lta_threshold and self.samples_since_last_trigger >= self.cooldown_samples:
                self.samples_since_last_trigger = 0

                # Extract audio segment around trigger for feature extraction
                # Take everything in the ring buffer up to current position
                available = min(self._ring_pos, len(self._ring_buffer))
                audio_segment = self._ring_buffer[:available].copy()

                # Compute features from the trigger region
                trigger_region = chunk[max(0, i - 128):min(len(chunk), i + 896)]
                if len(trigger_region) == 0:
                    trigger_region = chunk

                peak_amp = int(np.max(np.abs(trigger_region)))
                rms = float(np.sqrt(np.mean(trigger_region.astype(np.float64) ** 2)))

                # Compute SNR estimate
                signal_power = rms ** 2
                noise_power = (self.current_lta / self.alpha_lta * self.alpha_sta) ** 2
                if noise_power > 0:
                    snr_db = int(10 * np.log10(max(signal_power / noise_power, 1e-6)))
                else:
                    snr_db = 0

                detection = {
                    "trigger_sample": self.total_samples_processed,
                    "audio_segment": audio_segment,
                    "trigger_chunk": trigger_region,
                    "peak_amplitude": peak_amp,
                    "rms_energy": rms,
                    "sta_lta_ratio": ratio,
                    "snr_db": max(0, min(255, snr_db)),
                }
                break  # One detection per chunk max

        return detection

    def reset(self):
        """Reset the processor state."""
        self.current_sta = 0.0
        self.current_lta = 1.0
        self.samples_since_last_trigger = self.cooldown_samples
        self.total_samples_processed = 0
        self._ring_buffer[:] = 0
        self._ring_pos = 0

    def update_config(self, config: Dict):
        """Update STA/LTA parameters at runtime."""
        if "sta_lta_threshold" in config:
            self.sta_lta_threshold = config["sta_lta_threshold"]
        if "cooldown_ms" in config:
            self.cooldown_samples = int(config["cooldown_ms"] / 1000.0 * self.sample_rate)
