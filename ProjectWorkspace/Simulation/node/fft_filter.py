"""
FFT Pre-Filter — Optional band-pass filter for the whistle frequency range.
When enabled, acts as a gate before the ML classifier to save computation.
"""

import numpy as np
from typing import Dict, Optional, Tuple


class FFTFilter:
    """
    Performs FFT on audio chunks and checks if the dominant frequency
    falls within the target band (default 2500-4000 Hz for whistles).
    """

    def __init__(self, sample_rate: int = 16000, fft_size: int = 1024,
                 freq_min: int = 2500, freq_max: int = 4000,
                 enabled: bool = True):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.enabled = enabled

        # Pre-compute bin indices
        self.bin_width = sample_rate / fft_size
        self.start_bin = max(1, int(freq_min / self.bin_width))
        self.end_bin = min(fft_size // 2 - 1, int(freq_max / self.bin_width))

        # Hamming window (matches ESP32 firmware FFT_WIN_TYP_HAMMING)
        self.window = np.hamming(fft_size)

    def analyze(self, audio_chunk: np.ndarray) -> Dict:
        """
        Perform FFT analysis on an audio chunk.
        Returns dict with:
          - peak_freq_hz: dominant frequency in Hz
          - peak_magnitude: magnitude at peak bin
          - in_target_band: whether peak is in whistle band
          - passed: whether the filter passes this chunk
          - spectrum: full magnitude spectrum (for visualization)
        """
        # Take last fft_size samples if chunk is larger
        if len(audio_chunk) >= self.fft_size:
            data = audio_chunk[-self.fft_size:].astype(np.float64)
        else:
            data = np.zeros(self.fft_size, dtype=np.float64)
            data[:len(audio_chunk)] = audio_chunk.astype(np.float64)

        # Remove DC offset
        data -= np.mean(data)

        # Apply Hamming window
        windowed = data * self.window

        # FFT
        fft_result = np.fft.rfft(windowed)
        magnitudes = np.abs(fft_result)

        # Find peak in target band
        band_mags = magnitudes[self.start_bin:self.end_bin + 1]
        if len(band_mags) == 0:
            return {
                "peak_freq_hz": 0, "peak_magnitude": 0,
                "in_target_band": False, "passed": not self.enabled,
                "spectrum": magnitudes,
            }

        peak_bin_offset = np.argmax(band_mags)
        peak_bin = self.start_bin + peak_bin_offset
        peak_freq = peak_bin * self.bin_width
        peak_mag = float(band_mags[peak_bin_offset])

        # Also find global peak
        global_peak_bin = np.argmax(magnitudes[1:]) + 1  # skip DC
        global_peak_freq = global_peak_bin * self.bin_width
        global_peak_mag = float(magnitudes[global_peak_bin])

        # Is the overall dominant frequency in our target band?
        in_band = self.freq_min <= global_peak_freq <= self.freq_max

        # Filter passes if disabled OR if dominant frequency is in band
        passed = (not self.enabled) or in_band

        return {
            "peak_freq_hz": int(peak_freq),
            "peak_magnitude": peak_mag,
            "global_peak_freq_hz": int(global_peak_freq),
            "global_peak_magnitude": global_peak_mag,
            "in_target_band": in_band,
            "passed": passed,
            "spectrum": magnitudes,
        }

    def update_config(self, config: Dict):
        """Update filter parameters at runtime."""
        if "enabled" in config:
            self.enabled = config["enabled"]
        if "whistle_freq_min" in config:
            self.freq_min = config["whistle_freq_min"]
            self.start_bin = max(1, int(self.freq_min / self.bin_width))
        if "whistle_freq_max" in config:
            self.freq_max = config["whistle_freq_max"]
            self.end_bin = min(self.fft_size // 2 - 1, int(self.freq_max / self.bin_width))
