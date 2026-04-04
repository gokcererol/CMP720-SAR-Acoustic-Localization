"""
Speaker Playback — Plays synthesized event audio through PC speakers.
Non-blocking: audio plays in a separate thread.
"""

import threading
import os
import io
import wave
import numpy as np
from typing import Optional

# sounddevice import is deferred to prevent startup hangs on Windows
_SD_INSTANCE = None

def _get_sounddevice():
    global _SD_INSTANCE
    if _SD_INSTANCE is None:
        try:
            import sounddevice as sd
            _SD_INSTANCE = sd
        except ImportError:
            _SD_INSTANCE = False
            print("[WARNING] sounddevice not installed. Speaker playback disabled.")
        except Exception as e:
            _SD_INSTANCE = False
            print(f"[WARNING] sounddevice error: {e}. Speaker playback disabled.")
    return _SD_INSTANCE


class Speaker:
    """Manages non-blocking audio playback through PC speakers."""

    def __init__(self, sample_rate: int = 16000, enabled: bool = True,
                 volume: float = 0.8, fallback_beep: bool = True,
                 backend: str = "auto"):
        self.sample_rate = sample_rate
        self.user_enabled = enabled
        self.volume = max(0.0, min(1.0, volume))
        self.fallback_beep = fallback_beep
        self.backend = (backend or "auto").lower()
        self.is_playing = False
        self._lock = threading.Lock()
        self._backend_logged = False

    @property
    def enabled(self):
        """User-level speaker switch.

        Backend availability is resolved lazily in playback thread to avoid
        blocking the main request/event path on Windows audio init.
        """
        return self.user_enabled

    def play(self, waveform: np.ndarray, blocking: bool = False):
        """Play a waveform through speakers. Non-blocking by default."""
        if not self.user_enabled:
            self._play_fallback_beep()
            return
        with self._lock:
            if self.is_playing and not blocking:
                return  # Don't overlap
            self.is_playing = True

        def _play_thread():
            ok = False
            try:
                if self.backend == "winsound":
                    ok = self._play_with_winsound(waveform)
                elif self.backend == "sounddevice":
                    ok = self._play_with_sounddevice(waveform)
                else:
                    # auto: prefer winsound on Windows for predictable default-device output
                    if os.name == "nt":
                        ok = self._play_with_winsound(waveform)
                        if not ok:
                            ok = self._play_with_sounddevice(waveform)
                    else:
                        ok = self._play_with_sounddevice(waveform)

                if not ok:
                    self._play_fallback_beep(waveform)
            finally:
                with self._lock:
                    self.is_playing = False

        if blocking:
            _play_thread()
        else:
            t = threading.Thread(target=_play_thread, daemon=True)
            t.start()

    def stop(self):
        """Stop any currently playing audio."""
        global _SD_INSTANCE
        sd = _SD_INSTANCE if _SD_INSTANCE not in (None, False) else None
        if sd is not None:
            try:
                sd.stop()
            except Exception:
                pass
        with self._lock:
            self.is_playing = False

    def set_volume(self, volume: float):
        """Set playback volume (0.0 - 1.0)."""
        self.volume = max(0.0, min(1.0, volume))

    def set_enabled(self, enabled: bool):
        """Enable or disable speaker."""
        self.user_enabled = enabled
        if not self.user_enabled:
            self.stop()

    def _play_fallback_beep(self, waveform: Optional[np.ndarray] = None):
        """Best-effort Windows fallback via winsound PlaySound on default output device."""
        if not self.fallback_beep or os.name != "nt":
            return
        try:
            import winsound
            if waveform is None or len(waveform) == 0:
                t = np.linspace(0.0, 0.25, int(self.sample_rate * 0.25), endpoint=False)
                tone = (0.35 * np.sin(2.0 * np.pi * 1200.0 * t) * 32767.0).astype(np.int16)
                waveform = tone

            pcm = np.clip(waveform.astype(np.float32) * self.volume, -32768, 32767).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(pcm.tobytes())

            winsound.PlaySound(buf.getvalue(), winsound.SND_MEMORY)
            print("[Speaker] Using winsound WAV fallback.")
        except Exception as e:
            print(f"[Speaker] Fallback playback failed: {e}")

    def _play_with_sounddevice(self, waveform: np.ndarray) -> bool:
        sd = _get_sounddevice()
        if not sd:
            return False
        try:
            audio = waveform.astype(np.float32) / 32768.0 * self.volume
            if not self._backend_logged:
                try:
                    default_dev = sd.default.device
                    print(f"[Speaker] Backend=sounddevice default_device={default_dev}")
                except Exception:
                    print("[Speaker] Backend=sounddevice")
                self._backend_logged = True
            sd.play(audio, samplerate=self.sample_rate)
            sd.wait()
            return True
        except Exception as e:
            print(f"[Speaker] sounddevice playback error: {e}")
            return False

    def _play_with_winsound(self, waveform: np.ndarray) -> bool:
        if os.name != "nt":
            return False
        try:
            import winsound
            pcm = np.clip(waveform.astype(np.float32) * self.volume, -32768, 32767).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(pcm.tobytes())

            if not self._backend_logged:
                print("[Speaker] Backend=winsound")
                self._backend_logged = True
            winsound.PlaySound(buf.getvalue(), winsound.SND_MEMORY)
            return True
        except Exception as e:
            print(f"[Speaker] winsound playback error: {e}")
            return False
