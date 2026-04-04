"""
Audio Synthesizer — Generates real PCM waveforms for all 11 sound classes.
Each sound type is synthesized with realistic spectral characteristics
matching what the ML classifier will encounter in the real system.
"""

import os
import random
import numpy as np
from typing import Optional, Dict, List

# Heavy libraries are deferred to prevent startup hangs on some Windows systems
_PD = None
_SF = None

def _get_pandas():
    global _PD
    if _PD is None:
        import pandas as pd
        _PD = pd
    return _PD

def _get_soundfile():
    global _SF
    if _SF is None:
        import soundfile as sf
        _SF = sf
    return _SF

SAMPLE_RATE = 16000  # Hz

# SAR 11-class Taxonomy
SOUND_CLASSES = {
    "whistle": 0,
    "human_voice": 1,
    "impact": 2,
    "knocking": 3,
    "collapse": 4,
    "machinery": 5,
    "motor": 6,
    "animal": 7,
    "wind": 8,
    "rain": 9,
    "ambient": 10,
}

# Mapping from SAR categories back to ESC-50 folder names (Exact matches for esc50.csv)
# NOTE: ESC-50 has no true human whistle class; runtime whistle is generated procedurally.
SAR_TO_ESC = {
    "whistle": [],
    "human_voice": ["crying_baby", "sneezing", "laughing", "breathing", "coughing"],
    "impact": ["clapping", "footsteps", "glass_breaking"],
    "knocking": ["door_wood_knock"],
    "collapse": ["crackling_fire", "glass_breaking"],
    "machinery": ["chainsaw", "hand_saw", "drilling", "engine"],
    "motor": ["helicopter", "airplane"],
    "animal": ["dog", "rooster", "crickets", "crow", "cat", "pig", "cow", "frog", "sheep", "hen"],
    "wind": ["wind", "thunderstorm"],
    "rain": ["rain"],
    "ambient": ["sea_waves", "water_drops", "insects"]
}

class RealSoundLoader:
    """Loads and caches authentic recordings from the ESC-50 dataset."""
    def __init__(self, dataset_root: str):
        self.root = dataset_root
        self.audio_dir = os.path.join(self.root, "audio")
        self.csv_path = os.path.join(self.root, "meta", "esc50.csv")
        self.samples: Dict[str, List[str]] = {cat: [] for cat in SAR_TO_ESC}
        self.cache: Dict[str, np.ndarray] = {}
        self._initialized = False
    def _initialize(self):
        if self._initialized: return
        if not os.path.exists(self.csv_path):
            print(f"⚠️ [SYNTH] ESC-50 Metadata not found at {self.csv_path}")
            return
            
        pd = _get_pandas()
        df = pd.read_csv(self.csv_path)
        # Create a reverse map: ESC_category -> SAR_category
        esc_to_sar = {}
        for sar, esc_list in SAR_TO_ESC.items():
            for esc in esc_list: esc_to_sar[esc] = sar
            
        for _, row in df.iterrows():
            esc_cat = row['category']
            if esc_cat in esc_to_sar:
                sar_cat = esc_to_sar[esc_cat]
                fp = os.path.join(self.audio_dir, row['filename'])
                if os.path.exists(fp):
                    self.samples[sar_cat].append(fp)
        
        self._initialized = True
        total = sum(len(v) for v in self.samples.values())
        print(f"✅ [SYNTH] Real-World Loader ready: {total} files mapped across {len(self.samples)} classes.")

    def get_sample(self, sound_type: str, amplitude: float = 0.8) -> np.ndarray:
        self._initialize()
        if sound_type not in self.samples or not self.samples[sound_type]:
            print(f"⚠️ [SYNTH] No real-world samples for {sound_type}. Falling back to silence.")
            return np.zeros(SAMPLE_RATE, dtype=np.int16)

        file_path = random.choice(self.samples[sound_type])
        
        # Load and Resample
        if file_path in self.cache:
            waveform = self.cache[file_path]
        else:
            try:
                sf = _get_soundfile()
                y, native_sr = sf.read(file_path)
                if len(y.shape) > 1: y = np.mean(y, axis=1)
                
                if native_sr != SAMPLE_RATE:
                    import soxr
                    y = soxr.resample(y, native_sr, SAMPLE_RATE)

                # Trim long leading/trailing silence so events are audible immediately.
                abs_y = np.abs(y)
                peak = float(np.max(abs_y)) if len(abs_y) else 0.0
                if peak > 0:
                    threshold = max(peak * 0.05, 1e-4)
                    idx = np.where(abs_y > threshold)[0]
                    if len(idx) > 0:
                        start = max(0, idx[0] - int(0.02 * SAMPLE_RATE))
                        end = min(len(y), idx[-1] + int(0.05 * SAMPLE_RATE))
                        y = y[start:end]

                # Keep enough context to sound natural while remaining responsive.
                max_len = int(2.5 * SAMPLE_RATE)
                if len(y) > max_len:
                    y = y[:max_len]

                # Normalize by peak, then apply a mild RMS lift for audibility
                # without flattening natural real-world dynamics.
                y = y / (np.max(np.abs(y)) + 1e-10)
                rms = float(np.sqrt(np.mean(y * y))) if len(y) else 0.0
                target_rms = 0.16 * amplitude
                if rms > 1e-6:
                    gain = min(3.0, target_rms / rms)
                    y = y * gain
                y = np.clip(y * amplitude, -0.98, 0.98)
                waveform = (y * 32767).astype(np.int16)
                
                # Simple LRU-style cache limit (keep last 50)
                if len(self.cache) > 50: self.cache.clear()
                self.cache[file_path] = waveform
            except Exception as e:
                print(f"❌ [SYNTH] Error loading {file_path}: {e}")
                return self._fallback_beep(amplitude)
        
        return waveform

    def _fallback_beep(self, amplitude: float) -> np.ndarray:
        """Generate a short synthetic chirp when a real sample cannot be loaded."""
        dur = 0.6
        t = np.linspace(0.0, dur, int(SAMPLE_RATE * dur), endpoint=False)
        chirp = np.sin(2 * np.pi * (900 + 1800 * t) * t)
        env = np.exp(-4.5 * t)
        y = np.clip(chirp * env * max(0.15, min(1.0, amplitude)), -1.0, 1.0)
        return (y * 32767).astype(np.int16)

# Singleton loader pointing to the default data location
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ESC50 = os.path.join(PROJECT_ROOT, "data", "esc50_raw", "ESC-50-master")
loader = RealSoundLoader(DEFAULT_ESC50)


def _procedural_whistle(duration: float = 0.8, amplitude: float = 0.8,
                        freq: Optional[float] = None) -> np.ndarray:
    """Generate a human-like whistle waveform (not bird chirp proxy)."""
    duration = float(max(0.2, min(2.0, duration)))
    f0 = float(freq if freq is not None else random.uniform(2400.0, 3400.0))

    t = np.linspace(0.0, duration, int(SAMPLE_RATE * duration), endpoint=False)

    # Subtle pitch modulation and slow drift make it sound human rather than synthetic.
    vibrato_hz = random.uniform(4.0, 6.5)
    vibrato_depth = random.uniform(25.0, 60.0)
    drift = np.linspace(0.0, random.uniform(-80.0, 80.0), len(t))
    f_t = f0 + vibrato_depth * np.sin(2.0 * np.pi * vibrato_hz * t) + drift

    phase = 2.0 * np.pi * np.cumsum(f_t) / SAMPLE_RATE
    base = np.sin(phase)
    # Light harmonic content to avoid pure sine tone artifacts.
    harm2 = 0.18 * np.sin(2.0 * phase + 0.2)
    harm3 = 0.07 * np.sin(3.0 * phase + 1.1)

    # Breath noise and smooth attack/release envelope.
    noise = 0.02 * np.random.normal(0.0, 1.0, len(t))
    attack = int(0.05 * len(t))
    release = int(0.10 * len(t))
    sustain = max(0, len(t) - attack - release)
    env = np.concatenate([
        np.linspace(0.0, 1.0, attack, endpoint=False),
        np.ones(sustain),
        np.linspace(1.0, 0.0, release, endpoint=True),
    ])
    if len(env) < len(t):
        env = np.pad(env, (0, len(t) - len(env)), mode="edge")
    else:
        env = env[:len(t)]

    y = (base + harm2 + harm3 + noise) * env
    y = y / (np.max(np.abs(y)) + 1e-10)
    y = np.clip(y * float(max(0.05, min(1.0, amplitude))) * 0.95, -1.0, 1.0)
    return (y * 32767).astype(np.int16)

def synthesize_sound(sound_type: str, duration: Optional[float] = None,
                      amplitude: float = 0.8, **kwargs) -> np.ndarray:
    """
    Returns an authentic ESC-50 recording for the requested class.
    Synthetization is deprecated and replaced by real-world data.
    """
    # ESC-50 does not have true human whistle samples; use procedural whistle.
    if sound_type == "whistle":
        return _procedural_whistle(
            duration=duration if duration is not None else kwargs.get("duration", 0.8),
            amplitude=amplitude,
            freq=kwargs.get("freq", None),
        )

    return loader.get_sample(sound_type, amplitude=amplitude)

CLASS_NAMES = {v: k for k, v in SOUND_CLASSES.items()}
TARGET_CLASSES = {"whistle", "human_voice", "impact", "knocking"}
LOG_ONLY_CLASSES = {"collapse"}
REJECT_CLASSES = {"machinery", "motor", "animal", "wind", "rain", "ambient"}
