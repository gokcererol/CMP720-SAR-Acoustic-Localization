"""
TinyML Sound Classifier — 11-class sound classification for edge detection.
Uses audio features (MFCC, spectral) to classify sounds at the node level.
If no trained model is available, uses a rule-based fallback.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import json

# Sound class definitions
CLASS_NAMES = {
    0: "whistle", 1: "human_voice", 2: "impact", 3: "knocking",
    4: "collapse", 5: "machinery", 6: "motor", 7: "animal",
    8: "wind", 9: "rain", 10: "ambient",
}
NUM_CLASSES = 11
TARGET_CLASSES = {0, 1, 2, 3}     # whistle, human_voice, impact, knocking
LOG_ONLY_CLASSES = {4}             # collapse
REJECT_CLASSES = {5, 6, 7, 8, 9, 10}


def extract_features(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Extract high-precision audio features for ML classification.
    Returns a feature vector of 35 values:
    [0-12]   MFCCs (Window Average)
    [13]     RMS Energy
    [14]     Zero-Crossing Rate
    [15]     Spectral Centroid
    [16]     Spectral Bandwidth
    [17]     Spectral Rolloff (85%)
    [18]     Spectral Flatness
    [19-31]  Delta-MFCCs (Temporal Spectral Velocity)
    [32]     Crest Factor (Peak / RMS)
    [33]     Envelope Variance (Sub-window energy)
    [34]     Spectral Skewness
    """
    audio_float = audio.astype(np.float64)
    if len(audio_float) < 128:
        return np.zeros(35, dtype=np.float64)

# Global Cache for Mel Filterbanks to avoid redundant calculations
_MEL_FILTERBANK_CACHE = {}

def get_mel_filters(n_mels, n_fft, sample_rate):
    """Pre-calculate Mel filterbank bins and filters."""
    key = (n_mels, n_fft, sample_rate)
    if key in _MEL_FILTERBANK_CACHE:
        return _MEL_FILTERBANK_CACHE[key]
    
    mel_min = 0
    mel_max = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    
    # Pre-calculate triangular filters
    filters = []
    for m in range(n_mels):
        f_start, f_center, f_end = bin_points[m], bin_points[m+1], bin_points[m+2]
        filt = np.zeros(n_fft // 2 + 1)
        if f_center > f_start:
            filt[f_start:f_center] = np.linspace(0, 1, f_center - f_start)
        if f_end > f_center:
            filt[f_center:f_end] = np.linspace(1, 0, f_end - f_center)
        filters.append(filt)
    
    filters = np.array(filters)
    _MEL_FILTERBANK_CACHE[key] = (bin_points, filters)
    return bin_points, filters

def get_segment_mfcc(segment, sample_rate, n_mels=13):
    """Optimized MFCC extraction using pre-calculated filters."""
    if len(segment) < 16: return np.zeros(n_mels)
    n_fft = 512 # Standard window for chunking
    windowed = segment[:n_fft] * np.hamming(len(segment[:n_fft])) # Handle short segments safely
    spec = np.abs(np.fft.rfft(windowed, n=n_fft))
    
    _, filters = get_mel_filters(n_mels, n_fft, sample_rate)
    
    # Vectorized application of filters
    # spec is (257,), filters is (13, 257)
    mel_spec = np.dot(filters, spec)
    return np.log(mel_spec + 1e-10)

def extract_features(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Extract high-precision audio features for ML classification.
    Returns a feature vector of 35 values.
    Optimized for training and high-frequency edge execution.
    """
    audio_float = audio.astype(np.float64)
    if len(audio_float) < 128:
        return np.zeros(35, dtype=np.float64)

    # 1. Global Spectral Analysis
    n_fft = min(1024, len(audio_float))
    windowed = audio_float[:n_fft] * np.hamming(n_fft)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    
    spec_sum = np.sum(spectrum)
    if spec_sum > 0:
        centroid = np.sum(freqs * spectrum) / spec_sum
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / spec_sum)
        # Spectral Flatness: Geometric / Arithmetic mean
        geom_mean = np.exp(np.mean(np.log(spectrum + 1e-10)))
        arith_mean = np.mean(spectrum)
        flatness = geom_mean / (arith_mean + 1e-10)
        # Spectral Rolloff
        cumsum = np.cumsum(spectrum)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * spec_sum)
        rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]
        # Spectral Skewness
        m3 = np.sum(((freqs - centroid) ** 3) * spectrum) / spec_sum
        skewness = m3 / (bandwidth ** 3 + 1e-10)
    else:
        centroid = bandwidth = flatness = rolloff = skewness = 0.0

    # 2. Temporal/Energy features (Fast)
    rms = np.sqrt(np.mean(audio_float ** 2))
    zcr = np.sum(np.abs(np.diff(np.sign(audio_float)))) / (2 * len(audio_float))
    
    # Crest Factor (Peak to RMS)
    peak = np.max(np.abs(audio_float))
    crest_factor = peak / (rms + 1e-10)
    
    # Envelope Variance (Sub-window energy)
    n_sub = 4
    sub_energy = []
    sub_size = len(audio_float) // n_sub
    for i in range(n_sub):
        sub = audio_float[i*sub_size : (i+1)*sub_size]
        if len(sub) > 0: sub_energy.append(np.sqrt(np.mean(sub**2)))
    env_variance = np.var(sub_energy) / (rms**2 + 1e-10) if sub_energy else 0.0

    # 3. MFCC and Deltas (Using Optimized functions)
    half = len(audio_float) // 2
    mfcc_h1 = get_segment_mfcc(audio_float[:half], sample_rate)
    mfcc_h2 = get_segment_mfcc(audio_float[half:], sample_rate)
    
    mfcc_avg = (mfcc_h1 + mfcc_h2) / 2
    mfcc_delta = mfcc_h2 - mfcc_h1

    # 4. Assemble 35-dimensional vector
    features = np.zeros(35, dtype=np.float64)
    features[:13] = mfcc_avg
    features[13] = rms / 32768.0
    features[14] = zcr
    features[15] = centroid / (sample_rate / 2)
    features[16] = bandwidth / (sample_rate / 2)
    features[17] = rolloff / (sample_rate / 2)
    features[18] = flatness
    features[19:32] = mfcc_delta
    features[32] = crest_factor / 10.0 # Normalize 
    features[33] = env_variance
    features[34] = skewness / 5.0 # Normalize

    return features


class SoundClassifier:
    """
    11-class sound classifier. Uses a trained model if available,
    otherwise falls back to rule-based classification.
    """

    def __init__(self, model_path: Optional[str] = None,
                 confidence_threshold: float = 0.7,
                 target_classes: List[str] = None):
        self.confidence_threshold = confidence_threshold
        self.target_classes = set(target_classes or ["whistle", "human_voice", "impact", "knocking"])
        self.model = None
        self.use_rules = True
        self.hybrid_rescue_enabled = True

        # Try to load trained model
        if model_path and os.path.exists(model_path):
            try:
                print(f"🔍 [3/5] Loading ML Model: {os.path.basename(model_path)}...", flush=True)
                import joblib
                self.model = joblib.load(model_path)
                self.use_rules = False
                print(f"✅ ML Model Loaded.", flush=True)
            except Exception as e:
                print(f"⚠️ [ML] Could not load model: {e}. Using rule-based fallback.", flush=True)

    def classify(self, audio: np.ndarray, fft_result: Optional[Dict] = None,
                 sample_rate: int = 16000) -> Dict:
        """
        Classify a sound from its audio waveform.
        Returns dict with:
          - class_id: int (0-10)
          - class_name: str
          - confidence: float (0-1)
          - is_target: bool
          - action: str ("target", "log_only", "reject")
        """
        features = extract_features(audio, sample_rate)

        if self.model is not None and not self.use_rules:
            return self._classify_ml(features, fft_result)
        else:
            return self._classify_rules(features, fft_result)

    def _classify_ml(self, features: np.ndarray, fft_result: Optional[Dict]) -> Dict:
        """Classify using the trained ML bundle (Supports Sklearn and Pure NumPy)."""
        try:
            if isinstance(self.model, dict):
                if "weights" in self.model:
                    # Pure NumPy Rescue Bundle
                    x = (features - self.model["mean"]) / self.model["std"]
                    x = x.reshape(1, -1)
                    
                    # Forward Pass
                    activations = [x]
                    weights = self.model["weights"]
                    biases = self.model["biases"]
                    
                    for i in range(len(weights)-1):
                        z = np.dot(activations[-1], weights[i]) + biases[i]
                        activations.append(np.maximum(0, z)) # ReLU
                    
                    final_z = np.dot(activations[-1], weights[-1]) + biases[-1]
                    # Softmax logic for probability
                    exps = np.exp(final_z - np.max(final_z))
                    proba = (exps / np.sum(exps))[0]
                else:
                    # Sklearn Bundle format: {scaler, model, features...}
                    feat_scaled = self.model["scaler"].transform(features.reshape(1, -1))
                    proba = self.model["model"].predict_proba(feat_scaled)[0]
            else:
                # Fallback for raw model
                proba = self.model.predict_proba(features.reshape(1, -1))[0]

            class_id = int(np.argmax(proba))
            confidence = float(proba[class_id])
        except Exception as e:
            print(f"⚠️ [ML Classifier] Prediction failed: {e}")
            return self._classify_rules(features, None)

        class_name = CLASS_NAMES.get(class_id, "unknown")
        rules = self._classify_rules(features, fft_result)

        # Hybrid rescue: when ML is uncertain or predicts non-target, consult rules.
        # This reduces false rejections in noisy field audio where tiny models are brittle.
        if self.hybrid_rescue_enabled:
            ml_is_uncertain = confidence < max(0.42, self.confidence_threshold)
            ml_is_reject = class_id in REJECT_CLASSES
            rules_id = int(rules.get("class_id", -1))
            fft_pass = bool((fft_result or {}).get("passed", False))
            # Conservative rescue: allow whistle/impact/knocking only.
            # Avoid automatic human_voice rescue which can absorb machinery-like sounds.
            rescue_allowed = (
                (rules_id == 0 and fft_pass)
                or (rules_id == 2 and float(rules.get("confidence", 0.0)) >= 0.80)
                or (rules_id == 3 and float(rules.get("confidence", 0.0)) >= 0.75)
            )
            if rescue_allowed and ml_is_reject and ml_is_uncertain:
                boosted = dict(rules)
                boosted["confidence"] = max(float(rules.get("confidence", 0.0)), confidence)
                return boosted

        # Post-ML disambiguation for frequent confusion pair: impact vs knocking.
        # impacts are typically more impulsive (higher crest/env variance and RMS).
        rms = features[13] * 32768.0
        crest = features[32] * 10.0
        env_var = features[33]

        # Strong impulse override: prioritize likely impact over reject/non-target labels.
        if class_id in REJECT_CLASSES and int(rules.get("class_id", -1)) == 2:
            if float(rules.get("confidence", 0.0)) >= 0.85 and rms > 4000 and crest > 6.2 and env_var > 0.30:
                class_id = 2
                class_name = CLASS_NAMES[class_id]
                confidence = max(confidence, float(rules.get("confidence", 0.0)))

        if class_id == 3 and rms > 3000 and crest > 6.0 and env_var > 0.28:
            class_id = 2
            class_name = CLASS_NAMES[class_id]
            confidence = max(confidence, 0.72)
        elif class_id == 2 and crest < 4.2 and env_var < 0.18 and rms < 9000:
            class_id = 3
            class_name = CLASS_NAMES[class_id]
            confidence = max(confidence, 0.60)

        if class_id in TARGET_CLASSES:
            # Respect confidence threshold for target classes.
            # If uncertain, only keep target if rule engine independently agrees.
            if confidence < self.confidence_threshold:
                if rules.get("is_target", False):
                    promoted = dict(rules)
                    promoted["confidence"] = max(float(rules.get("confidence", 0.0)), confidence)
                    return promoted
                return {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "is_target": False,
                    "action": "reject",
                }
            # SAR priority: always forward target classes to solver.
            # The solver has a 9-stage filter pipeline for FP rejection.
            # Missing a survivor (false negative) is far worse than a false positive.
            action = "target"
            is_target = True
        elif class_id in LOG_ONLY_CLASSES:
            action = "log_only"
            is_target = False
        else:
            action = "reject"
            is_target = False

        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "is_target": is_target,
            "action": action,
        }

    def _classify_rules(self, features: np.ndarray, fft_result: Optional[Dict]) -> Dict:
        """
        Rule-based fallback classification when no ML model is available.
        """
        # features[13] is normalized RMS. Multiply by 32768 to get back to scale.
        rms = features[13] * 32768.0
        zcr = features[14]
        centroid_norm = features[15]
        centroid_hz = centroid_norm * 8000
        flatness = features[18]

        # Rule-based heuristics
        confidence = 0.5
        class_id = 10  # default: ambient

        if rms < 30:
            class_id = 10  # ambient
            confidence = 0.90
        elif 2500 <= centroid_hz <= 4000 and flatness < 0.1:
            class_id = 0  # whistle
            confidence = 0.85
        elif 800 <= centroid_hz <= 4000:
            if centroid_hz > 1500:
                class_id = 1  # human_voice
                confidence = 0.70
            else:
                class_id = 3  # knocking
                confidence = 0.65
        elif centroid_hz < 300:
            if rms > 5000:
                class_id = 5  # machinery
                confidence = 0.60
            else:
                class_id = 6  # motor
                confidence = 0.60
        else:
            class_id = 8  # wind
            confidence = 0.50

        # High RMS + Peakiness + Variance = impact
        # Normal Crest Factor for white noise is ~3-4. Impacts should be > 6.
        # Normal Env Variance for steady noise is < 0.1. Impacts should be > 0.5.
        # If it has machinery-like spectral traits (low freq), use stricter thresholds.
        is_machine_context = (centroid_hz < 600 and flatness < 0.2)
        thresh_crest = 7.5 if is_machine_context else 5.5
        thresh_var = 0.5 if is_machine_context else 0.3

        if rms > 5000 and features[32]*10 > thresh_crest and features[33] > thresh_var:
            class_id = 2  # impact
            confidence = 0.85
        elif rms > 15000 and zcr > 0.3 and not is_machine_context:
            # Emergency fallback for extremely loud broadband (might still be impact)
            if features[33] > 0.2:
                class_id = 2
                confidence = 0.70

        if class_id in TARGET_CLASSES:
            action = "target"
            is_target = True
        elif class_id in LOG_ONLY_CLASSES:
            action = "log_only"
            is_target = False
        else:
            action = "reject"
            is_target = False

        class_name = CLASS_NAMES.get(class_id, "unknown")

        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "is_target": is_target,
            "action": action,
        }

    def update_config(self, config: Dict):
        """Update classifier parameters at runtime."""
        if "confidence_threshold" in config:
            self.confidence_threshold = config["confidence_threshold"]
        if "target_classes" in config:
            self.target_classes = set(config["target_classes"])
        if "hybrid_rescue_enabled" in config:
            self.hybrid_rescue_enabled = bool(config["hybrid_rescue_enabled"])
