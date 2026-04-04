"""
ML Solver — Neural network-based position regressor.
Predicts (x, y) directly from TDoA features.
Placeholder implementation: will be functional after training.
"""

import numpy as np
import os
from typing import Dict, Optional


class MLSolver:
    """
    ML-based position regressor that directly predicts source position
    from TDoA measurements and associated features.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = None

        if model_path and os.path.exists(model_path):
            try:
                import joblib
                data = joblib.load(model_path)
                self.model = data.get("model")
                self.scaler = data.get("scaler")
                print(f"[ML Solver] Loaded model from {model_path}")
            except Exception as e:
                print(f"[ML Solver] Could not load model: {e}")

    def extract_features(self, packets: Dict[int, Dict]) -> np.ndarray:
        """
        Extract feature vector from a set of TDoA packets.
        Returns 21-dimensional feature vector.
        """
        node_ids = sorted(packets.keys())
        features = np.zeros(21, dtype=np.float64)

        if len(node_ids) < 3:
            return features

        ref_ts = packets[node_ids[0]]["ts_micros"]

        # 3x TDoA deltas (relative to first node)
        for i, nid in enumerate(node_ids[1:4]):
            if nid in packets:
                features[i] = (packets[nid]["ts_micros"] - ref_ts) / 1_000_000.0

        # 4x magnitudes
        for i, nid in enumerate(node_ids[:4]):
            if nid in packets:
                features[3 + i] = packets[nid].get("magnitude", 0) / 65535.0

        # 4x ML confidence
        for i, nid in enumerate(node_ids[:4]):
            if nid in packets:
                features[7 + i] = packets[nid].get("ml_confidence", 0) / 100.0

        # 4x SNR
        for i, nid in enumerate(node_ids[:4]):
            if nid in packets:
                features[11 + i] = packets[nid].get("snr_db", 0) / 50.0

        # 4x peak frequencies
        for i, nid in enumerate(node_ids[:4]):
            if nid in packets:
                features[15 + i] = packets[nid].get("peak_freq_hz", 0) / 8000.0

        # Number of packets
        features[19] = len(packets) / 4.0

        # Mean ML class (encoded)
        mean_class = np.mean([p.get("ml_class", 0) for p in packets.values()])
        features[20] = mean_class / 10.0

        return features

    def predict(self, packets: Dict[int, Dict]) -> Dict:
        """
        Predict source position from TDoA packets.
        Returns dict with x, y coordinates in meters (local frame).
        """
        if self.model is None:
            return {"success": False, "reason": "no_model_loaded"}

        features = self.extract_features(packets)

        try:
            if self.scaler:
                features = self.scaler.transform(features.reshape(1, -1))
            else:
                features = features.reshape(1, -1)

            prediction = self.model.predict(features)[0]
            x_est, y_est = prediction[0], prediction[1]

            return {
                "success": True,
                "x": float(x_est),
                "y": float(y_est),
                "method": "ml_regressor",
                "compute_time_ms": 0.1,
            }
        except Exception as e:
            return {"success": False, "reason": f"ml_predict_error: {e}"}
