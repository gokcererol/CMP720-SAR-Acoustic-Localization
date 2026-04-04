"""
Train ML Position Regressor — Predicts source (x, y) from TDoA features.

This runs on the Jetson solver side (not on ESP32), so it has
fewer model size constraints. Uses a small MLP or RandomForest.
"""

import os
import sys
import numpy as np
import time
import math

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def generate_tdoa_training_data(config: dict, n_samples: int = 5000) -> tuple:
    """
    Generate synthetic TDoA training data: random source positions → TDoA features.
    Returns (X_features, Y_positions).
    """
    from source.propagation import get_distance_meters

    node_positions = config.get("nodes", {}).get("positions", {})
    speed = config.get("environment", {}).get("speed_of_sound_ms", 343.0)

    if not node_positions:
        raise ValueError("No node positions in config")

    # Get node positions
    nodes = {}
    lats = []
    lons = []
    for nid, pos in node_positions.items():
        nodes[int(nid)] = pos
        lats.append(pos["lat"])
        lons.append(pos["lon"])

    # Bounding box for random sources
    lat_min, lat_max = min(lats) - 0.002, max(lats) + 0.002
    lon_min, lon_max = min(lons) - 0.002, max(lons) + 0.002

    # Reference point (centroid)
    ref_lat = sum(lats) / len(lats)
    ref_lon = sum(lons) / len(lons)

    X = []
    Y = []

    print(f"   Generating {n_samples} synthetic TDoA samples...")

    for i in range(n_samples):
        # Random source position
        src_lat = np.random.uniform(lat_min, lat_max)
        src_lon = np.random.uniform(lon_min, lon_max)

        # Compute true distances and arrival times
        node_ids = sorted(nodes.keys())
        distances = {}
        for nid in node_ids:
            d = get_distance_meters(src_lat, src_lon, nodes[nid]["lat"], nodes[nid]["lon"])
            distances[nid] = d

        # Compute arrival times
        arrival_times = {nid: d / speed for nid, d in distances.items()}

        # TDoA relative to reference (first node)
        ref_node = node_ids[0]
        ref_time = arrival_times[ref_node]

        # Add realistic noise to TDoA measurements
        jitter_sigma = np.random.uniform(0.0001, 0.005)  # 0.1ms - 5ms
        tdoa = []
        for nid in node_ids[1:]:
            dt = (arrival_times[nid] - ref_time) + np.random.normal(0, jitter_sigma)
            tdoa.append(dt)
        while len(tdoa) < 3:
            tdoa.append(0.0)

        # Random magnitudes (lower = farther)
        magnitudes = []
        for nid in node_ids:
            d = distances[nid]
            mag = max(50, int(30000 * math.sqrt(1.0 / max(d, 1.0))))
            mag += np.random.randint(-500, 500)
            magnitudes.append(max(0, min(65535, mag)) / 65535.0)
        while len(magnitudes) < 4:
            magnitudes.append(0.0)

        # Confidence (random, higher when closer)
        confidences = [min(1.0, 0.5 + 0.5 * np.random.random()) for _ in range(4)]

        # SNR
        snrs = [max(0, min(1.0, 0.3 + 0.5 * np.random.random())) for _ in range(4)]

        # Peak frequencies (simulated)
        freq_norms = [np.random.uniform(0.3, 0.5) for _ in range(4)]

        # Feature vector: 3 TDoA + 4 mag + 4 conf + 4 snr + 4 freq + n_packets + mean_class = 21
        features = np.zeros(21)
        features[0:3] = tdoa[:3]
        features[3:7] = magnitudes[:4]
        features[7:11] = confidences[:4]
        features[11:15] = snrs[:4]
        features[15:19] = freq_norms[:4]
        features[19] = np.random.choice([0.75, 1.0])  # 3 or 4 packets
        features[20] = 0.0  # mean class

        # Target: (x, y) in meters relative to centroid
        src_x = (src_lon - ref_lon) * 111320.0 * math.cos(math.radians(ref_lat))
        src_y = (src_lat - ref_lat) * 111132.0
        position = [src_x, src_y]

        X.append(features)
        Y.append(position)

    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)

    print(f"   Generated: X={X.shape}, Y={Y.shape}")
    return X, Y


def train_position_regressor(config: dict = None, output_path: str = None,
                              n_samples: int = 5000):
    """
    Train ML position regressor.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import joblib
    import yaml

    if config is None:
        config_path = os.path.join(PROJECT_ROOT, "config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "models", "position_regressor.joblib")

    print("\n" + "=" * 60)
    print("  🎯 Training ML Position Regressor")
    print("  Target: Jetson Nano (GPU available)")
    print("=" * 60)

    # Generate training data
    t_start = time.time()
    X, Y = generate_tdoa_training_data(config, n_samples=n_samples)
    gen_time = time.time() - t_start
    print(f"\n   Data generation: {gen_time:.1f}s")

    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train regressor
    print("   Training RandomForestRegressor (n=50, depth=12)...")
    t_start = time.time()
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, Y_train)
    train_time = time.time() - t_start
    print(f"   Training: {train_time:.1f}s")

    # Evaluate
    Y_pred = model.predict(X_test_s)
    mse = mean_squared_error(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)

    # Per-sample position error
    errors = np.sqrt(np.sum((Y_pred - Y_test) ** 2, axis=1))
    median_error = np.median(errors)
    p90_error = np.percentile(errors, 90)

    print(f"\n   === Results ===")
    print(f"   RMSE:          {rmse:.2f} m")
    print(f"   MAE:           {mae:.2f} m")
    print(f"   Median error:  {median_error:.2f} m")
    print(f"   90th pct:      {p90_error:.2f} m")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, output_path)
    model_size = os.path.getsize(output_path)
    print(f"\n   Model saved: {output_path}")
    print(f"   Model size:  {model_size / 1024:.1f} KB")

    # Feature importance
    importances = model.feature_importances_
    feature_names = (
        ["TDoA_12", "TDoA_13", "TDoA_14"] +
        ["Mag_1", "Mag_2", "Mag_3", "Mag_4"] +
        ["Conf_1", "Conf_2", "Conf_3", "Conf_4"] +
        ["SNR_1", "SNR_2", "SNR_3", "SNR_4"] +
        ["Freq_1", "Freq_2", "Freq_3", "Freq_4"] +
        ["N_packets", "Mean_class"]
    )
    top_features = sorted(zip(importances, feature_names), reverse=True)[:5]
    print(f"\n   Top 5 Features:")
    for imp, name in top_features:
        print(f"      {name:12s}: {imp:.3f}")

    print(f"\n{'=' * 60}\n")
    return model, scaler


if __name__ == "__main__":
    train_position_regressor()
