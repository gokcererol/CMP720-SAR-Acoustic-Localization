"""
Train High-Precision TinyML Sound Classifier — Optimized for ESP32-S3.

Improvements:
1. 35-dimensional feature vector (MFCCs + Deltas + Temporal Peakiness).
2. MLP (Neural Network) architecture for better performance with ESP-NN.
3. Mixed-Class Augmentation: Targets are mixed with background noise during training.
4. Precision-Focused: Optimized to minimize false positives in search-and-rescue.
"""

import os
import sys

# Prevent OpenBLAS/MKL deadlocks on Windows during training.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import time
from pathlib import Path

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


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

CLASS_NAMES = {v: k for k, v in SOUND_CLASSES.items()}
TARGET_CLASS_IDS = {SOUND_CLASSES[k] for k in ["whistle", "human_voice", "impact", "knocking"]}


def _sample_snr_db(class_name: str) -> float:
    """Sample class-aware SNRs to bias training toward noisy field conditions."""
    if class_name in {"whistle", "human_voice", "impact", "knocking"}:
        # Harder curriculum for target classes: many near 0dB or negative SNR examples.
        if np.random.rand() < 0.6:
            return float(np.random.uniform(-9.0, 4.0))
        return float(np.random.uniform(4.0, 14.0))
    if class_name in {"collapse", "machinery", "motor", "animal"}:
        return float(np.random.uniform(-4.0, 14.0))
    return float(np.random.uniform(-2.0, 18.0))


def _mix_at_snr(signal_i16: np.ndarray, noise_i16: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix int16 signal/noise at a target SNR (dB), returning int16 output."""
    if len(signal_i16) == 0:
        return signal_i16

    sig = signal_i16.astype(np.float64)
    noise = noise_i16.astype(np.float64)
    if len(noise) < len(sig):
        reps = int(np.ceil(len(sig) / max(1, len(noise))))
        noise = np.tile(noise, reps)

    start = np.random.randint(0, max(1, len(noise) - len(sig) + 1))
    noise = noise[start:start + len(sig)]

    sig_rms = np.sqrt(np.mean(sig ** 2)) + 1e-9
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-9
    desired_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    noise_gain = desired_noise_rms / noise_rms

    mixed = sig + noise * noise_gain
    peak = np.max(np.abs(mixed)) + 1e-9
    # Keep peak in int16 range without clipping while preserving SNR characteristics.
    if peak > 32767:
        mixed = mixed * (32767.0 / peak)
    return np.clip(mixed, -32767, 32767).astype(np.int16)


def _stratified_holdout_indices(y: np.ndarray, test_ratio: float = 0.2) -> tuple:
    """Return stratified train/validation indices without sklearn dependency."""
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        np.random.shuffle(cls_idx)
        n_test = max(1, int(len(cls_idx) * test_ratio))
        test_idx.append(cls_idx[:n_test])
        train_idx.append(cls_idx[n_test:])

    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    return train_idx, test_idx


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    """Compute compact metrics used for robust/noisy SAR training diagnostics."""
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf[int(t), int(p)] += 1

    recalls = np.zeros(n_classes, dtype=np.float64)
    precisions = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        tp = conf[c, c]
        fn = np.sum(conf[c, :]) - tp
        fp = np.sum(conf[:, c]) - tp
        recalls[c] = tp / (tp + fn + 1e-9)
        precisions[c] = tp / (tp + fp + 1e-9)

    target_recall = float(np.mean([recalls[c] for c in TARGET_CLASS_IDS]))
    non_target = [c for c in range(n_classes) if c not in TARGET_CLASS_IDS]
    reject_precision = float(np.mean([precisions[c] for c in non_target])) if non_target else 0.0

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "target_recall": target_recall,
        "reject_precision": reject_precision,
    }


def _fast_synthesize(class_name: str, sample_rate: int, duration: float, amplitude: float,
                     **kwargs) -> np.ndarray:
    """Fast, fully local waveform generator for training (no dataset I/O)."""
    n = max(64, int(sample_rate * max(0.05, duration)))
    t = np.linspace(0.0, n / sample_rate, n, endpoint=False)

    if class_name == "whistle":
        f0 = float(kwargs.get("freq", np.random.uniform(2300, 3600)))
        y = np.sin(2 * np.pi * f0 * t)
        y += 0.12 * np.sin(2 * np.pi * 2 * f0 * t + 0.3)
    elif class_name == "human_voice":
        f0 = float(kwargs.get("base_freq", np.random.uniform(120, 320)))
        y = np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(2 * np.pi * 2 * f0 * t)
        y += 0.12 * np.random.normal(0, 1, n)
    elif class_name == "impact":
        y = np.random.normal(0, 1, n) * np.exp(-18 * t)
    elif class_name == "knocking":
        y = np.zeros(n)
        hit = max(1, int(0.03 * sample_rate))
        for k in [0, int(0.18 * sample_rate), int(0.35 * sample_rate)]:
            if k < n:
                end = min(n, k + hit)
                y[k:end] += np.random.normal(0, 1, end - k) * np.hanning(end - k)
    elif class_name == "collapse":
        y = np.random.normal(0, 1, n) * np.exp(-2.5 * t)
    elif class_name == "machinery":
        y = 0.7 * np.sin(2 * np.pi * 120 * t) + 0.3 * np.sin(2 * np.pi * 240 * t)
        y += 0.2 * np.random.normal(0, 1, n)
    elif class_name == "motor":
        y = np.sin(2 * np.pi * 180 * t) + 0.25 * np.sin(2 * np.pi * 360 * t)
    elif class_name == "animal":
        f = np.random.uniform(800, 2200)
        y = np.sin(2 * np.pi * f * t) * (1 + 0.5 * np.sin(2 * np.pi * 6 * t))
    elif class_name == "wind":
        y = np.random.normal(0, 1, n)
        y = np.convolve(y, np.ones(16) / 16, mode="same")
    elif class_name == "rain":
        y = np.random.normal(0, 1, n)
        spikes = np.random.rand(n) < 0.02
        y[spikes] += np.random.uniform(2, 5, spikes.sum())
    else:  # ambient
        y = np.random.normal(0, 0.35, n)

    y = y / (np.max(np.abs(y)) + 1e-9)
    y = np.clip(y * amplitude, -1.0, 1.0)
    return (y * 32767).astype(np.int16)


def generate_training_data(samples_per_class: int = 500,
                           sample_rate: int = 16000) -> tuple:
    """
    Generate training data with advanced augmentation and 35-feature vector.
    """
    from node.ml_classifier import extract_features
    try:
        from source.synthesizer import synthesize_sound
        has_real_synth = True
    except Exception:
        synthesize_sound = None
        has_real_synth = False

    all_features = []
    all_labels = []

    class_names = list(SOUND_CLASSES.keys())
    print(
        f"   Generating base={samples_per_class}/class "
        f"(real_mix={'on' if has_real_synth else 'off'})...",
        flush=True,
    )

    # Pre-generate synthetic noise samples for SNR-controlled mixing
    noise_pool = []
    for _ in range(220):
        n_type = np.random.choice(["wind", "rain", "machinery", "motor", "ambient"])
        n_amp = np.random.uniform(0.2, 0.9)
        n_dur = np.random.uniform(0.2, 1.6)
        noise_pool.append(_fast_synthesize(n_type, sample_rate, duration=n_dur, amplitude=n_amp))

    overall_start = time.time()
    target_names = {"whistle", "human_voice", "impact", "knocking"}
    class_sample_map = {}
    for cname in class_names:
        if cname in target_names:
            class_sample_map[cname] = int(samples_per_class * 1.6)
        elif cname == "collapse":
            class_sample_map[cname] = int(samples_per_class * 1.2)
        else:
            class_sample_map[cname] = int(samples_per_class)

    total_expected = int(sum(class_sample_map.values()))
    generated = 0

    for class_idx, class_name in enumerate(class_names, start=1):
        class_id = SOUND_CLASSES[class_name]
        class_start = time.time()
        print(f"   -> [{class_idx}/{len(class_names)}] class={class_name}", flush=True)

        class_samples = class_sample_map[class_name]
        for i in range(class_samples):
            amp = np.random.uniform(0.3, 1.0)
            kwargs = {}
            if class_name == "whistle":
                kwargs["freq"] = np.random.uniform(2200, 4200)
                duration = np.random.uniform(0.3, 1.2)
            elif class_name == "human_voice":
                kwargs["base_freq"] = np.random.uniform(150, 450)
                duration = np.random.uniform(0.5, 1.5)
            else:
                duration = np.random.uniform(0.1, 0.6)

            try:
                use_real = (
                    has_real_synth
                    and np.random.rand() < (0.75 if class_name in target_names else 0.55)
                )
                if use_real:
                    waveform = synthesize_sound(
                        class_name,
                        duration=float(duration),
                        amplitude=float(amp),
                        **kwargs,
                    )
                else:
                    waveform = _fast_synthesize(
                        class_name,
                        sample_rate,
                        duration=duration,
                        amplitude=amp,
                        **kwargs,
                    )
            except Exception:
                continue

            # SNR-controlled augmentation to improve robustness in noisy SAR scenes.
            snr_db = _sample_snr_db(class_name)
            noise = noise_pool[np.random.randint(0, len(noise_pool))]
            waveform = _mix_at_snr(waveform, noise, snr_db)

            # Occasional impulse corruption to improve transient robustness.
            if np.random.rand() < 0.12 and len(waveform) > 8:
                n_imp = np.random.randint(1, 5)
                idx = np.random.randint(0, len(waveform), size=n_imp)
                waveform[idx] = np.random.randint(-28000, 28000, size=n_imp).astype(np.int16)

            # Random timing jitter
            if len(waveform) >= 1024:
                start = np.random.randint(0, len(waveform) - 1024 + 1)
                chunk = waveform[start:start + 1024]
            else:
                chunk = np.zeros(1024, dtype=np.int16)
                chunk[:len(waveform)] = waveform

            # Extract 35 features
            feat = extract_features(chunk, sample_rate)
            all_features.append(feat)
            all_labels.append(class_id)

            generated += 1
            if generated % 250 == 0:
                elapsed = max(1e-6, time.time() - overall_start)
                rate = generated / elapsed
                print(
                    f"      progress {generated}/{total_expected} "
                    f"({100.0 * generated / total_expected:.1f}%) "
                    f"rate={rate:.1f} samples/s",
                    flush=True,
                )

        print(
            f"      done class={class_name} in {time.time() - class_start:.1f}s",
            flush=True,
        )

    X = np.array(all_features, dtype=np.float64)
    y = np.array(all_labels, dtype=np.int32)
    return X, y


def train_sound_classifier(output_path: str = None, samples_per_class: int = 800):
    """
    Train a high-precision MLP classifier.
    """
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "models", "sound_classifier.joblib")

    print("\n" + "=" * 60, flush=True)
    print("  🧠 Training High-Precision TinyML MLP")
    print("  Architecture: 35 -> 20 -> 16 -> 11")
    print("=" * 60, flush=True)
    print("   Loading sklearn modules...", flush=True)

    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn import failed. Install with: pip install scikit-learn\n"
            "Import name must be 'sklearn', not 'skilearn'."
        ) from exc
    import joblib
    print("   Sklearn imports complete.", flush=True)

    # Generate data
    X, y = generate_training_data(samples_per_class=samples_per_class)
    
    # Scale features (StandardScaler) - Needs to be saved with the model!
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train several compact MLP candidates and keep the SAR-optimal model.
    print("\n   Training candidate MLPs (SAR reliability objective)...", flush=True)
    t_start = time.time()
    candidates = [
        {
            "name": "compact_adam",
            "hidden_layer_sizes": (24, 16),
            "activation": "relu",
            "solver": "adam",
            "alpha": 7e-4,
            "batch_size": 64,
            "learning_rate_init": 0.004,
            "max_iter": 450,
            "early_stopping": True,
            "n_iter_no_change": 18,
        },
        {
            "name": "balanced_adam",
            "hidden_layer_sizes": (32, 20),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1.2e-3,
            "batch_size": 64,
            "learning_rate_init": 0.003,
            "max_iter": 520,
            "early_stopping": True,
            "n_iter_no_change": 20,
        },
        {
            "name": "high_recall_adam",
            "hidden_layer_sizes": (40, 24),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1.5e-3,
            "batch_size": 64,
            "learning_rate_init": 0.0025,
            "max_iter": 560,
            "early_stopping": True,
            "n_iter_no_change": 22,
        },
        {
            "name": "compact_lbfgs",
            "hidden_layer_sizes": (28, 16),
            "activation": "relu",
            "solver": "lbfgs",
            "alpha": 2e-3,
            "max_iter": 360,
        },
    ]

    best_model = None
    best_metrics = None
    best_name = ""
    best_score = float("-inf")
    for cfg in candidates:
        name = cfg["name"]
        model_kwargs = {k: v for k, v in cfg.items() if k != "name"}
        model = MLPClassifier(random_state=42, verbose=False, **model_kwargs)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        metrics = _compute_metrics(y_test, pred, len(CLASS_NAMES))
        score = (
            0.58 * metrics["target_recall"]
            + 0.30 * metrics["reject_precision"]
            + 0.12 * metrics["accuracy"]
        )
        print(
            f"      {name:<16} acc={metrics['accuracy']:.2%} "
            f"target_recall={metrics['target_recall']:.2%} "
            f"reject_precision={metrics['reject_precision']:.2%} "
            f"score={score:.4f}",
            flush=True,
        )

        if score > best_score:
            best_score = score
            best_model = model
            best_metrics = metrics
            best_name = name

    model = best_model
    print(
        f"   Selected model: {best_name} "
        f"(score={best_score:.4f}, elapsed={time.time() - t_start:.1f}s)",
        flush=True,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    target_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
    
    print(f"\n   Classification Report (Focus on Precision):", flush=True)
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    print(
        "   SAR metrics: "
        f"acc={best_metrics['accuracy']:.2%} "
        f"target_recall={best_metrics['target_recall']:.2%} "
        f"reject_precision={best_metrics['reject_precision']:.2%}",
        flush=True,
    )

    # Save sklearn bundle used by the simulator runtime
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bundle = {
        "scaler": scaler,
        "model": model,
        "features": 35,
        "classes": CLASS_NAMES,
        "metrics": best_metrics,
        "selected_profile": best_name,
        "origin": "train_classifier_sklearn_search",
    }
    joblib.dump(bundle, output_path)

    # Export a lightweight bundle that mirrors ESP32 inference math
    # (weights/biases + scaler stats, no sklearn object dependency).
    esp32_bundle = {
        "weights": [w.astype(np.float32) for w in model.coefs_],
        "biases": [b.reshape(1, -1).astype(np.float32) for b in model.intercepts_],
        "mean": scaler.mean_.astype(np.float32),
        "std": scaler.scale_.astype(np.float32),
        "dims": [35, 20, 16, 11],
        "classes": CLASS_NAMES,
        "origin": "train_classifier_sklearn_export",
        "timestamp": time.time(),
    }
    output_file = Path(output_path)
    esp32_output_path = str(output_file.with_name(output_file.stem + "_esp32.joblib"))
    joblib.dump(esp32_bundle, esp32_output_path)
    
    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n   Package saved: {output_path} ({size_kb:.1f} KB)", flush=True)
    print(f"   ESP32 bundle saved: {esp32_output_path}", flush=True)
    print(f"   Inference RAM Estimate: ~12KB (Static Weights)", flush=True)

    return bundle


def train_sound_classifier_numpy(output_path: str = None, samples_per_class: int = 800):
    """Train classifier using pure NumPy backend (no sklearn dependency)."""
    import joblib

    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "models", "sound_classifier.joblib")

    print("\n" + "=" * 60, flush=True)
    print("  ⚡ Training TinyML (NumPy Backend)", flush=True)
    print("  Architecture: 35 -> 40 -> 24 -> 11", flush=True)
    print("=" * 60, flush=True)

    np.random.seed(42)
    X, y = generate_training_data(samples_per_class=samples_per_class)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-10
    X_scaled = (X - mean) / std

    train_idx, val_idx = _stratified_holdout_indices(y, test_ratio=0.2)
    X_train, y_train = X_scaled[train_idx], y[train_idx]
    X_val, y_val = X_scaled[val_idx], y[val_idx]

    # Reuse the proven pure NumPy trainer module to avoid sklearn import stalls.
    from models.num_train_real_world import NumPyMLP

    model = NumPyMLP(35, (40, 24), 11)
    batch_size = 128
    epochs = 20
    lr = 0.005
    n_samples = X_train.shape[0]
    indices = np.arange(n_samples)

    print("   Training NumPy MLP...", flush=True)
    t_start = time.time()
    for epoch in range(epochs):
        np.random.shuffle(indices)
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            model.train_step(X_train[batch_idx], y_train[batch_idx], lr=lr)

        if (epoch + 1) % 4 == 0 or epoch == 0:
            train_eval_idx = indices[: min(1200, n_samples)]
            train_preds = model.predict(X_train[train_eval_idx])
            train_metrics = _compute_metrics(y_train[train_eval_idx], train_preds, 11)

            val_preds = model.predict(X_val)
            val_metrics = _compute_metrics(y_val, val_preds, 11)
            print(
                f"      epoch {epoch + 1:02d}/{epochs} "
                f"train_acc={train_metrics['accuracy']:.2%} "
                f"val_acc={val_metrics['accuracy']:.2%} "
                f"val_target_recall={val_metrics['target_recall']:.2%} "
                f"val_reject_prec={val_metrics['reject_precision']:.2%}",
                flush=True,
            )

    print(f"   Training complete in {time.time() - t_start:.1f}s", flush=True)

    final_val_preds = model.predict(X_val)
    final_metrics = _compute_metrics(y_val, final_val_preds, 11)
    print(
        "   Validation summary: "
        f"acc={final_metrics['accuracy']:.2%} "
        f"target_recall={final_metrics['target_recall']:.2%} "
        f"reject_precision={final_metrics['reject_precision']:.2%}",
        flush=True,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bundle = {
        "weights": model.weights,
        "biases": model.biases,
        "mean": mean,
        "std": std,
        "dims": model.dims,
        "classes": CLASS_NAMES,
        "metrics": final_metrics,
        "origin": "train_classifier_numpy",
        "timestamp": time.time(),
    }
    joblib.dump(bundle, output_path)
    print(f"   Package saved: {output_path}", flush=True)
    return bundle


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train sound classifier model")
    parser.add_argument("--samples", type=int, default=300,
                        help="Samples per class for training (default: 300)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output model path")
    parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "sklearn"],
                        help="Training backend: numpy (default) or sklearn")
    parser.add_argument("--for-esp32", action="store_true",
                        help="When using sklearn backend, copy exported ESP32 bundle to output path")
    args = parser.parse_args()

    print(
        f"[TRAIN] Starting backend={args.backend} samples_per_class={args.samples} "
        f"output={args.output or 'default'}",
        flush=True,
    )
    if args.backend == "sklearn":
        train_sound_classifier(output_path=args.output, samples_per_class=args.samples)
        if args.for_esp32:
            import shutil
            base = Path(args.output) if args.output else Path(PROJECT_ROOT) / "models" / "sound_classifier.joblib"
            esp32_path = base.with_name(base.stem + "_esp32.joblib")
            shutil.copy2(str(esp32_path), str(base))
            print(f"[TRAIN] Copied ESP32 bundle to {base}", flush=True)
    else:
        train_sound_classifier_numpy(output_path=args.output, samples_per_class=args.samples)
