import sys
import os
import argparse
import subprocess

# Prevent multi-threading library conflicts (e.g., OpenBLAS/MKL) during import
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["JOBLIB_MULTIPROCESSING_BACKEND"] = "threading"

# Immediate start signal
print("🔍 [DEBUG] train_real_world.py: Starting script...", flush=True)

import time
import numpy as np

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from node.ml_classifier import CLASS_NAMES
from models.num_train_real_world import NumPyMLP

FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "esc50_features_35d.npy")
LABELS_PATH = os.path.join(PROJECT_ROOT, "data", "esc50_labels.npy")
MODEL_OUTPUT = os.path.join(PROJECT_ROOT, "models", "sound_classifier.joblib")


def _probe_sklearn_import(timeout_sec: int = 12) -> bool:
    """Check sklearn availability in an isolated process to avoid hard hangs."""
    cmd = [
        sys.executable,
        "-c",
        "import sklearn; print(sklearn.__version__)",
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(1, int(timeout_sec)),
            text=True,
        )
        if result.returncode == 0:
            ver = (result.stdout or "").strip() or "unknown"
            print(f"✅ sklearn probe succeeded (version={ver}).", flush=True)
            return True

        print("⚠️ sklearn probe failed with non-zero exit code.", flush=True)
        if result.stderr:
            print(result.stderr.strip(), flush=True)
        return False
    except subprocess.TimeoutExpired:
        print(f"⚠️ sklearn probe timed out after {timeout_sec}s.", flush=True)
        return False
    except Exception as exc:
        print(f"⚠️ sklearn probe error: {exc}", flush=True)
        return False


def _stratified_holdout_indices(y: np.ndarray, test_ratio: float = 0.15):
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


def train_real_world_numpy():
    print("\n" + "="*50, flush=True)
    print(" ⚡ TRAINING REAL-WORLD SAR CLASSIFIER (NumPy)", flush=True)
    print("="*50, flush=True)

    print(f"\n[STEP 1/4] Loading {FEATURES_PATH}...", flush=True)
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError("Features not found. Please run ingest_esc50.py first.")

    X = np.load(FEATURES_PATH)
    y = np.load(LABELS_PATH)
    print(f"✅ Loaded {len(X)} samples with 35 features each.", flush=True)

    print("\n[STEP 2/4] Normalizing and splitting...", flush=True)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-10
    X_scaled = (X - mean) / std
    train_idx, test_idx = _stratified_holdout_indices(y, test_ratio=0.15)
    X_train, y_train = X_scaled[train_idx], y[train_idx]
    X_test, y_test = X_scaled[test_idx], y[test_idx]
    print(f"🏁 Train size: {len(X_train)} | Test size: {len(X_test)}", flush=True)

    print("\n[STEP 3/4] Training NumPy MLP...", flush=True)
    model = NumPyMLP(35, (40, 24), 11)
    batch_size = 256
    epochs = 30
    lr = 0.005
    n_samples = X_train.shape[0]
    indices = np.arange(n_samples)

    t_start = time.time()
    for epoch in range(epochs):
        np.random.shuffle(indices)
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            model.train_step(X_train[batch_idx], y_train[batch_idx], lr=lr)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            eval_idx = indices[: min(2000, n_samples)]
            train_acc = float(np.mean(model.predict(X_train[eval_idx]) == y_train[eval_idx]))
            test_acc = float(np.mean(model.predict(X_test) == y_test))
            print(f"  Epoch {epoch+1:02d}/{epochs} | train_acc={train_acc:.2%} | test_acc={test_acc:.2%}", flush=True)

    print(f"✅ Training complete in {time.time() - t_start:.1f} seconds.", flush=True)

    print("\n[STEP 4/4] Saving NumPy model bundle...", flush=True)
    bundle = {
        "weights": model.weights,
        "biases": model.biases,
        "mean": mean,
        "std": std,
        "dims": model.dims,
        "classes": CLASS_NAMES,
        "timestamp": time.time(),
        "origin": "ESC-50 Real World NumPy"
    }
    import joblib
    joblib.dump(bundle, MODEL_OUTPUT)
    size_kb = os.path.getsize(MODEL_OUTPUT) / 1024
    print(f"✅ Model saved to {MODEL_OUTPUT}", flush=True)
    print(f"📦 Total Bundle Size: {size_kb:.1f} KB", flush=True)


def train_real_world_sklearn():
    # DEFERRED IMPORTS (to prevent stalling during startup)
    print("🧠 [STAGE 0/5] Loading Machine Learning libraries...", flush=True)
    try:
        import joblib
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        from sklearn.preprocessing import StandardScaler
        print("✅ Libraries loaded successfully.", flush=True)
    except Exception as e:
        print(f"\n❌ CRITICAL: Could not load scikit-learn: {e}", flush=True)
        print("💡 Suggestion: run with --backend numpy", flush=True)
        return

    print("\n" + "="*50, flush=True)
    print(" 🧠 TRAINING REAL-WORLD SAR CLASSIFIER", flush=True)
    print("="*50, flush=True)

    # 1. Load cached features
    print(f"\n[STEP 1/5] Loading {FEATURES_PATH}...", flush=True)
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError("Features not found. Please run ingest_esc50.py first.")

    X = np.load(FEATURES_PATH)
    y = np.load(LABELS_PATH)
    print(f"✅ Loaded {len(X)} samples with 35 features each.")

    # 2. Scale and Split
    print("\n[STEP 2/5] Normalizing features (StandardScaler)...", flush=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("[STEP 2/5] Splitting into Train (85%) and Test (15%)...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"🏁 Train size: {len(X_train)} | Test size: {len(X_test)}", flush=True)

    # 3. Define and Train MLP
    print("\n[STEP 3/5] Starting MLP Neural Network Training...", flush=True)
    print("💡 You will see 'Iteration' logs below. This means the model is learning.", flush=True)

    model = MLPClassifier(
        hidden_layer_sizes=(40, 24),
        activation='relu',
        solver='adam',
        alpha=0.0005,
        learning_rate_init=0.005,
        max_iter=200,
        batch_size=256,
        random_state=42,
        verbose=True
    )

    t_start = time.time()
    print("⏳ Learning from real-world samples...", flush=True)
    model.fit(X_train, y_train)
    print(f"✅ Training complete in {time.time() - t_start:.1f} seconds.", flush=True)

    # 4. Evaluate
    print("\n[STEP 4/5] Evaluating on Real-World Hold-out set...", flush=True)
    y_pred = model.predict(X_test)

    present_ids = sorted(np.unique(y))
    target_names = [CLASS_NAMES[i] for i in present_ids]

    print("\nClassification Report (Pure Real Data):", flush=True)
    print(classification_report(y_test, y_pred, target_names=target_names), flush=True)

    # 5. Save Bundle
    print("\n[STEP 5/5] Saving model bundle...", flush=True)
    bundle = {
        "scaler": scaler,
        "model": model,
        "features": 35,
        "classes": CLASS_NAMES,
        "timestamp": time.time(),
        "origin": "ESC-50 Real World"
    }
    joblib.dump(bundle, MODEL_OUTPUT)

    size_kb = os.path.getsize(MODEL_OUTPUT) / 1024
    print(f"✅ Model saved to {MODEL_OUTPUT}", flush=True)
    print(f"📦 Total Bundle Size: {size_kb:.1f} KB", flush=True)
    print(f"⚙️  Architecture: {model.hidden_layer_sizes}", flush=True)
    print("🎯 Target hardware: ESP32-S3 (ESP-NN ready)", flush=True)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Train real-world classifier")
        parser.add_argument("--backend", choices=["auto", "numpy", "sklearn"], default="auto")
        parser.add_argument("--probe-timeout", type=int, default=12,
                            help="Seconds to wait for sklearn import probe in auto mode")
        args = parser.parse_args()

        if args.backend == "sklearn":
            train_real_world_sklearn()
        elif args.backend == "numpy":
            train_real_world_numpy()
        else:
            print("🔎 Auto mode: probing sklearn import health...", flush=True)
            if _probe_sklearn_import(timeout_sec=args.probe_timeout):
                print("➡️ Auto mode selected sklearn backend.", flush=True)
                train_real_world_sklearn()
            else:
                print("➡️ Auto mode selected numpy backend.", flush=True)
                train_real_world_numpy()
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
