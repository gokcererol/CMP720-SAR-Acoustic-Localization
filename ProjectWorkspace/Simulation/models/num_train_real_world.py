import os
import sys
import time
import numpy as np
import joblib

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from node.ml_classifier import CLASS_NAMES

FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "esc50_features_35d.npy")
LABELS_PATH = os.path.join(PROJECT_ROOT, "data", "esc50_labels.npy")
MODEL_OUTPUT = os.path.join(PROJECT_ROOT, "models", "sound_classifier.joblib")

# ReLU Activation
def relu(x): return np.maximum(0, x)
def relu_derivative(x): return (x > 0).astype(float)

# Softmax for multi-class
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

class NumPyMLP:
    """Pure NumPy implementation of MLP (Adam Optimizer)."""
    def __init__(self, input_size, hidden_layers, output_size):
        self.dims = [input_size] + list(hidden_layers) + [output_size]
        self.weights = []
        self.biases = []
        
        # Xavier/He Initialization
        for i in range(len(self.dims)-1):
            w = np.random.randn(self.dims[i], self.dims[i+1]) * np.sqrt(2.0 / self.dims[i])
            b = np.zeros((1, self.dims[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            
        # Adam components
        self.mw = [np.zeros_like(w) for w in self.weights]
        self.vw = [np.zeros_like(w) for w in self.weights]
        self.mb = [np.zeros_like(b) for b in self.biases]
        self.vb = [np.zeros_like(b) for b in self.biases]
        self.t = 0

    def forward(self, X):
        activations = [X]
        zs = []
        for i in range(len(self.weights)-1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            zs.append(z)
            activations.append(relu(z))
        
        # Final layer (no ReLU, just Softmax)
        z_final = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        zs.append(z_final)
        activations.append(softmax(z_final))
        return activations, zs

    def train_step(self, X_batch, y_batch, lr=0.001):
        self.t += 1
        m = X_batch.shape[0]
        activations, zs = self.forward(X_batch)
        probs = activations[-1]
        
        # One-hot labels
        y_oh = np.zeros((m, self.dims[-1]))
        y_oh[np.arange(m), y_batch] = 1
        
        # Backward Pass
        dz = probs - y_oh
        grads_w = []
        grads_b = []
        
        for i in reversed(range(len(self.weights))):
            dw = np.dot(activations[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            grads_w.insert(0, dw)
            grads_b.insert(0, db)
            
            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * relu_derivative(zs[i-1])
        
        # Adam Updates
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.weights)):
            self.mw[i] = beta1 * self.mw[i] + (1 - beta1) * grads_w[i]
            self.vw[i] = beta2 * self.vw[i] + (1 - beta2) * (grads_w[i]**2)
            mw_hat = self.mw[i] / (1 - beta1**self.t)
            vw_hat = self.vw[i] / (1 - beta2**self.t)
            self.weights[i] -= lr * mw_hat / (np.sqrt(vw_hat) + eps)
            
            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * grads_b[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (grads_b[i]**2)
            mb_hat = self.mb[i] / (1 - beta1**self.t)
            vb_hat = self.vb[i] / (1 - beta2**self.t)
            self.biases[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    def predict(self, X):
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)

def run_rescue_training():
    print("\n" + "="*50, flush=True)
    print(" ⚡ PURE NUMPY RESCUE TRAINER 🧠", flush=True)
    print("="*50, flush=True)
    
    # 1. Load data
    print(f"\n[1/4] Loading {FEATURES_PATH}...", flush=True)
    X = np.load(FEATURES_PATH)
    y = np.load(LABELS_PATH)
    print(f"✅ Loaded {X.shape[0]} samples with {X.shape[1]} features.", flush=True)
    
    # 2. Scale Data (Manual StandardScaler)
    print("[2/4] Normalizing data (Mean/Std)...", flush=True)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-10
    X_scaled = (X - mean) / std
    
    # 3. Training
    num_classes = 11
    model = NumPyMLP(35, (40, 24), num_classes)
    
    batch_size = 256
    epochs = 30
    lr = 0.005
    
    print(f"[3/4] Training MLP (40, 24) for {epochs} epochs...", flush=True)
    n_samples = X_scaled.shape[0]
    indices = np.arange(n_samples)
    
    start_t = time.time()
    for epoch in range(epochs):
        np.random.shuffle(indices)
        epoch_loss = 0
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            model.train_step(X_scaled[batch_idx], y[batch_idx], lr=lr)
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            preds = model.predict(X_scaled[:1000]) # Quick evaluation
            acc = np.mean(preds == y[:1000])
            print(f"  🚩 Epoch {epoch+1:2d}/{epochs} | Estimate Acc: {acc:.2%}", flush=True)
            
    print(f"✅ Training complete in {time.time() - start_t:.1f}s.", flush=True)
    
    # 4. Save
    print("[4/4] Saving model bundle...", flush=True)
    bundle = {
        "weights": model.weights,
        "biases": model.biases,
        "mean": mean,
        "std": std,
        "dims": model.dims,
        "classes": CLASS_NAMES,
        "origin": "NumPy Rescue Trainer",
        "timestamp": time.time()
    }
    joblib.dump(bundle, MODEL_OUTPUT)
    print(f"🚀 SUCCESS! Model saved to {MODEL_OUTPUT}", flush=True)

if __name__ == "__main__":
    try:
        run_rescue_training()
    except Exception as e:
        print(f"\n❌ Rescue Training Failed: {e}", flush=True)
