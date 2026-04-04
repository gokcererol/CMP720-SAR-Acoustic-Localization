"""
ESC-50 Data Ingestor — High-Precision TinyML Pipeline.
Downloads, Resamples (16kHz), Maps, and Featurizes (35D) the real-world dataset.
Optimized for visibility and absolute reliability on Windows.
"""

import os
import sys
import zipfile
import requests
import pandas as pd
import numpy as np
import librosa
import soundfile as sf

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from node.ml_classifier import extract_features, CLASS_NAMES

DATA_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
DOWNLOAD_PATH = os.path.join(PROJECT_ROOT, "data", "esc50.zip")
EXTRACT_DIR = os.path.join(PROJECT_ROOT, "data", "esc50_raw")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "esc50_features_35d.npy")
LABELS_PATH = os.path.join(PROJECT_ROOT, "data", "esc50_labels.npy")

# Mapping from ESC-50 classes to SAR 11-class Taxonomy.
# NOTE: ESC-50 has no human whistle class; chirping_birds belongs to "animal".
ESC_TO_SAR = {
    "chirping_birds": 7,
    "crying_baby": 1, "sneezing": 1, "laughing": 1, "breathing": 1, "coughing": 1,
    "clapping": 2, "footsteps": 2, "glass_breaking": 2,
    "door_wood_knock": 3,
    "crackling_fire": 4, 
    "chainsaw": 5, "hand_saw": 5, "drilling": 5, "engine": 5,
    "helicopter": 6, "airplane": 6,
    "dog": 7, "rooster": 7, "crickets": 7, "crow": 7, "cat": 7, "pig": 7, "cow": 7, "frog": 7, "sheep": 7, "hen": 7,
    "wind": 8, "thunderstorm": 8,
    "rain": 9,
    "sea_waves": 10, "water_drops": 10, "insects": 10
}

def download_dataset():
    os.makedirs(os.path.dirname(DOWNLOAD_PATH), exist_ok=True)
    if os.path.exists(EXTRACT_DIR):
        for root, _, files in os.walk(EXTRACT_DIR):
            if "esc50.csv" in files:
                print("✅ ESC-50 already extracted.")
                return

    print(f"📡 Downloading ESC-50 Dataset (~600MB)...")
    try:
        r = requests.get(DATA_URL, stream=True, timeout=30)
        r.raise_for_status()
        with open(DOWNLOAD_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024): # 1MB chunks
                if chunk: 
                    f.write(chunk)
                    print(".", end="", flush=True)
        print("\n📦 Extracting zip...")
        with zipfile.ZipFile(DOWNLOAD_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("✅ Extraction complete.")
    except Exception as e:
        if os.path.exists(DOWNLOAD_PATH): os.remove(DOWNLOAD_PATH)
        raise e

def process_single_file(args):
    """Worker function to process a single audio file."""
    file_path, sar_class_id = args
    try:
        y, native_sr = sf.read(file_path)
        if len(y.shape) > 1: y = np.mean(y, axis=1)
        
        # Optimization: Use soxr for extremely fast resampling
        if native_sr != 16000:
            try:
                import soxr
                y = soxr.resample(y, native_sr, 16000)
            except ImportError:
                y = librosa.resample(y, orig_sr=native_sr, target_sr=16000, res_type='kaiser_fast')
        
        features_list = []
        labels_list = []
        
        for start in range(0, len(y) - 1024, 512):
            chunk_int16 = (y[start:start + 1024] * 32767).astype(np.int16)
            feat = extract_features(chunk_int16, sample_rate=16000)
            if not np.all(feat == 0):
                features_list.append(feat)
                labels_list.append(sar_class_id)
        
        return features_list, labels_list
    except Exception as e:
        return None, None

def process_audio():
    meta_path = ""
    audio_root = ""
    for root, dirs, files in os.walk(EXTRACT_DIR):
        if "esc50.csv" in files:
            meta_path = os.path.join(root, "esc50.csv")
            parent = os.path.dirname(root)
            if os.path.exists(os.path.join(parent, "audio")): audio_root = os.path.join(parent, "audio")
            elif os.path.exists(os.path.join(root, "audio")): audio_root = os.path.join(root, "audio")
            break
    
    if not meta_path or not audio_root:
        raise FileNotFoundError("Could not find meta/audio dir.")

    df = pd.read_csv(meta_path)
    all_tasks = []
    for _, row in df.iterrows():
        if row['category'] in ESC_TO_SAR:
            fp = os.path.join(audio_root, row['filename'])
            if os.path.exists(fp): all_tasks.append((fp, ESC_TO_SAR[row['category']]))

    print(f"✅ Found {len(all_tasks)} target files.")
    print("🎧 ⌛ INITIALIZING Optimized Sequential Ingestion...")
    
    all_features = []
    all_labels = []
    
    import time
    start_t = time.time()
    processed_count = 0
    
    # Check for tqdm
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    # Sequential processing loop
    pbar = tqdm(total=len(all_tasks), desc="Processing ESC-50") if has_tqdm else None
    
    for i, task in enumerate(all_tasks):
        file_path, sar_class_id = task
        # Optional: Print file name if it's taking too long (every 10 files)
        if not has_tqdm and i % 10 == 0:
            print(f"   ⌛ Processing {os.path.basename(file_path)}...")
            
        feats, labels = process_single_file(task)
        if feats:
            all_features.extend(feats)
            all_labels.extend(labels)
        else:
            print(f"   ⚠️  Failed to process {file_path}")
        
        processed_count += 1
        if pbar:
            pbar.update(1)
        elif processed_count % 50 == 0:
            print(f"   [{processed_count}/{len(all_tasks)}] files processed...")
    
    if pbar: pbar.close()

    if not all_features:
        print("\n❌ No features extracted!")
        return

    print(f"\n🔗 Dataset processed in {time.time() - start_t:.1f}s.")
    print("💾 Finalizing and caching...")
    X = np.vstack(all_features)
    y = np.array(all_labels, dtype=np.int32)
    np.save(FEATURES_PATH, X)
    np.save(LABELS_PATH, y)
    print(f"📊 Final Dataset: {len(y)} 35-D feature vectors cached at {FEATURES_PATH}")

if __name__ == "__main__":
    try:
        print("\n" + "="*50)
        print(" 🚀 STARTING SOUND DATASET INGESTION")
        print("="*50)
        download_dataset()
        process_audio()
        print("\nReady for Step 3: models/train_real_world.py! 🚀")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
