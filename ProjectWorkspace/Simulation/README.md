# SAR-TDoA Acoustic Localization Simulation

This project simulates a Search and Rescue acoustic localization system.

You run one command, then use a web page to fire sounds and see where the solver estimates the source.

## What You Get

- 4 virtual sensor nodes (ESP32-like behavior)
- LoRa link simulation between nodes and solver
- TDoA localization solver
- Node-side sound classifier (whistle, voice, impact, knocking, etc.)
- Live web dashboard with map, controls, and event history

## Beginner Quick Start (Windows)

### 1) Open terminal in Simulation folder

```powershell
cd ProjectWorkspace/Simulation
```

### 2) Create virtual environment

```powershell
python -m venv .venv
```

### 3) Activate virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4) Install dependencies

```powershell
pip install -r requirements.txt
```

### 5) Run simulator

```powershell
python run.py
```

Open browser at:

- `http://127.0.0.1:8080`

## First 2 Things To Try

### Fire one predefined scenario

```powershell
python run.py --test A1
```

### Run without speaker playback

```powershell
python run.py --no-speaker
```

## Most Useful Commands

```powershell
# Show all launcher options
python run.py --help

# Train models before running
python run.py --train

# Verify predefined scenarios
python run.py --verify
```

## Project Structure (Simple View)

```text
Simulation/
  run.py              # Start everything
  config.yaml         # Main settings
  requirements.txt    # Python packages

  node/               # Node-side detection/classification
  solver/             # TDoA solver and filters
  network/            # LoRa channel model
  source/             # Audio generation + propagation
  web/                # Flask dashboard
  testing/            # Scenario catalog and benchmark scripts
```

## Configuration Basics

Edit `config.yaml` if you want to tune behavior.

Most important sections:

- `environment`: speed of sound, multipath, ambient noise
- `nodes`: sensor positions, trigger threshold, classifier settings
- `lora`: reliability and collision behavior
- `solver`: method and quality thresholds
- `web`: host/port

## Data Setup (Optional)

Large dataset/cache files are not stored in Git.

If you want full real-audio ingestion/training, run:

```powershell
python models/ingest_esc50.py
python models/train_real_world.py --backend auto
```

## Troubleshooting

### Module import errors

Run inside activated `.venv` and reinstall:

```powershell
pip install -r requirements.txt
```

### Web page does not open

- Check if `127.0.0.1:8080` is already used.
- Change `web.port` in `config.yaml` and run again.

### Low event detection quality

- Lower `nodes.stream_processor.sta_lta_threshold` in `config.yaml`.
- Check `nodes.ml_classifier.confidence_threshold`.
- Run scenario tests with `--verify` to compare behavior.
