# ESP32 Node Firmware (Hardware Runtime)

This firmware implements the same node-side flow used in the Simulation project:

1. I2S microphone capture at 16 kHz, 32-bit raw I2S converted to PCM16
2. STA/LTA trigger detection
3. FFT whistle-band prefilter (2500-4000 Hz)
4. TinyML MLP inference (35 -> 40 -> 24 -> 11) from exported Simulation model weights
5. 31-byte LoRa payload with CRC-8 (same layout as Simulation/node/lora_tx.py)
6. GPS status and epoch timestamp derived from parsed NMEA (GGA+RMC) when available, otherwise mocked GPS stream

## Packet Compatibility

The transmitted binary packet is exactly 31 bytes:

- node_id: uint8
- ts_micros: uint64 (Unix epoch microseconds derived from GPS UTC)
- magnitude: uint16
- peak_freq_hz: uint16
- ml_class: uint8
- ml_confidence: uint8
- snr_db: uint8
- battery_pct: uint8
- temperature_c: int8
- gps_hdop: uint8
- latitude_e7: int32 (latitude scaled by 1e7)
- longitude_e7: int32 (longitude scaled by 1e7)
- event_count: uint16
- crc8: uint8
- reserved: uint8

CRC-8 polynomial is 0x07 (same as the Simulation implementation).

## Wiring Defaults (ESP32-S3 DevKitC-1)

You can override these in platformio.ini build flags.

- I2S mic BCLK: GPIO5
- I2S mic WS/LRCLK: GPIO4
- I2S mic SD: GPIO6
- LoRa E22 TX (ESP32->E22 RX): GPIO8
- LoRa E22 RX (ESP32<-E22 TX): GPIO13
- LoRa AUX: GPIO11
- LoRa M0: GPIO10
- LoRa M1: GPIO9

## Build and Flash

From the Node folder:

```powershell
pio run
pio run -t upload
pio device monitor
```

## Updating TinyML Weights From Simulation

When you retrain in `Simulation/models`, export the ESP32 bundle to firmware header:

```powershell
cd ../Simulation
python models/export_esp32_header.py --input models/sound_classifier_esp32.joblib --output ../Node/include/tinyml_model_params.h
```

Then rebuild/flash in `Node`.

## Deploying Multiple Hardware Nodes

Set a unique node id for each device before flashing:

- Node 1: -DNODE_ID=1
- Node 2: -DNODE_ID=2
- Node 3: -DNODE_ID=3
- Node 4: -DNODE_ID=4

Example env override:

```ini
build_flags =
  -DNODE_ID=2
```

## Notes

- LoRa UART is initialized at 9600 baud (E22 default profile expectation).
- GPS parser listens to UART NMEA GGA/RMC if `PIN_GPS_RX` and `PIN_GPS_TX` are configured.
- If no raw GPS fix is available, firmware keeps producing and parsing mocked GGA+RMC sentences so packet `ts_micros` stays epoch-based while `latitude_e7`/`longitude_e7` and `gps_hdop` remain populated.
- This firmware uses ESP-IDF level drivers through Arduino framework for direct hardware access.
- If your board pinout differs, only build flags need to change; packet format must remain unchanged for Simulation solver compatibility.
