"""
Test Scenarios — 58 predefined scenarios across categories A-H.
Each scenario defines sound event parameters and expected outcomes.
"""

from typing import Dict, List
import math
import random

# Default node quadrilateral center
CENTER_LAT = 39.867449
CENTER_LON = 32.733585

# Node positions
NODES = {
    1: {"lat": 39.867000, "lon": 32.733000},
    2: {"lat": 39.867000, "lon": 32.734170},
    3: {"lat": 39.867898, "lon": 32.734170},
    4: {"lat": 39.867898, "lon": 32.733000},
}


def _scenario(name: str, category: str, sound_type: str,
              lat: float, lon: float, amplitude: float,
              expected_status: str, max_error_m: float = 5.0,
              jitter_ms: float = 2.0, reliability: float = 0.95,
              terrain: str = "flat", notes: str = "",
              **synth_kwargs) -> Dict:
    """Build a scenario definition."""
    return {
        "name": name,
        "category": category,
        "sound_type": sound_type,
        "lat": lat,
        "lon": lon,
        "amplitude": amplitude,
        "expected_status": expected_status,
        "max_error_m": max_error_m,
        "jitter_ms": jitter_ms,
        "reliability": reliability,
        "terrain": terrain,
        "notes": notes,
        "synth_kwargs": synth_kwargs,
    }


def _offset_latlon(center_lat: float, center_lon: float,
                   east_m: float, north_m: float) -> tuple:
    """Convert local ENU meter offsets to lat/lon around a center point."""
    lat = center_lat + north_m / 111132.0
    lon = center_lon + east_m / (111320.0 * math.cos(math.radians(center_lat)))
    return lat, lon


# ===== CATEGORY A: Successful Detection → CONFIRMED =====
CATEGORY_A = [
    _scenario("A1: Perfect center whistle", "A", "whistle",
              CENTER_LAT, CENTER_LON, 0.8,
              "CONFIRMED", 5.0, jitter_ms=2.0),
    _scenario("A2: Strong whistle, light breeze", "A", "whistle",
              CENTER_LAT, CENTER_LON, 0.7,
              "CONFIRMED", 5.0, jitter_ms=5.0),
    _scenario("A3: Human scream", "A", "human_voice",
              CENTER_LAT, CENTER_LON, 0.6,
              "CONFIRMED", 5.0, jitter_ms=3.0),
    _scenario("A4: Impact clap", "A", "impact",
              CENTER_LAT, CENTER_LON, 0.9,
              "CONFIRMED", 3.0, jitter_ms=1.0),
    _scenario("A5: Human knocking (rhythmic)", "A", "knocking",
              CENTER_LAT, CENTER_LON, 0.7,
              "CONFIRMED", 5.0, jitter_ms=4.0),
    _scenario("A6: Whistle from SE corner", "A", "whistle",
              39.866800, 32.734300, 0.6,
              "CONFIRMED", 8.0, jitter_ms=4.0),
    _scenario("A7: Whistle from NW corner", "A", "whistle",
              39.868100, 32.732800, 0.65,
              "CONFIRMED", 8.0, jitter_ms=3.0),
    _scenario("A8: Knocking near Node 3", "A", "knocking",
              39.867850, 32.734100, 0.75,
              "CONFIRMED", 5.0, jitter_ms=2.0),
]

# ===== CATEGORY B: Degraded Detection → WEAK DATA =====
CATEGORY_B = [
    _scenario("B1: Deep rubble weak whistle", "B", "whistle",
              CENTER_LAT, CENTER_LON, 0.15,
              "WEAK_DATA", 25.0, jitter_ms=15.0, terrain="rubble"),
    _scenario("B2: 3-node detection (1 deaf)", "B", "whistle",
              CENTER_LAT, CENTER_LON, 0.3,
              "WEAK_DATA", 15.0, reliability=0.75),
    _scenario("B3: Edge/boundary whistle", "B", "whistle",
              39.866500, 32.733585, 0.5,
              "CONFIRMED", 12.0, jitter_ms=6.0,
              notes="On sensor quad perimeter"),
    _scenario("B4: Singularity (on Node 1)", "B", "whistle",
              39.867010, 32.733010, 0.8,
              "CONFIRMED", 5.0, jitter_ms=2.0,
              notes="<5m from one sensor"),
    _scenario("B5: Collinear source", "B", "whistle",
              39.867000, 32.733585, 0.7,
              "CONFIRMED", 10.0, jitter_ms=3.0,
              notes="Between Node 1 and Node 2"),
    _scenario("B6: Two whistles 100ms apart", "B", "whistle",
              CENTER_LAT, CENTER_LON, 0.7,
              "CONFIRMED", 8.0, jitter_ms=3.0,
              notes="Dual source test"),
]

# ===== CATEGORY C: Out-of-Bounds → OUT OF BOUNDS =====
CATEGORY_C = [
    _scenario("C1: 1km away strong whistle", "C", "whistle",
              39.876000, 32.733585, 0.9,
              "OUT_OF_BOUNDS", 100.0,
              notes="~1000m north"),
    _scenario("C2: 500m medium whistle", "C", "whistle",
              39.872000, 32.733585, 0.7,
              "OUT_OF_BOUNDS", 100.0,
              notes="~500m north"),
    _scenario("C3: 2km away very strong", "C", "whistle",
              39.885000, 32.733585, 0.95,
              "OUT_OF_BOUNDS", 200.0,
              notes="~2000m north"),
]

# ===== CATEGORY D: False Positive Rejection → REJECTED at Node Level =====
CATEGORY_D = [
    _scenario("D1: Bulldozer", "D", "machinery",
              CENTER_LAT, CENTER_LON, 0.9,
              "FILTERED", 0.0,
              notes="Should be rejected at node ML level"),
    _scenario("D2: Generator hum", "D", "motor",
              CENTER_LAT, CENTER_LON, 0.5,
              "FILTERED", 0.0),
    _scenario("D3: Dog barking", "D", "animal",
              CENTER_LAT, CENTER_LON, 0.5,
              "FILTERED", 0.0),
    _scenario("D4: Wind gust", "D", "wind",
              CENTER_LAT, CENTER_LON, 0.3,
              "FILTERED", 0.0),
    _scenario("D5: Rain on debris", "D", "rain",
              CENTER_LAT, CENTER_LON, 0.2,
              "FILTERED", 0.0),
    _scenario("D6: Rubble collapse", "D", "collapse",
              CENTER_LAT, CENTER_LON, 0.7,
              "LOG_ONLY", 0.0),
]

# ===== CATEGORY E: System Stress Tests =====
CATEGORY_E = [
    _scenario("E1: Severe storm (40ms jitter)", "E", "whistle",
              CENTER_LAT, CENTER_LON, 0.7,
              "REJECTED", 50.0, jitter_ms=40.0),
    _scenario("E2: Sensor malfunction (500ms)", "E", "whistle",
              CENTER_LAT, CENTER_LON, 0.7,
              "REJECTED", 100.0, jitter_ms=500.0),
    _scenario("E3: All LoRa packets lost", "E", "whistle",
              CENTER_LAT, CENTER_LON, 0.7,
              "NO_EVENT", 0.0, reliability=0.0),
    _scenario("E4: LoRa collision (2 overlap)", "E", "whistle",
              CENTER_LAT, CENTER_LON, 0.7,
              "WEAK_DATA", 15.0, reliability=0.5),
    _scenario("E5: GPS cold start (no PPS)", "E", "whistle",
              CENTER_LAT, CENTER_LON, 0.7,
              "REJECTED", 50.0,
              notes="Timestamps unreliable"),
    _scenario("E6: Low battery Node 4", "E", "whistle",
              CENTER_LAT, CENTER_LON, 0.6,
              "WEAK_DATA", 12.0, reliability=0.75),
    _scenario("E7: Multipath ghost", "E", "whistle",
              CENTER_LAT, CENTER_LON, 0.7,
              "CONFIRMED", 8.0,
              notes="Strong multipath reflection"),
    _scenario("E8: Rapid repeated whistles", "E", "whistle",
              CENTER_LAT, CENTER_LON, 0.7,
              "CONFIRMED", 5.0,
              notes="5 events in 10 seconds"),
]

# ===== CATEGORY F: ML Classifier Stress Tests =====
CATEGORY_F = [
    _scenario("F1: Whistle + machinery overlap", "F", "whistle",
              CENTER_LAT, CENTER_LON, 0.6,
              "CONFIRMED", 8.0,
              notes="3kHz whistle masked by 150Hz dozer"),
    _scenario("F2: Low-confidence whistle", "F", "whistle",
              CENTER_LAT, CENTER_LON, 0.12,
              "REJECTED", 0.0,
              notes="Very weak, ML confidence should be below threshold"),
    _scenario("F3: Bird chirp (whistle-like)", "F", "animal",
              CENTER_LAT, CENTER_LON, 0.5,
              "FILTERED", 0.0,
              notes="2800 Hz periodic — ML should classify as animal"),
    _scenario("F4: Knocking vs impact confusion", "F", "knocking",
              CENTER_LAT, CENTER_LON, 0.7,
              "CONFIRMED", 5.0,
              notes="Single knock — could be classified as impact (both valid TARGET)"),
]

# ===== CATEGORY G: Random Coordinates (Deterministic) =====
_RNG = random.Random(42)
CATEGORY_G = []
for i in range(1, 6):
    lat_offset = _RNG.uniform(-0.0007, 0.0007)
    lon_offset = _RNG.uniform(-0.0007, 0.0007)
    sound_type = _RNG.choice(["whistle", "impact", "knocking"])
    
    CATEGORY_G.append(
        _scenario(f"G{i}: Random {sound_type}", "G", sound_type,
                  CENTER_LAT + lat_offset, CENTER_LON + lon_offset, 0.8,
                  "CONFIRMED", 8.0, jitter_ms=2.0)
    )

# ===== CATEGORY H: Circular Coordinate Distribution =====
CATEGORY_H = []
_circle_radii_m = [35.0, 65.0, 95.0]
_circle_sound_cycle = ["whistle", "impact", "knocking", "human_voice"]
idx = 1
for ridx, radius_m in enumerate(_circle_radii_m):
    for angle_deg in [0, 60, 120, 180, 240, 300]:
        theta = math.radians(angle_deg)
        east_m = radius_m * math.cos(theta)
        north_m = radius_m * math.sin(theta)
        lat, lon = _offset_latlon(CENTER_LAT, CENTER_LON, east_m, north_m)
        sound = _circle_sound_cycle[(idx - 1) % len(_circle_sound_cycle)]
        CATEGORY_H.append(
            _scenario(
                f"H{idx}: Circular r={int(radius_m)}m a={angle_deg} {sound}",
                "H",
                sound,
                lat,
                lon,
                0.75,
                "CONFIRMED",
                max_error_m=10.0 if radius_m <= 65 else 14.0,
                jitter_ms=3.0 + ridx,
                reliability=1.0,
                notes="Circular area distribution coverage",
            )
        )
        idx += 1

# All scenarios combined
ALL_SCENARIOS = CATEGORY_A + CATEGORY_B + CATEGORY_C + CATEGORY_D + CATEGORY_E + CATEGORY_F + CATEGORY_G + CATEGORY_H

SCENARIOS_BY_NAME = {s["name"]: s for s in ALL_SCENARIOS}

CATEGORIES = {
    "A": CATEGORY_A,
    "B": CATEGORY_B,
    "C": CATEGORY_C,
    "D": CATEGORY_D,
    "E": CATEGORY_E,
    "F": CATEGORY_F,
    "G": CATEGORY_G,
    "H": CATEGORY_H,
}

def get_scenario(name: str) -> Dict:
    """Get a scenario by name or ID (e.g., 'A1' or full name)."""
    # Try exact match first
    if name in SCENARIOS_BY_NAME:
        return SCENARIOS_BY_NAME[name]
    # Try prefix match (e.g., "A1")
    for sname, scenario in SCENARIOS_BY_NAME.items():
        if sname.startswith(name + ":") or sname.startswith(name + " "):
            return scenario
    raise ValueError(f"Unknown scenario: {name}")


def get_scenario_list() -> List[Dict]:
    """Get summary list of all scenarios for UI display."""
    result = []
    for s in ALL_SCENARIOS:
        result.append({
            "name": s["name"],
            "category": s["category"],
            "sound_type": s["sound_type"],
            "expected_status": s["expected_status"],
            "notes": s.get("notes", ""),
        })
    return result
