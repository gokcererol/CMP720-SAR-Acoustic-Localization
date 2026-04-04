"""
Wave Propagation Model — Computes delay, attenuation, and multipath
for acoustic waves traveling from source to each sensor node.
"""

import math
import random
import numpy as np
from typing import List, Tuple, Dict, Optional


def get_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute distance in meters between two lat/lon points (flat-earth approx)."""
    dlat = (lat1 - lat2) * 111132.0
    dlon = (lon1 - lon2) * 111320.0 * math.cos(math.radians(lat1))
    return math.sqrt(dlat ** 2 + dlon ** 2)


def compute_attenuation(distance: float, ref_distance: float = 1.0) -> float:
    """
    Acoustic 1/r pressure decay with floor.
    Sound pressure (amplitude) decays as 1/r in free field.
    ref_distance is the distance at which amplitude = 1.0 (source level).
    At 70m: factor = 1/70 = 0.014 (too low), so we use 1/sqrt(r) for
    a more realistic simulation where sounds are still detectable at range.
    """
    if distance <= ref_distance:
        return 1.0
    # Use 1/sqrt(r) decay — gives ~0.12 at 70m (audible but attenuated)
    # This accounts for coherent acoustic sources being louder than point sources
    factor = math.sqrt(ref_distance / distance)
    return max(factor, 0.001)  # Floor at -60dB


def compute_travel_time(distance: float, speed_of_sound: float = 343.0) -> float:
    """Compute travel time in seconds."""
    if distance <= 0 or speed_of_sound <= 0:
        return 0.0
    return distance / speed_of_sound


def compute_debris_attenuation(terrain: str) -> float:
    """Additional attenuation factor for debris/rubble terrain."""
    if terrain == "rubble":
        # Random 5-25 dB loss converted to linear
        loss_db = random.uniform(5, 25)
        return 10 ** (-loss_db / 20.0)
    elif terrain == "urban_canyon":
        loss_db = random.uniform(3, 15)
        return 10 ** (-loss_db / 20.0)
    return 1.0  # flat terrain, no additional loss


def generate_multipath(waveform: np.ndarray, sample_rate: int,
                       max_reflections: int = 2,
                       reflection_loss_min: float = 0.1,
                       reflection_loss_max: float = 0.4,
                       delay_min_ms: float = 2.0,
                       delay_max_ms: float = 50.0) -> List[Tuple[np.ndarray, int]]:
    """
    Generate multipath reflections of a waveform.
    Returns list of (reflected_waveform, delay_in_samples).
    """
    reflections = []
    num_reflections = random.randint(0, max_reflections)
    for _ in range(num_reflections):
        delay_ms = random.uniform(delay_min_ms, delay_max_ms)
        delay_samples = int(delay_ms / 1000.0 * sample_rate)
        amp_factor = random.uniform(reflection_loss_min, reflection_loss_max)
        # Phase may invert on reflection
        phase = random.choice([1.0, -1.0])
        reflected = (waveform.astype(np.float64) * amp_factor * phase).astype(np.int16)
        reflections.append((reflected, delay_samples))
    return reflections


def propagate_to_node(source_waveform: np.ndarray, source_lat: float, source_lon: float,
                      node_lat: float, node_lon: float, sample_rate: int = 16000,
                      speed_of_sound: float = 343.0, terrain: str = "flat",
                      multipath_config: Optional[Dict] = None) -> Dict:
    """
    Compute the full propagation from source to a single node.
    Returns dict with:
      - waveform: attenuated waveform (int16)
      - delay_samples: propagation delay in samples
      - delay_seconds: propagation delay in seconds
      - distance: distance in meters
      - amplitude_factor: total attenuation factor
      - multipath: list of (waveform, delay_samples) for reflections
    """
    distance = get_distance_meters(source_lat, source_lon, node_lat, node_lon)
    travel_time = compute_travel_time(distance, speed_of_sound)
    delay_samples = int(travel_time * sample_rate)

    # Compute total attenuation
    geo_atten = compute_attenuation(distance)
    debris_atten = compute_debris_attenuation(terrain)
    total_atten = geo_atten * debris_atten

    # Apply attenuation to waveform
    attenuated = (source_waveform.astype(np.float64) * total_atten).astype(np.int16)

    # Generate multipath reflections
    multipath = []
    if multipath_config and multipath_config.get("enabled", False):
        multipath = generate_multipath(
            attenuated, sample_rate,
            max_reflections=multipath_config.get("max_reflections", 2),
            reflection_loss_min=multipath_config.get("reflection_loss_min", 0.1),
            reflection_loss_max=multipath_config.get("reflection_loss_max", 0.4),
            delay_min_ms=multipath_config.get("delay_min_ms", 2.0),
            delay_max_ms=multipath_config.get("delay_max_ms", 50.0),
        )

    return {
        "waveform": attenuated,
        "delay_samples": delay_samples,
        "delay_seconds": travel_time,
        "distance": distance,
        "amplitude_factor": total_atten,
        "multipath": multipath,
    }


def inject_into_stream(stream: np.ndarray, waveform: np.ndarray,
                       offset_samples: int) -> np.ndarray:
    """Inject a waveform (and its multipath copies) into a continuous audio stream buffer."""
    result = stream.copy().astype(np.float64)
    # Inject direct path
    end = min(offset_samples + len(waveform), len(result))
    if offset_samples < len(result) and offset_samples >= 0:
        actual_len = end - offset_samples
        result[offset_samples:end] += waveform[:actual_len].astype(np.float64)
    # Clip to int16 range
    result = np.clip(result, -32768, 32767)
    return result.astype(np.int16)
