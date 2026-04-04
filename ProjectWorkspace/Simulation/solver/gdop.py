"""
GDOP Calculator — Geometric Dilution of Precision computation
and confidence ellipse generation for map visualization.
"""
print("      - gdop.py: top-level reached", flush=True)

import numpy as np
import math
from typing import Dict, List, Tuple


def compute_gdop(node_positions: Dict[int, Tuple[float, float]],
                 source_x: float, source_y: float) -> float:
    """
    Compute GDOP (Geometric Dilution of Precision) for a position estimate.
    Lower GDOP = better geometry = more precise position estimate.
    GDOP < 3: Excellent | GDOP 3-6: Fair | GDOP > 6: Poor
    """
    nodes = list(node_positions.values())
    n = len(nodes)
    if n < 3:
        return 99.0

    # Build geometry matrix H
    H = np.zeros((n, 2))
    for i, (nx, ny) in enumerate(nodes):
        dx = source_x - nx
        dy = source_y - ny
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist < 0.01:
            dist = 0.01
        H[i, 0] = dx / dist
        H[i, 1] = dy / dist

    try:
        HTH = H.T @ H
        Q = np.linalg.inv(HTH)
        gdop = math.sqrt(Q[0, 0] + Q[1, 1])
        return min(gdop, 99.0)
    except np.linalg.LinAlgError:
        return 99.0


def compute_confidence_ellipse(node_positions: Dict[int, Tuple[float, float]],
                               source_x: float, source_y: float,
                               sigma_tdoa: float = 0.001) -> Dict:
    """
    Compute a confidence ellipse for the position estimate.
    Returns ellipse parameters for map visualization.
    """
    nodes = list(node_positions.values())
    n = len(nodes)
    if n < 3:
        return {"semi_major": 50, "semi_minor": 50, "rotation_deg": 0}

    H = np.zeros((n, 2))
    for i, (nx, ny) in enumerate(nodes):
        dx = source_x - nx
        dy = source_y - ny
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist < 0.01:
            dist = 0.01
        H[i, 0] = dx / dist
        H[i, 1] = dy / dist

    try:
        HTH = H.T @ H
        Q = np.linalg.inv(HTH) * (sigma_tdoa ** 2)

        # Eigenvalue decomposition for ellipse axes
        eigenvalues, eigenvectors = np.linalg.eigh(Q)
        eigenvalues = np.abs(eigenvalues)

        # Semi-axes (multiply by chi-squared critical value for 90% confidence)
        chi2_90 = 4.605  # 90% confidence for 2 DOF
        chi2_50 = 1.386  # 50% confidence

        semi_major_90 = math.sqrt(max(eigenvalues) * chi2_90)
        semi_minor_90 = math.sqrt(min(eigenvalues) * chi2_90)
        semi_major_50 = math.sqrt(max(eigenvalues) * chi2_50)
        semi_minor_50 = math.sqrt(min(eigenvalues) * chi2_50)

        # Rotation angle
        angle = math.degrees(math.atan2(eigenvectors[1, -1], eigenvectors[0, -1]))

        return {
            "semi_major_90": float(semi_major_90),
            "semi_minor_90": float(semi_minor_90),
            "semi_major_50": float(semi_major_50),
            "semi_minor_50": float(semi_minor_50),
            "rotation_deg": float(angle),
            "cep50_m": float(semi_major_50),
            "cep90_m": float(semi_major_90),
        }

    except np.linalg.LinAlgError:
        return {
            "semi_major_90": 50.0, "semi_minor_90": 50.0,
            "semi_major_50": 25.0, "semi_minor_50": 25.0,
            "rotation_deg": 0, "cep50_m": 25.0, "cep90_m": 50.0,
        }


def gdop_color(gdop: float) -> str:
    """Return color code based on GDOP value."""
    if gdop < 3:
        return "#22c55e"  # green
    elif gdop < 6:
        return "#eab308"  # yellow
    else:
        return "#ef4444"  # red


def gdop_label(gdop: float) -> str:
    """Return human-readable GDOP quality label."""
    if gdop < 2:
        return "Excellent"
    elif gdop < 3:
        return "Good"
    elif gdop < 6:
        return "Fair"
    elif gdop < 10:
        return "Poor"
    else:
        return "Very Poor"
