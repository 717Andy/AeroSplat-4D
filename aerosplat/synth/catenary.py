"""Catenary centerlines for power lines and crane cables.

``sag_m`` is the vertical drop of the midpoint below the straight chord, in
meters, positive downward in the flight frame (ENU, Z up). Level supports
produce the physical catenary ``z = a cosh(x / a)``. Unequal supports use that
same catenary as a deviation from the chord, so the midpoint still drops by
``sag_m`` and the endpoints stay put. The lowest point of a real unequal-height
catenary is not always the midpoint; this definition keeps the YAML sag
unambiguous.

A sag of 0 is a straight line. The parameter ``a`` is solved with Newton from
``a (cosh(L / 2a) - 1) = sag``.
"""

from __future__ import annotations

import math

import numpy as np


def catenary_parameter(span_m: float, sag_m: float) -> float:
    """Return the catenary scale ``a`` (meters) for a level span."""
    if span_m <= 0:
        raise ValueError("span must be positive")
    if sag_m < 0:
        raise ValueError("sag must be >= 0")
    if sag_m == 0:
        return math.inf
    # Parabola approximation a ≈ L^2 / (8 sag) is the Newton start.
    estimate = max(span_m * span_m / (8.0 * sag_m), 1e-3)
    value = estimate
    for _ in range(40):
        half = span_m / (2.0 * value)
        # Overflow guard: a tiny ``a`` makes cosh explode. Clamp and retry.
        if half > 20.0:
            value *= 2.0
            continue
        cosh_half = math.cosh(half)
        sinh_half = math.sinh(half)
        residual = value * (cosh_half - 1.0) - sag_m
        derivative = (cosh_half - 1.0) - half * sinh_half
        if abs(derivative) < 1e-14:
            break
        value = value - residual / derivative
        if value < 1e-4:
            value = 1e-4
        if abs(residual) < 1e-10 * max(sag_m, 1e-6):
            break
    return float(value)


def catenary_polyline(
    start: np.ndarray,
    end: np.ndarray,
    sag_m: float,
    samples: int,
) -> np.ndarray:
    """Flight-frame points along the cable, including both endpoints.

    ``samples`` is the number of points. Two points are the endpoints only.
    """
    if samples < 2:
        raise ValueError("samples must be >= 2")
    if sag_m < 0:
        raise ValueError("sag_m must be >= 0")
    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)
    if sag_m == 0.0:
        weights = np.linspace(0.0, 1.0, samples, dtype=np.float64)[:, None]
        return (1.0 - weights) * start + weights * end
    chord = end - start
    span = float(np.linalg.norm(chord[:2]))
    if span < 1e-8:
        # A vertical crane cable. Sag is a lateral bulge in the horizontal
        # plane so the curve is not a second copy of the chord.
        return _vertical_cable(start, end, sag_m, samples)
    parameter = catenary_parameter(span, sag_m)
    weights = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    points = np.empty((samples, 3), dtype=np.float64)
    for index, weight in enumerate(weights):
        chord_point = (1.0 - weight) * start + weight * end
        offset = (weight - 0.5) * span
        drop = parameter * (math.cosh(span / (2.0 * parameter)) - math.cosh(offset / parameter))
        chord_point[2] -= drop
        points[index] = chord_point
    points[0] = start
    points[-1] = end
    return points


def _vertical_cable(start: np.ndarray, end: np.ndarray, sag_m: float, samples: int) -> np.ndarray:
    """Hang a vertical cable with a horizontal bow of ``sag_m`` at mid-height."""
    weights = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    points = np.empty((samples, 3), dtype=np.float64)
    bow = np.array([sag_m, 0.0, 0.0], dtype=np.float64)
    for index, weight in enumerate(weights):
        bulge = 4.0 * weight * (1.0 - weight)
        points[index] = (1.0 - weight) * start + weight * end + bulge * bow
    points[0] = start
    points[-1] = end
    return points
