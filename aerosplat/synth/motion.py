"""Trajectories, yaw, and finite-difference velocity in the flight frame.

Positions are ENU meters. The AirSim client receives the NED conversion at
the spawn boundary. Velocity is meters per second in ENU, the same vector
``velocity_from_positions`` returns for a pair of origins.
"""

from __future__ import annotations

import numpy as np

from aerosplat.geometry.coordinates import enu_to_ned


def velocities_from_positions(positions: np.ndarray, dt: float) -> np.ndarray:
    """World-frame m/s. Interior samples use a central difference.

    The first and last samples use a one-sided difference. A single sample
    is the zero vector. ``dt`` is seconds and must be positive.
    """
    pts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if dt <= 0:
        raise ValueError("dt must be positive")
    velocity = np.zeros_like(pts)
    count = len(pts)
    if count < 2:
        return velocity
    step = float(dt)
    velocity[0] = (pts[1] - pts[0]) / step
    velocity[-1] = (pts[-1] - pts[-2]) / step
    if count > 2:
        velocity[1:-1] = (pts[2:] - pts[:-2]) / (2.0 * step)
    return velocity


def sample_polyline(waypoints: np.ndarray, speed_mps: float, times: np.ndarray) -> np.ndarray:
    """Arc-length sample. Past the end of the path the position holds."""
    pts = np.asarray(waypoints, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 1:
        return np.repeat(pts, len(times), axis=0)
    if speed_mps < 0:
        raise ValueError("speed_mps must be >= 0")
    segment = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    total = float(cumulative[-1])
    distance = np.clip(np.asarray(times, dtype=np.float64) * float(speed_mps), 0.0, total)
    out = np.empty((len(distance), 3), dtype=np.float64)
    for index, traveled in enumerate(distance):
        if traveled >= total:
            out[index] = pts[-1]
            continue
        segment_index = int(np.searchsorted(cumulative, traveled, side="right") - 1)
        segment_index = min(max(segment_index, 0), len(pts) - 2)
        span = float(segment[segment_index])
        if span < 1e-12:
            weight = 0.0
        else:
            weight = (traveled - cumulative[segment_index]) / span
        out[index] = (1.0 - weight) * pts[segment_index] + weight * pts[segment_index + 1]
    return out


def path_tangent(waypoints: np.ndarray, speed_mps: float, times: np.ndarray) -> np.ndarray:
    """Unit ENU tangent. A hold at the end of the path repeats the last direction."""
    pts = np.asarray(waypoints, dtype=np.float64).reshape(-1, 3)
    count = len(times)
    if len(pts) < 2:
        return np.tile(np.array([1.0, 0.0, 0.0]), (count, 1))
    epsilon = 1e-3
    clock = np.asarray(times, dtype=np.float64)
    ahead = sample_polyline(pts, speed_mps, clock + epsilon)
    behind = sample_polyline(pts, speed_mps, np.maximum(clock - epsilon, 0.0))
    direction = ahead - behind
    tangent = np.zeros_like(direction)
    for index, vector in enumerate(direction):
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            previous = tangent[index - 1] if index else np.array([1.0, 0.0, 0.0])
            tangent[index] = previous
        else:
            tangent[index] = vector / norm
    return tangent


def ned_yaw_quaternion(yaw_rad: float) -> np.ndarray:
    """Body-to-NED quaternion, wxyz. Yaw 0 points the nose north, positive toward east."""
    half = 0.5 * float(yaw_rad)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def yaw_rad_from_enu_direction(direction_enu: np.ndarray) -> float:
    """Aerospace yaw: atan2(east, north), which is atan2(ENU x, ENU y)."""
    direction = np.asarray(direction_enu, dtype=np.float64).reshape(3)
    ned = enu_to_ned(direction)
    return float(np.arctan2(ned[1], ned[0]))


def quat_from_enu_direction(direction_enu: np.ndarray) -> np.ndarray:
    return ned_yaw_quaternion(yaw_rad_from_enu_direction(direction_enu))


def enu_position_to_ned(position_enu: np.ndarray) -> np.ndarray:
    return enu_to_ned(np.asarray(position_enu, dtype=np.float64).reshape(3))


def quat_align_body_x(direction_ned: np.ndarray) -> np.ndarray:
    """Body-to-NED quaternion, wxyz, that maps body +X onto ``direction_ned``."""
    direction = np.asarray(direction_ned, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    target = direction / norm
    source = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < 1e-8:
        if dot > 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return np.array([0.0, 0.0, 1.0, 0.0])
    axis = cross / cross_norm
    half = 0.5 * float(np.arccos(dot))
    return np.array(
        [np.cos(half), axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half)],
        dtype=np.float64,
    )
