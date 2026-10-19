"""Optional dynamics and geodesy sidecars next to a canonical recording.

A recording without ``dynamics.json`` loads exactly as before. A synthetic
shard written by ``aerosplat.synth`` stores latitude, longitude, and one
flight-frame velocity per timestamp. The velocity is present only when a
single actor is moving. Per-pixel maps live beside each camera and are
loaded onto ``CameraFrame.velocity`` when the file exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT = "aerosplat.dynamics.v1"


@dataclass
class DynamicsTable:
    latitude_deg: float | None
    longitude_deg: float | None
    velocity_by_time: dict[tuple[str, int], np.ndarray]


def load_dynamics(root: Path) -> DynamicsTable | None:
    path = Path(root) / "dynamics.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if payload.get("format") != FORMAT:
        raise ValueError(f"{path} format is {payload.get('format')!r}, expected {FORMAT}")
    table: dict[tuple[str, int], np.ndarray] = {}
    for scene_id, scene in (payload.get("scenes") or {}).items():
        for frame in scene.get("frames", []):
            velocity = frame.get("gt_velocity_enu")
            if velocity is None:
                continue
            table[(str(scene_id), _time_key(float(frame["timestamp"])))] = np.asarray(
                velocity, dtype=np.float32
            ).reshape(3)
    latitude = payload.get("latitude_deg")
    longitude = payload.get("longitude_deg")
    return DynamicsTable(
        latitude_deg=None if latitude is None else float(latitude),
        longitude_deg=None if longitude is None else float(longitude),
        velocity_by_time=table,
    )


def velocity_at(table: DynamicsTable, scene_id: str, timestamp: float) -> np.ndarray | None:
    return table.velocity_by_time.get((scene_id, _time_key(timestamp)))


def _time_key(timestamp: float) -> int:
    return int(round(float(timestamp) * 1_000_000.0))
