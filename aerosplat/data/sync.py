"""Timestamp synchronization across cameras with unequal frame rates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aerosplat.data.errors import DataError
from aerosplat.data.records import FrameRecord, SyncConfig
from aerosplat.geometry.camera import PoseSample, interpolate_w2c


@dataclass
class CameraMatch:
    frame: FrameRecord
    R_query: np.ndarray
    t_query: np.ndarray
    sync: str


@dataclass
class SyncedStamp:
    scene_id: str
    timestamp: float
    cameras: dict[str, CameraMatch]


def nearest_index(times: np.ndarray, query: float) -> int:
    """Index of the closest timestamp. Ties keep the earlier frame."""
    if times.size == 0:
        raise ValueError("empty timestamp array")
    pos = int(np.searchsorted(times, query, side="left"))
    if pos <= 0:
        return 0
    if pos >= times.size:
        return int(times.size - 1)
    before = pos - 1
    if abs(float(times[pos]) - query) < abs(float(times[before]) - query):
        return pos
    return before


def match_stream(
    frames: list[FrameRecord],
    query_time: float,
    policy: str,
    max_image_dt: float,
) -> CameraMatch | None:
    """Pick one frame for ``query_time``, or None if no image is close enough."""
    if not frames:
        return None
    ordered = sorted(frames, key=lambda f: f.timestamp)
    times = np.array([f.timestamp for f in ordered], dtype=np.float64)
    index = nearest_index(times, query_time)
    frame = ordered[index]
    image_dt = abs(frame.timestamp - query_time)
    if image_dt > max_image_dt:
        return None
    poses = [PoseSample(f.timestamp, f.R, f.t) for f in ordered]
    if policy == "nearest":
        status = "exact" if image_dt <= 1e-9 else "nearest"
        return CameraMatch(frame, frame.R.copy(), frame.t.copy(), status)
    R_query, t_query, status = interpolate_w2c(poses, query_time)
    if image_dt <= 1e-9 and status == "exact":
        sync = "exact"
    else:
        sync = status
    return CameraMatch(frame, R_query, t_query, sync)


def synchronize(
    scene_id: str,
    streams: dict[str, list[FrameRecord]],
    config: SyncConfig,
) -> list[SyncedStamp]:
    """Align ``streams`` onto the reference camera's timestamps.

    Camera order follows ``config.camera_order`` when set, otherwise the
    reference camera first and then the remaining names in insertion order.
    """
    if config.reference_camera not in streams:
        have = ", ".join(streams) or "<none>"
        raise DataError(
            f"scene {scene_id}: reference camera {config.reference_camera!r} "
            f"is not in the recording (have {have})"
        )
    if config.camera_order:
        missing = [name for name in config.camera_order if name not in streams]
        if missing:
            raise DataError(f"scene {scene_id}: missing cameras {missing}")
        names = list(config.camera_order)
    else:
        names = [config.reference_camera] + [
            name for name in streams if name != config.reference_camera
        ]
    stamps: list[SyncedStamp] = []
    for ref in sorted(streams[config.reference_camera], key=lambda f: f.timestamp):
        matches: dict[str, CameraMatch] = {}
        complete = True
        for name in names:
            match = match_stream(streams[name], ref.timestamp, config.policy, config.max_image_dt)
            if match is None:
                complete = False
                if config.require_all_cameras:
                    break
                continue
            matches[name] = match
        if config.require_all_cameras and not complete:
            continue
        if not matches:
            continue
        stamps.append(SyncedStamp(scene_id, ref.timestamp, matches))
    return stamps
