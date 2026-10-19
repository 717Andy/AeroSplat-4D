"""Write one shard in the Phase 1 canonical layout, plus ground-truth sidecars.

``read_canonical`` and ``MultiCameraDataset`` load RGB, planar depth,
segmentation, ``K``, ``R|t``, timestamps, and flare-off frames without a
new code path. Actor poses, per-pixel velocity, wire polylines, coverage,
and the sun vector are sidecars. They are ignored by a loader that only
wants the images, and read when the files are present.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aerosplat.data.adapters.canonical import write_canonical
from aerosplat.geometry.wires import save_wires
from aerosplat.synth.palette import segmentation_color

DYNAMICS_FORMAT = "aerosplat.dynamics.v1"
SUN_FORMAT = "aerosplat.sun.v1"


def write_shard(
    root: Path,
    scene_id: str,
    cameras: dict[str, list[dict]],
    *,
    dynamics: dict,
    sun: dict,
    wires: list[dict],
    segmentation_ids: dict[str, dict],
    settings: dict,
    episode_meta: dict,
) -> None:
    """Write a self-contained canonical dataset plus sidecars."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    write_canonical(root, {scene_id: cameras})
    for name, frames in cameras.items():
        for index, frame in enumerate(frames):
            cam_dir = root / scene_id / name
            _save_array(cam_dir / "velocity" / f"{index:06d}.npy", frame["velocity"])
            _save_array(cam_dir / "instance" / f"{index:06d}.npy", frame["instance"])
            if frame.get("wire_coverage") is not None:
                _save_array(
                    cam_dir / "wire_coverage" / f"{index:06d}.npy",
                    np.asarray(frame["wire_coverage"], dtype=np.float16),
                )
    (root / "dynamics.json").write_text(json.dumps(_jsonable(dynamics)) + "\n")
    (root / "sun.json").write_text(json.dumps(_jsonable(sun)) + "\n")
    (root / "segmentation_ids.json").write_text(json.dumps(segmentation_ids, indent=2) + "\n")
    (root / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (root / "episode.json").write_text(json.dumps(_jsonable(episode_meta), indent=2) + "\n")
    wire_spans = []
    for span in wires:
        wire_spans.append(
            {
                "name": span["name"],
                "kind": span["kind"],
                "segmentation_id": int(span["segmentation_id"]),
                "diameter_m": float(span["diameter_m"]),
                "sag_m": float(span["sag_m"]),
                "points_enu": np.asarray(span["points_enu"], dtype=np.float64).tolist(),
            }
        )
    save_wires(root / scene_id / "wires.json", wire_spans)
    (root / "DONE").write_text("ok\n")


def dynamics_document(
    *,
    latitude_deg: float,
    longitude_deg: float,
    scene_id: str,
    frames: list[dict],
) -> dict:
    return {
        "format": DYNAMICS_FORMAT,
        "frame": "enu_z_up",
        "latitude_deg": float(latitude_deg),
        "longitude_deg": float(longitude_deg),
        "note": (
            "gt_velocity on a sample is set only when exactly one actor is moving, "
            "because the optional loss broadcasts one vector. Per-pixel velocity "
            "maps are the multi-actor target."
        ),
        "scenes": {scene_id: {"frames": frames}},
    }


def sun_document(
    *,
    latitude_deg: float,
    longitude_deg: float,
    frames: list[dict],
) -> dict:
    return {
        "format": SUN_FORMAT,
        "frame": "enu_z_up",
        "camera_frame": "opencv",
        "latitude_deg": float(latitude_deg),
        "longitude_deg": float(longitude_deg),
        "frames": frames,
    }


def segmentation_document(ids: dict[str, int]) -> dict:
    entries = {}
    for name, object_id in ids.items():
        color = segmentation_color(object_id)
        entries[name] = {"id": int(object_id), "color_rgb": list(color)}
    return {
        "palette": "airsim_pascal_bit_interleave",
        "note": (
            "Colosseum uses this palette for ImageType 5. Cosys-AirSim may use "
            "simGetSegmentationColorMap instead. cache.wire_colors should list "
            "the wire entry's color_rgb."
        ),
        "ids": entries,
    }


def estimate_shard_bytes(rig: dict, frames: int, *, supersample: int = 1) -> int:
    """Upper bound, uncompressed. PNG will be smaller. Supersampling is not stored."""
    del supersample
    total = 0
    for cam in rig["cameras"].values():
        height, width = int(cam["height"]), int(cam["width"])
        pixels = height * width
        # RGB, segmentation, flare-off, plus raw-sized slack for PNG headers.
        total += pixels * 3 * 3
        total += pixels * 4  # depth float32
        total += pixels * 4 * 3  # velocity float32 CHW
        total += pixels * 4  # instance int32
        total += pixels * 2  # coverage float16
    return int(total * frames)


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def _save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(array))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value
