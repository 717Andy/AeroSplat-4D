"""Canonical multi-camera recording written by the fixture and the capture script.

Layout::

    root/
      dataset.json
      <scene_id>/
        <camera_name>/
          frames.jsonl
          images/000000.png
          depth/000000.npy      # optional

``frames.jsonl`` stores capture-time world-to-camera extrinsics in the ENU /
OpenCV flight frame, a timestamp in seconds, and paths relative to the camera
directory. This is the format every other adapter can be converted into.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aerosplat.data.errors import DataError
from aerosplat.data.io import save_depth_npy, save_rgb
from aerosplat.data.records import FrameRecord

FORMAT = "aerosplat.multicam.v1"


def write_canonical(
    root: Path,
    scenes: dict[str, dict[str, list[dict]]],
    *,
    depth_kind: str = "planar_z",
) -> None:
    """Write scenes.

    Each frame dict needs ``timestamp``, ``image`` (uint8 HWC), ``K``, ``R``,
    ``t``, and optional ``depth`` (float HxW, meters).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    scene_ids = list(scenes)
    manifest = {
        "format": FORMAT,
        "world_frame": "enu_z_up",
        "camera_frame": "opencv",
        "extrinsics": "world_to_camera",
        "depth_kind": depth_kind,
        "scenes": scene_ids,
    }
    (root / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for scene_id, cameras in scenes.items():
        for name, frames in cameras.items():
            cam_dir = root / scene_id / name
            (cam_dir / "images").mkdir(parents=True, exist_ok=True)
            lines = []
            for index, frame in enumerate(frames):
                image_rel = f"images/{index:06d}.png"
                save_rgb(cam_dir / image_rel, frame["image"])
                depth_rel = None
                if frame.get("depth") is not None:
                    depth_rel = f"depth/{index:06d}.npy"
                    save_depth_npy(cam_dir / depth_rel, frame["depth"])
                record = {
                    "index": index,
                    "timestamp": float(frame["timestamp"]),
                    "image": image_rel,
                    "depth": depth_rel,
                    "K": np.asarray(frame["K"], dtype=np.float64).reshape(3, 3).tolist(),
                    "R": np.asarray(frame["R"], dtype=np.float64).reshape(3, 3).tolist(),
                    "t": np.asarray(frame["t"], dtype=np.float64).reshape(3).tolist(),
                }
                if frame.get("segmentation") is not None:
                    seg_rel = f"segmentation/{index:06d}.png"
                    save_rgb(cam_dir / seg_rel, frame["segmentation"])
                    record["segmentation"] = seg_rel
                if frame.get("clean_image") is not None:
                    clean_rel = f"clean/{index:06d}.png"
                    save_rgb(cam_dir / clean_rel, frame["clean_image"])
                    record["clean_image"] = clean_rel
                lines.append(json.dumps(record))
            (cam_dir / "frames.jsonl").write_text("\n".join(lines) + "\n")


def read_canonical(root: Path) -> dict[str, dict[str, list[FrameRecord]]]:
    root = Path(root)
    manifest_path = root / "dataset.json"
    if not manifest_path.is_file():
        raise DataError(f"{root} has no dataset.json (not a canonical AeroSplat recording)")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != FORMAT:
        raise DataError(f"{manifest_path} format is {manifest.get('format')!r}, expected {FORMAT}")
    if manifest.get("world_frame") != "enu_z_up" or manifest.get("camera_frame") != "opencv":
        raise DataError(f"{manifest_path} is not in the ENU/OpenCV flight frame")
    depth_kind = manifest.get("depth_kind", "planar_z")
    if depth_kind not in {"planar_z", "euclidean", "planar_z_meters"}:
        raise DataError(f"unsupported depth_kind {depth_kind!r}")
    if depth_kind == "planar_z_meters":
        depth_kind = "planar_z"
    scenes: dict[str, dict[str, list[FrameRecord]]] = {}
    for scene_id in manifest["scenes"]:
        scene_dir = root / scene_id
        if not scene_dir.is_dir():
            raise DataError(f"missing scene directory {scene_dir}")
        cameras: dict[str, list[FrameRecord]] = {}
        for cam_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
            jsonl = cam_dir / "frames.jsonl"
            if not jsonl.is_file():
                continue
            cameras[cam_dir.name] = _read_camera(jsonl, depth_kind)
        if not cameras:
            raise DataError(f"scene {scene_id} has no camera streams")
        scenes[scene_id] = cameras
    return scenes


def _read_camera(jsonl: Path, depth_kind: str) -> list[FrameRecord]:
    frames: list[FrameRecord] = []
    for line_no, line in enumerate(jsonl.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataError(f"{jsonl}:{line_no} is not valid JSON") from exc
        depth_rel = record.get("depth")
        index = int(record["index"])
        seg_rel = record.get("segmentation")
        if not seg_rel:
            candidate = Path("segmentation") / f"{index:06d}.png"
            if (jsonl.parent / candidate).is_file():
                seg_rel = str(candidate)
        clean_rel = record.get("clean_image")
        frames.append(
            FrameRecord(
                frame_index=index,
                timestamp=float(record["timestamp"]),
                image_path=jsonl.parent / record["image"],
                depth_path=(jsonl.parent / depth_rel) if depth_rel else None,
                K=np.asarray(record["K"], dtype=np.float64),
                R=np.asarray(record["R"], dtype=np.float64),
                t=np.asarray(record["t"], dtype=np.float64),
                depth_kind=depth_kind,
                segmentation_path=(jsonl.parent / seg_rel) if seg_rel else None,
                clean_image_path=(jsonl.parent / clean_rel) if clean_rel else None,
            )
        )
    if not frames:
        raise DataError(f"{jsonl} has no frames")
    return frames
