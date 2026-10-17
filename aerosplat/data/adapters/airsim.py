"""AirSim-family recordings (Colosseum and Cosys-AirSim).

Two inputs are supported:

* Canonical recordings produced by :mod:`aerosplat.capture.airsim`. Prefer
  these. They already store ENU/OpenCV extrinsics per camera.
* Legacy ``airsim_rec.txt`` plus a ``settings.json``. The vehicle pose in that
  log is NED, quaternion wxyz, and each camera is a rigid body mount. This
  module converts both into the flight frame.

Colosseum (https://github.com/CodexLabsLLC/Colosseum) is archived. Cosys-AirSim
(https://github.com/Cosys-Lab/Cosys-AirSim) is the maintained simulator. Pass
``simulator`` or let ``SettingsVersion`` decide (1.2 Colosseum, 2.0 Cosys).
Extra Cosys columns (latitude, accelerometer, and so on) are ignored as long
as the pose header is present. ``External: true`` cameras are rejected: their
pose is Unreal world, or local NED from the start pose when ``ExternalLocal``
is set, which is not the body mount composed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aerosplat.airsim_family import (
    DEPTH_PERSPECTIVE,
    DEPTH_PLANAR,
    DEPTH_VIS,
    DISPARITY_NORMALIZED,
    NON_METRIC_DEPTH,
    SCENE,
    SEGMENTATION,
    SimulatorProfile,
    resolve_simulator,
)
from aerosplat.data.errors import DataError
from aerosplat.data.records import FrameRecord
from aerosplat.geometry.coordinates import airsim_mount_to_w2c, intrinsics_from_hfov

IMAGE_SCENE = SCENE
IMAGE_DEPTH_PLANAR = DEPTH_PLANAR
IMAGE_DEPTH_PERSPECTIVE = DEPTH_PERSPECTIVE

_TIMESTAMP_SCALE = {
    "seconds": 1.0,
    "milliseconds": 1e-3,
    "microseconds": 1e-6,
    "nanoseconds": 1e-9,
}


def read_airsim_raw(
    recording: Path,
    settings: Path,
    *,
    timestamp_unit: str = "nanoseconds",
    depth_kind: str | None = None,
    simulator: str | None = None,
) -> dict[str, list[FrameRecord]]:
    """Parse ``airsim_rec.txt`` into per-camera :class:`FrameRecord` streams.

    ``simulator`` is ``"colosseum"`` or ``"cosys"``. When omitted, SettingsVersion
    selects the profile. The pose columns are the same on both; Cosys may append
    further columns, which the header-keyed parser skips.
    """
    recording = Path(recording)
    settings = Path(settings)
    if timestamp_unit not in _TIMESTAMP_SCALE:
        raise DataError(
            f"timestamp_unit must be one of {sorted(_TIMESTAMP_SCALE)}, got {timestamp_unit!r}"
        )
    mounts = parse_settings(settings, simulator=simulator)
    depth_kind = _resolve_depth_kind(mounts, depth_kind)
    rows = _read_recording_table(recording)
    scale = _TIMESTAMP_SCALE[timestamp_unit]
    streams: dict[str, list[FrameRecord]] = {name: [] for name in mounts}
    for row_index, row in enumerate(rows):
        names = _image_names(row, list(mounts))
        position = np.array([row["POS_X"], row["POS_Y"], row["POS_Z"]], dtype=np.float64)
        quat = np.array([row["Q_W"], row["Q_X"], row["Q_Y"], row["Q_Z"]], dtype=np.float64)
        timestamp = float(row["TimeStamp"]) * scale
        for name in names:
            mount = mounts[name]
            image_path = _resolve(recording.parent, row["images"][name])
            depth_path = _find_depth(image_path)
            R, t = airsim_mount_to_w2c(
                position,
                quat,
                cam_position_body=mount["position"],
                pitch_deg=mount["pitch"],
                roll_deg=mount["roll"],
                yaw_deg=mount["yaw"],
            )
            streams[name].append(
                FrameRecord(
                    frame_index=row_index,
                    timestamp=timestamp,
                    image_path=image_path,
                    depth_path=depth_path,
                    K=mount["K"],
                    R=R,
                    t=t,
                    depth_kind=depth_kind,
                )
            )
    missing = [name for name, frames in streams.items() if not frames]
    if missing:
        raise DataError(f"{recording} produced no frames for cameras {missing}")
    return streams


def parse_settings(path: Path, simulator: str | None = None) -> dict[str, dict]:
    """Return camera mounts in settings-file order.

    Each value has position (body, meters), pitch/roll/yaw degrees, K, and the
    scene-image resolution used to build K. Cameras with ``External: true`` are
    rejected: that pose is not a body-frame mount.
    """
    try:
        data = json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise DataError(f"{path} is not valid JSON") from exc
    version = float(data.get("SettingsVersion") or 0)
    profile = resolve_simulator(simulator, version)
    vehicles = data.get("Vehicles") or {}
    if not vehicles:
        raise DataError(f"{path} has no Vehicles entry")
    vehicle = next(iter(vehicles.values()))
    cameras = vehicle.get("Cameras") or {}
    if not cameras:
        raise DataError(f"{path} vehicle has no Cameras")
    move_origin = bool(data.get("MoveWorldOrigin", False))
    mounts: dict[str, dict] = {}
    for name, cam in cameras.items():
        _reject_external(path, name, cam, profile, move_origin)
        scene = _capture(cam, IMAGE_SCENE)
        if scene is None:
            raise DataError(f"{path} camera {name} has no Scene CaptureSettings (ImageType 0)")
        width = int(scene["Width"])
        height = int(scene["Height"])
        fov = float(scene.get("FOV_Degrees", data.get("CameraDefaults", {}).get("FOV_Degrees", 90)))
        has_planar = _capture(cam, IMAGE_DEPTH_PLANAR) is not None
        has_perspective = _capture(cam, IMAGE_DEPTH_PERSPECTIVE) is not None
        non_metric = [
            image_type
            for image_type in (DEPTH_VIS, DISPARITY_NORMALIZED)
            if _capture(cam, image_type) is not None
        ]
        mounts[name] = {
            "position": np.array(
                [float(cam.get("X", 0.0)), float(cam.get("Y", 0.0)), float(cam.get("Z", 0.0))],
                dtype=np.float64,
            ),
            "pitch": float(cam.get("Pitch", 0.0)),
            "roll": float(cam.get("Roll", 0.0)),
            "yaw": float(cam.get("Yaw", 0.0)),
            "width": width,
            "height": height,
            "fov_degrees": fov,
            "K": intrinsics_from_hfov(width, height, fov),
            "has_planar": has_planar,
            "has_perspective": has_perspective,
            "non_metric_depth": non_metric,
            "has_segmentation": _capture(cam, SEGMENTATION) is not None,
            "simulator": profile.name,
            "segmentation_id_space": profile.segmentation_id_space,
        }
    return mounts


def _reject_external(
    path: Path,
    name: str,
    camera: dict,
    profile: SimulatorProfile,
    move_origin: bool,
) -> None:
    if not bool(camera.get("External", False)):
        return
    external_local = bool(camera.get("ExternalLocal", True))
    if profile.name == "cosys":
        frame = "local NED measured from the vehicle start" if external_local else "Unreal NED"
        origin = (
            " MoveWorldOrigin is true, so the Unreal origin is the player start."
            if move_origin
            else ""
        )
        detail = (
            f"Cosys-AirSim External cameras report {frame}, which is a different "
            f"frame from the vehicle body mount.{origin}"
        )
    else:
        detail = (
            "External cameras are posed in the world, which is a different "
            "frame from the body mount."
        )
    raise DataError(
        f"{path} camera {name} has External true. {detail} "
        "Set External false so X/Y/Z/Pitch/Roll/Yaw stay in the vehicle body frame."
    )


def _capture(camera: dict, image_type: int) -> dict | None:
    for entry in camera.get("CaptureSettings") or []:
        if int(entry.get("ImageType", -1)) == image_type:
            return entry
    return None


def _resolve_depth_kind(mounts: dict[str, dict], requested: str | None) -> str:
    """Prefer DepthPlanar. Perspective-only rigs are Euclidean range.

    DepthVis and DisparityNormalized never become the depth channel. A rig that
    records only those types fails with the reason, instead of being labeled
    planar Z.
    """
    if requested is not None and requested not in {"planar_z", "euclidean"}:
        raise DataError(f"depth_kind must be 'planar_z' or 'euclidean', got {requested!r}")
    labels = {name: _available_depth(mount) for name, mount in mounts.items()}
    metric = {label for label in labels.values() if label in {"planar_z", "euclidean", "both"}}
    if requested == "planar_z" and metric == {"euclidean"}:
        raise DataError(
            "depth_kind planar_z was requested, but settings only capture "
            "DepthPerspective (ImageType 2, Euclidean range)."
        )
    if requested == "euclidean" and metric == {"planar_z"}:
        raise DataError(
            "depth_kind euclidean was requested, but settings only capture "
            "DepthPlanar (ImageType 1, camera-plane Z)."
        )
    disagrees = "planar_z" in metric and "euclidean" in metric
    if disagrees and requested is None:
        raise DataError(
            "cameras disagree on DepthPlanar vs DepthPerspective. "
            "Pass depth_kind='planar_z' or 'euclidean'."
        )
    if requested is not None:
        return requested
    if "both" in metric or "planar_z" in metric:
        return "planar_z"
    if metric == {"euclidean"}:
        return "euclidean"
    non_metric_names = [name for name, label in labels.items() if label == "non_metric"]
    if non_metric_names:
        sample = mounts[non_metric_names[0]]
        image_type = sample["non_metric_depth"][0]
        raise DataError(f"cameras {non_metric_names} capture {NON_METRIC_DEPTH[image_type]}")
    return "planar_z"


def _available_depth(mount: dict) -> str:
    if mount["has_planar"] and mount["has_perspective"]:
        return "both"
    if mount["has_planar"]:
        return "planar_z"
    if mount["has_perspective"]:
        return "euclidean"
    if mount["non_metric_depth"]:
        return "non_metric"
    return "none"


def _read_recording_table(path: Path) -> list[dict]:
    if not path.is_file():
        raise DataError(f"missing AirSim recording {path}")
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        raise DataError(f"{path} has no data rows")
    header_line = lines[0]
    delimiter = "\t" if "\t" in header_line else ","
    header = [cell.strip() for cell in header_line.split(delimiter)]
    required = ["TimeStamp", "POS_X", "POS_Y", "POS_Z", "Q_W", "Q_X", "Q_Y", "Q_Z", "ImageFile"]
    missing = [name for name in required if name not in header]
    if missing:
        raise DataError(f"{path} header is missing {missing}")
    rows = []
    for line_no, line in enumerate(lines[1:], start=2):
        cells = [cell.strip() for cell in line.split(delimiter)]
        if len(cells) != len(header):
            raise DataError(f"{path}:{line_no} has {len(cells)} columns, header has {len(header)}")
        raw = dict(zip(header, cells, strict=True))
        try:
            numeric = {key: float(raw[key]) for key in required if key != "ImageFile"}
        except ValueError as exc:
            raise DataError(f"{path}:{line_no} has a non-numeric pose or timestamp") from exc
        images = _split_images(raw["ImageFile"])
        rows.append({**numeric, "images_list": images, "images": {}})
    return rows


def _split_images(field: str) -> list[str]:
    parts = [part.strip() for part in field.split(";") if part.strip()]
    if not parts:
        raise DataError(f"ImageFile entry is empty: {field!r}")
    return parts


def _image_names(row: dict, camera_order: list[str]) -> list[str]:
    paths = row["images_list"]
    if len(paths) != len(camera_order):
        raise DataError(
            f"timestamp {row['TimeStamp']}: {len(paths)} image paths for "
            f"{len(camera_order)} cameras ({camera_order}). "
            "ImageFile must be ';'-separated in settings.json camera order."
        )
    row["images"] = dict(zip(camera_order, paths, strict=True))
    return camera_order


def _resolve(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    return root / path


def _find_depth(image_path: Path) -> Path | None:
    """Depth next to an RGB frame, if the recording saved one.

    Search order: ``<stem>.pfm``, ``<stem>_depth.npy``, ``<stem>.npy``,
    ``../depth/<stem>.npy``, ``../depth/<stem>.pfm``.
    """
    candidates = [
        image_path.with_suffix(".pfm"),
        image_path.with_name(image_path.stem + "_depth.npy"),
        image_path.with_suffix(".npy"),
        image_path.parent.parent / "depth" / (image_path.stem + ".npy"),
        image_path.parent.parent / "depth" / (image_path.stem + ".pfm"),
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate != image_path:
            return candidate
    return None
