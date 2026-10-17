"""MatrixCity aerial loader.

Per-block ``transforms.json`` files store a NeRF cam2world (camera x right, y
up, z backward). In those files the translation is meters but the 3x3 rotation
was written at 1/100 of a real rotation and has to be multiplied by 100 before
it is orthonormal. The paper ``pose/`` splits are already normalized and store
translation in units of 100 m; pass ``rotation_fix=none`` and
``translation_scale=100`` for those.

Depth is a float32 EXR, Z-depth in centimeters (use the R channel). The
official ``load_data.py`` divides by 10,000 to land in units of 100 m. We
divide by 100 so the stack stays in meters. The float16 EXR clips at 655.04 m
and is not the file to train on.

The aerial capture is four yaw directions at one location, usually stored as
consecutive frames. ``views_per_sample`` (default 4) chunks them into a virtual
rig. The world frame stays the city frame, which is Z-up, not a georeferenced
ENU origin. Camera axes are converted to OpenCV.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aerosplat.data.errors import DataError
from aerosplat.data.specs import SpecDataset, ViewGroup, ViewSpec, chunk_views
from aerosplat.geometry.coordinates import intrinsics_from_hfov, nerf_c2w_to_opencv_w2c


def load_matrixcity(root: Path, cfg) -> SpecDataset:
    root = Path(root)
    files = discover_transforms(root)
    if not files:
        raise DataError(f"{root} has no MatrixCity transforms.json")
    views_per_sample = int(getattr(cfg, "views_per_sample", 4))
    rotation_fix = str(getattr(cfg, "rotation_fix", "auto"))
    translation_scale = float(getattr(cfg, "translation_scale", 1.0))
    depth_scale = float(getattr(cfg, "depth_scale", 0.01))
    groups: list[ViewGroup] = []
    for path in files:
        views = _read_transforms(
            path,
            rotation_fix=rotation_fix,
            translation_scale=translation_scale,
            depth_scale=depth_scale,
        )
        scene_id = path.parent.name
        groups.extend(chunk_views(views, views_per_sample, scene_id))
    return SpecDataset(groups)


def discover_transforms(root: Path) -> list[Path]:
    root = Path(root)
    if root.is_file() and root.name.endswith(".json"):
        return [root]
    if (root / "transforms.json").is_file():
        return [root / "transforms.json"]
    return sorted(root.glob("**/transforms.json"))


def fix_matrixcity_rotation(rotation: np.ndarray, mode: str = "auto") -> np.ndarray:
    """Undo the per-block ``rotation * 100`` gotcha and check the result is SO(3)."""
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if mode == "none":
        scaled = matrix
    elif mode == "times_100":
        scaled = matrix * 100.0
    elif mode == "auto":
        scaled = matrix * 100.0 if abs(float(np.linalg.det(matrix))) < 1e-2 else matrix
    else:
        raise DataError(f"rotation_fix must be auto, none, or times_100, got {mode!r}")
    det = float(np.linalg.det(scaled))
    orthogonal = np.allclose(scaled.T @ scaled, np.eye(3), atol=1e-2)
    if abs(det - 1.0) > 1e-2 or not orthogonal:
        raise DataError(
            "MatrixCity rotation is not a proper rotation after the scale fix "
            f"(det={det:.3e}, mode={mode}). Paper pose/ splits are already "
            "normalized: set rotation_fix=none and translation_scale=100."
        )
    return scaled


def _read_transforms(
    path: Path,
    *,
    rotation_fix: str,
    translation_scale: float,
    depth_scale: float,
) -> list[ViewSpec]:
    payload = json.loads(path.read_text())
    frames = payload.get("frames", payload if isinstance(payload, list) else None)
    if not frames:
        raise DataError(f"{path} has no frames")
    views = []
    for index, frame in enumerate(frames):
        matrix = _frame_matrix(frame)
        matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        rotation = fix_matrixcity_rotation(matrix[:3, :3], rotation_fix)
        center = matrix[:3, 3] * translation_scale
        fixed = np.eye(4, dtype=np.float64)
        fixed[:3, :3] = rotation
        fixed[:3, 3] = center
        R, t = nerf_c2w_to_opencv_w2c(fixed)
        image_path = _image_path(path.parent, frame, index)
        from PIL import Image

        with Image.open(image_path) as img:
            width, height = img.size
        K = _intrinsics(payload, frame, width, height)
        depth_path = _depth_path(path.parent, frame, image_path)
        views.append(
            ViewSpec(
                name=f"view{index}",
                image_path=image_path,
                depth_path=depth_path,
                K=K,
                R=R,
                t=t,
                depth_scale=depth_scale,
                depth_format="exr"
                if depth_path is not None and depth_path.suffix.lower() == ".exr"
                else "auto",
            )
        )
    return views


def _frame_matrix(frame: dict) -> np.ndarray:
    if "transform_matrix" in frame:
        return np.asarray(frame["transform_matrix"], dtype=np.float64)
    if "rot_mat" in frame:
        rot = np.asarray(frame["rot_mat"], dtype=np.float64)
        if rot.shape == (4, 4):
            return rot
        if rot.shape == (3, 3):
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = rot
            position = frame.get("position", frame.get("translation", frame.get("T")))
            if position is None:
                raise DataError("rot_mat is 3x3 but the frame has no position")
            matrix[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
            return matrix
    raise DataError("frame has neither transform_matrix nor rot_mat")


def _intrinsics(payload: dict, frame: dict, width: int, height: int) -> np.ndarray:
    if "fl_x" in frame or "fl_x" in payload:
        source = frame if "fl_x" in frame else payload
        fx = float(source["fl_x"])
        fy = float(source.get("fl_y", fx))
        cx = float(source.get("cx", width / 2.0))
        cy = float(source.get("cy", height / 2.0))
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    angle = frame.get("camera_angle_x", payload.get("camera_angle_x"))
    if angle is None:
        raise DataError("frame has no fl_x and no camera_angle_x")
    # camera_angle_x is the horizontal field of view in radians.
    fov = float(np.rad2deg(float(angle)))
    return intrinsics_from_hfov(width, height, fov)


def _image_path(folder: Path, frame: dict, index: int) -> Path:
    raw = frame.get("file_path", frame.get("image"))
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = folder / candidate
        if candidate.is_file():
            return candidate
        for suffix in (".png", ".jpg", ".jpeg"):
            with_suffix = candidate.with_suffix(suffix)
            if with_suffix.is_file():
                return with_suffix
        raise DataError(f"image {raw} was not found beside the transforms file")
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = folder / f"{index:04d}{suffix}"
        if candidate.is_file():
            return candidate
    raise DataError(f"frame {index} has no file_path and no {index:04d}.png")


def _depth_path(folder: Path, frame: dict, image_path: Path) -> Path | None:
    raw = frame.get("depth_path")
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = folder / candidate
        return candidate if candidate.is_file() else None
    stem = image_path.stem
    candidates = [
        image_path.with_suffix(".exr"),
        image_path.with_name(stem + "_depth.exr"),
        folder / "depth" / f"{stem}.exr",
        image_path.parent.parent / "depth" / f"{stem}.exr",
        image_path.with_suffix(".npy"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
