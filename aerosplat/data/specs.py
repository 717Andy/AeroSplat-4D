"""Lazy multi-view samples for geometry pretraining.

AerialMegaDepth, TartanAir V2, and MatrixCity are not synchronized rigs in the
Phase 1 sense. Each loader groups a handful of views, converts them to OpenCV
world-to-camera in meters, and yields a :class:`MultiCameraSample`. The 4x4
OpenCV cam2world matrix is the inverse of that pose; helpers in
``aerosplat.geometry.coordinates`` build it. Images stay on disk until
``__getitem__``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from aerosplat.data.errors import DataError
from aerosplat.data.exr import read_exr_depth
from aerosplat.data.io import load_depth, load_rgb
from aerosplat.schema import CameraFrame, MultiCameraSample


@dataclass
class ViewSpec:
    """One calibrated view. ``R``, ``t`` are OpenCV world-to-camera, meters."""

    name: str
    image_path: Path
    depth_path: Path | None
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    depth_scale: float = 1.0
    depth_format: str = "auto"


@dataclass
class ViewGroup:
    scene_id: str
    views: list[ViewSpec]
    timestamp: float = 0.0


class SpecDataset(Dataset):
    """Dataset of static multi-view groups. Index is a group, not a clock tick."""

    def __init__(self, groups: list[ViewGroup]) -> None:
        if not groups:
            raise DataError("pretraining dataset has no view groups")
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> MultiCameraSample:
        if index < 0:
            index += len(self)
        group = self.groups[index]
        cameras: dict[str, CameraFrame] = {}
        for spec in group.views:
            image = torch.from_numpy(load_rgb(spec.image_path))
            depth = None
            if spec.depth_path is not None:
                raw = read_depth_array(spec.depth_path, spec.depth_format)
                meters = np.asarray(raw, dtype=np.float32) * np.float32(spec.depth_scale)
                meters = np.where(np.isfinite(meters) & (meters > 0), meters, 0).astype(np.float32)
                if meters.shape != image.shape[-2:]:
                    raise DataError(
                        f"{spec.depth_path} is {meters.shape}, image is {tuple(image.shape[-2:])}"
                    )
                depth = torch.from_numpy(np.ascontiguousarray(meters))
            cameras[spec.name] = CameraFrame(
                name=spec.name,
                image=image,
                K=spec.K,
                R=spec.R,
                t=spec.t,
                timestamp=group.timestamp,
                depth=depth,
                query_timestamp=group.timestamp,
                sync="exact",
            )
        return MultiCameraSample(
            scene_id=group.scene_id,
            sample_index=index,
            timestamp=group.timestamp,
            cameras=cameras,
        )


def chunk_views(views: list[ViewSpec], views_per_sample: int, scene_id: str) -> list[ViewGroup]:
    """Pack consecutive views into virtual rigs named ``view0`` .. ``view{n-1}``.

    AerialMegaDepth and MatrixCity are not a synchronized 4-camera mount.
    Consecutive frames are a stand-in so the lifter sees N OpenCV cameras.
    The names repeat across groups so a batch can collate after a resize.
    """
    if views_per_sample < 1:
        raise DataError("views_per_sample must be >= 1")
    if len(views) < views_per_sample:
        raise DataError(
            f"{scene_id} has {len(views)} views, which is fewer than views_per_sample "
            f"{views_per_sample}"
        )
    groups: list[ViewGroup] = []
    for start in range(0, len(views) - views_per_sample + 1, views_per_sample):
        chunk = views[start : start + views_per_sample]
        renamed = [replace(view, name=f"view{index}") for index, view in enumerate(chunk)]
        groups.append(ViewGroup(scene_id=scene_id, views=renamed, timestamp=float(len(groups))))
    return groups


def read_depth_array(path: Path, depth_format: str) -> np.ndarray:
    """Decode a depth file to a float32 array in the file's native unit."""
    path = Path(path)
    fmt = depth_format
    if fmt == "auto":
        suffix = path.suffix.lower()
        fmt = {".exr": "exr", ".h5": "h5", ".hdf5": "h5"}.get(suffix, "image")
    if fmt == "exr":
        return read_exr_depth(path)
    if fmt == "h5":
        return _read_h5_depth(path)
    if fmt == "tartan_v2":
        return read_tartanair_v2_depth(path)
    if fmt == "image":
        return load_depth(path)
    raise DataError(f"unknown depth format {depth_format!r} for {path}")


def read_tartanair_v2_depth(path: Path) -> np.ndarray:
    """Decode TartanAir V2 depth.

    V2 packs float32 meters into an 8-bit RGBA PNG (``view('<f4')``). ``.npy``
    is accepted for fixtures. The values are already meters.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        depth = np.load(path)
        return np.asarray(depth, dtype=np.float32)
    from PIL import Image

    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[-1] == 4 and arr.dtype == np.uint8:
        packed = np.ascontiguousarray(arr)
        return np.asarray(packed.view("<f4").reshape(arr.shape[0], arr.shape[1]))
    raise DataError(
        f"{path} is not a TartanAir V2 float32-in-RGBA depth PNG (got shape {arr.shape}, "
        f"dtype {arr.dtype})"
    )


def write_tartanair_v2_depth(path: Path, depth: np.ndarray) -> None:
    """Pack a float32 depth map the way TartanAir V2 does. Used by tests."""
    from PIL import Image

    array = np.ascontiguousarray(np.asarray(depth, dtype=np.float32))
    rgba = array.view(np.uint8).reshape(array.shape[0], array.shape[1], 4)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path)


def _read_h5_depth(path: Path) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:
        raise DataError(
            f"{path} is HDF5. Install h5py to read AerialMegaDepth depth "
            "(the sample key is 'depth')."
        ) from exc
    with h5py.File(path, "r") as handle:
        key = "depth" if "depth" in handle else next(iter(handle.keys()))
        return np.asarray(handle[key], dtype=np.float32)
