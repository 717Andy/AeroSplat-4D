"""Image and depth decoding.

Depth is returned in meters as float32 HxW. Zero and negative values mean
"no return" for every writer in this repo (fixture, AirSim-family capture).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as float32 RGB in [0, 1], shaped (3, H, W)."""
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def save_rgb(path: Path, rgb_hwc: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rgb_hwc)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def load_depth(path: Path, *, png_scale: float = 1000.0) -> np.ndarray:
    """Load a depth map as float32 meters, shape (H, W).

    Supported:
    * ``.npy`` / ``.npz`` (key ``depth``) float arrays, already in meters
    * ``.pfm`` AirSim / Middlebury float depth
    * 16-bit PNG, divided by ``png_scale`` (default: millimeters -> meters)
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as data:
            key = "depth" if "depth" in data else data.files[0]
            depth = data[key]
    elif suffix == ".pfm":
        depth = read_pfm(path)
    elif suffix == ".png":
        depth = _read_png_depth(path, png_scale)
    else:
        raise ValueError(f"unsupported depth file {path}")
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"depth in {path} must be HxW, got {depth.shape}")
    return depth


def save_depth_npy(path: Path, depth: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(depth, dtype=np.float32))


def read_pfm(path: Path) -> np.ndarray:
    """Read a Portable FloatMap. The scale line's sign selects endianness."""
    with path.open("rb") as handle:
        header = handle.readline().decode("ascii").strip()
        if header not in {"PF", "Pf"}:
            raise ValueError(f"{path} is not a PFM file (header {header!r})")
        color = header == "PF"
        dim_line = handle.readline().decode("ascii").strip()
        while dim_line.startswith("#"):
            dim_line = handle.readline().decode("ascii").strip()
        width, height = (int(x) for x in dim_line.split())
        scale = float(handle.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(handle, endian + "f")
    expected = width * height * (3 if color else 1)
    if data.size != expected:
        raise ValueError(f"{path} has {data.size} floats, expected {expected}")
    data = data.reshape((height, width, 3) if color else (height, width))
    data = np.flipud(data)
    if color:
        data = data[..., 0]
    return data.astype(np.float32)


def write_pfm(path: Path, depth: np.ndarray) -> None:
    """Write a little-endian grayscale PFM. Used by adapter tests."""
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("PFM writer expects HxW")
    path.parent.mkdir(parents=True, exist_ok=True)
    flipped = np.flipud(arr)
    with path.open("wb") as handle:
        handle.write(f"Pf\n{arr.shape[1]} {arr.shape[0]}\n-1.0\n".encode("ascii"))
        flipped.tofile(handle)


def _read_png_depth(path: Path, png_scale: float) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32) / float(png_scale)
    return arr.astype(np.float32)
