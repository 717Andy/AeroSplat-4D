"""PyTorch dataset that yields :class:`MultiCameraSample` objects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from aerosplat.data.errors import DataError
from aerosplat.data.io import load_depth, load_rgb, load_segmentation
from aerosplat.data.records import FrameRecord, SyncConfig, sync_config_from_dict
from aerosplat.data.sync import SyncedStamp, synchronize
from aerosplat.schema import CameraFrame, MultiCameraSample

Streams = dict[str, list[FrameRecord]]


class MultiCameraDataset(Dataset):
    """Index is a query time on the reference camera, not a raw frame id.

    ``scenes`` maps scene id -> camera name -> captured frames. Frames inside
    one camera may use a different clock and a different resolution than the
    other cameras. :func:`synchronize` decides which of those frames land in
    each sample.
    """

    def __init__(
        self,
        scenes: dict[str, Streams],
        sync: SyncConfig,
        *,
        load_depth: bool = True,
        png_depth_scale: float = 1000.0,
        load_segmentation: bool = False,
    ) -> None:
        if not scenes:
            raise DataError("dataset has no scenes")
        self.sync = sync
        self.load_depth = load_depth
        self.png_depth_scale = png_depth_scale
        self.load_segmentation = load_segmentation
        self._index: list[SyncedStamp] = []
        for scene_id, streams in scenes.items():
            self._index.extend(synchronize(scene_id, streams, sync))
        if not self._index:
            raise DataError(
                "synchronization produced no samples. Check reference_camera, "
                "max_image_dt, and that each camera stream is non-empty."
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> MultiCameraSample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        stamp = self._index[index]
        cameras: dict[str, CameraFrame] = {}
        for name, match in stamp.cameras.items():
            frame = match.frame
            image = torch.from_numpy(load_rgb(frame.image_path))
            depth = None
            if self.load_depth and frame.depth_path is not None:
                depth_np = load_depth(frame.depth_path, png_scale=self.png_depth_scale)
                if frame.depth_kind == "euclidean":
                    depth_np = _euclidean_to_planar(depth_np, frame.K)
                depth = torch.from_numpy(np.ascontiguousarray(depth_np))
            segmentation = None
            if self.load_segmentation and frame.segmentation_path is not None:
                segmentation = torch.from_numpy(load_segmentation(frame.segmentation_path))
            clean_image = None
            if frame.clean_image_path is not None and frame.clean_image_path.is_file():
                clean_image = torch.from_numpy(load_rgb(frame.clean_image_path))
            cameras[name] = CameraFrame(
                name=name,
                image=image,
                K=frame.K,
                R=frame.R,
                t=frame.t,
                timestamp=frame.timestamp,
                depth=depth,
                R_query=match.R_query,
                t_query=match.t_query,
                query_timestamp=stamp.timestamp,
                sync=match.sync,
                frame_index=frame.frame_index,
                segmentation=segmentation,
                clean_image=clean_image,
            )
        return MultiCameraSample(
            scene_id=stamp.scene_id,
            sample_index=index,
            timestamp=stamp.timestamp,
            cameras=cameras,
        )


def _euclidean_to_planar(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Convert range images to planar Z so the rest of the stack shares one kind."""
    height, width = depth.shape
    us = np.arange(width, dtype=np.float64)
    vs = np.arange(height, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    scale = np.sqrt(((uu - cx) / fx) ** 2 + ((vv - cy) / fy) ** 2 + 1.0)
    planar = np.asarray(depth, dtype=np.float64) / scale
    invalid = ~np.isfinite(depth) | (np.asarray(depth) <= 0)
    planar = planar.astype(np.float32)
    planar[invalid] = 0.0
    return planar


def build_dataset(cfg):
    """Construct a dataset from a composed config (see ``configs/``).

    Rig recordings (canonical, AirSim, TartanAir V1) return a
    :class:`MultiCameraDataset`. Geometry-pretraining sources
    (AerialMegaDepth, TartanAir V2, MatrixCity) return a
    :class:`~aerosplat.data.specs.SpecDataset`. Both yield
    :class:`~aerosplat.schema.MultiCameraSample`.
    """
    source = str(cfg.data.source)
    root = Path(str(cfg.data.root))
    if source == "aerial_megadepth":
        from aerosplat.data.adapters.aerial_megadepth import load_aerial_megadepth

        return load_aerial_megadepth(root, cfg.data)
    if source == "tartanair_v2":
        from aerosplat.data.adapters.tartanair_v2 import load_tartanair_v2

        return load_tartanair_v2(root, cfg.data)
    if source == "matrixcity":
        from aerosplat.data.adapters.matrixcity import load_matrixcity

        return load_matrixcity(root, cfg.data)
    sync = sync_config_from_dict(dict(cfg.data.sync))
    load_depth_flag = bool(cfg.data.load_depth)
    load_segmentation = bool(_cfg_get(cfg.data, "load_segmentation", False))
    cache = _cfg_get(cfg, "cache", None)
    if cache is not None:
        wire_ids = list(_cfg_get(cache, "wire_ids", []) or [])
        wire_colors = list(_cfg_get(cache, "wire_colors", []) or [])
        if wire_ids or wire_colors:
            load_segmentation = True
    if source == "canonical":
        from aerosplat.data.adapters.canonical import read_canonical

        scenes = read_canonical(root)
    elif source == "airsim_raw":
        from aerosplat.airsim_family import depth_kind_from_image_type
        from aerosplat.data.adapters.airsim import read_airsim_raw

        raw = cfg.data.airsim_raw
        simulator = str(raw.simulator) if "simulator" in raw and raw.simulator else None
        depth_kind = None
        if "image_types" in raw and raw.image_types and "depth" in raw.image_types:
            depth_kind = depth_kind_from_image_type(str(raw.image_types.depth))
        scenes = {
            "flight": read_airsim_raw(
                Path(str(raw.recording)),
                Path(str(raw.settings)),
                timestamp_unit=str(raw.timestamp_unit),
                depth_kind=depth_kind,
                simulator=simulator,
            )
        }
    elif source == "tartanair":
        from aerosplat.data.adapters.tartanair import read_tartanair

        scenes = read_tartanair(root, cfg.data)
    else:
        raise DataError(
            f"unknown data.source {source!r}. Expected canonical, airsim_raw, tartanair, "
            "aerial_megadepth, tartanair_v2, or matrixcity."
        )
    return MultiCameraDataset(
        scenes,
        sync,
        load_depth=load_depth_flag,
        load_segmentation=load_segmentation,
    )


def _cfg_get(node, key: str, default):
    if node is None:
        return default
    try:
        if key not in node:
            return default
        value = node[key]
    except TypeError:
        value = getattr(node, key, default)
    return default if value is None else value
