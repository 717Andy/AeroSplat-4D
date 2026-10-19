"""Recorded episodes in the Phase 1 canonical dataset format.

An episode directory is what ``aerosplat.data.adapters.canonical`` writes:
``dataset.json`` plus per-camera ``frames.jsonl``. Optional workbench files
live beside that recording and are not required to list or fetch a frame:

    clouds/00000.ags
    occupancy/00000.npz
    glare/00000/<camera>.png
    meta.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from aerosplat.data.adapters.canonical import read_canonical
from aerosplat.data.dataset import MultiCameraDataset
from aerosplat.data.records import SyncConfig
from aerosplat.models.occupancy_head import OccupancyVolume
from aerosplat.schema import MultiCameraSample


@dataclass
class Episode:
    id: str
    root: Path
    dataset: MultiCameraDataset
    meta: dict

    @property
    def samples(self) -> int:
        return len(self.dataset)

    def timestamps(self) -> list[float]:
        return [float(stamp.timestamp) for stamp in self.dataset._index]


def discover_episodes(dirs: list[Path]) -> dict[str, Episode]:
    """Map episode id (directory name) to a synchronized dataset.

    A later directory with the same name replaces an earlier one.
    """
    found: dict[str, Episode] = {}
    for directory in dirs:
        if not directory.exists():
            continue
        candidates = (
            [directory] if (directory / "dataset.json").is_file() else sorted(directory.iterdir())
        )
        for root in candidates:
            if not (root / "dataset.json").is_file():
                continue
            episode_id = root.name
            scenes = read_canonical(root)
            sync = _sync_for(root)
            dataset = MultiCameraDataset(scenes, sync, load_depth=True)
            meta_path = root / "meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
            found[episode_id] = Episode(id=episode_id, root=root, dataset=dataset, meta=meta)
    return found


def _sync_for(root: Path) -> SyncConfig:
    manifest = json.loads((root / "dataset.json").read_text())
    sync = manifest.get("sync") or {}
    return SyncConfig(
        policy=str(sync.get("policy", "interpolate_pose")),
        reference_camera=str(sync.get("reference_camera", "front")),
        max_image_dt=float(sync.get("max_image_dt", 0.15)),
        require_all_cameras=bool(sync.get("require_all_cameras", True)),
    )


def frame_payload(episode: Episode, index: int) -> dict:
    sample = episode.dataset[index]
    timestamps = episode.timestamps()
    dt = 0.0
    next_timestamp = None
    if index + 1 < len(timestamps):
        next_timestamp = timestamps[index + 1]
        dt = next_timestamp - timestamps[index]
    cameras = []
    for name, frame in sample.cameras.items():
        image = frame.image.detach().cpu().numpy()
        cameras.append(
            {
                "name": name,
                "timestamp": float(frame.timestamp),
                "sync": frame.sync,
                "frame_index": int(frame.frame_index),
                "width": int(image.shape[-1]),
                "height": int(image.shape[-2]),
                "K": frame.K.detach().cpu().tolist(),
                "R": frame.R.detach().cpu().tolist(),
                "t": frame.t.detach().cpu().tolist(),
                "image_url": f"/api/v1/episodes/{episode.id}/frames/{index}/cameras/{name}/image",
                "glare_url": f"/api/v1/episodes/{episode.id}/frames/{index}/cameras/{name}/glare",
                "has_glare": (episode.root / "glare" / f"{index:05d}" / f"{name}.png").is_file(),
            }
        )
    cloud = episode.root / "clouds" / f"{index:05d}.ags"
    return {
        "episode": episode.id,
        "index": index,
        "scene_id": sample.scene_id,
        "timestamp": float(sample.timestamp),
        "next_timestamp": next_timestamp,
        "dt": float(dt),
        "samples": episode.samples,
        "cameras": cameras,
        "cloud_url": f"/api/v1/episodes/{episode.id}/clouds/{index}" if cloud.is_file() else None,
        "occupancy_url": f"/api/v1/occupancy?episode={episode.id}&index={index}",
        "source_sun": episode.meta.get("source_sun"),
        "corridor": episode.meta.get("corridor"),
        "corridor_radius_m": episode.meta.get("corridor_radius_m"),
    }


def camera_image_png(episode: Episode, index: int, camera: str) -> bytes:
    from .bake import glare_png

    sample = episode.dataset[index]
    if camera not in sample.cameras:
        raise KeyError(camera)
    image = sample.cameras[camera].image.detach().cpu().numpy()
    hwc = np.transpose(image, (1, 2, 0))
    return glare_png(hwc)


def camera_glare_png(episode: Episode, index: int, camera: str) -> bytes | None:
    path = episode.root / "glare" / f"{index:05d}" / f"{camera}.png"
    if not path.is_file():
        return None
    return path.read_bytes()


def cloud_bytes(episode: Episode, index: int) -> bytes | None:
    path = episode.root / "clouds" / f"{index:05d}.ags"
    if not path.is_file():
        return None
    return path.read_bytes()


def load_volume(episode: Episode, index: int) -> OccupancyVolume | None:
    path = episode.root / "occupancy" / f"{index:05d}.npz"
    if not path.is_file():
        return None
    data = np.load(path)
    return OccupancyVolume(
        occupancy_logits=torch.from_numpy(data["occupancy_logits"].astype(np.float32)),
        wire_logits=torch.from_numpy(data["wire_logits"].astype(np.float32)),
        wire_direction=torch.from_numpy(data["wire_direction"].astype(np.float32)),
        wire_offset=torch.from_numpy(data["wire_offset"].astype(np.float32)),
        origin=torch.from_numpy(data["origin"].astype(np.float32)),
        voxel_size=torch.tensor(float(np.asarray(data["voxel_size"]))),
    )


def sample_for_infer(episode: Episode, index: int) -> MultiCameraSample:
    return episode.dataset[index]


def save_volume_npz(path: Path, volume_arrays: dict[str, np.ndarray], voxel_size: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, voxel_size=np.float32(voxel_size), **volume_arrays)
