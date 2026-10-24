"""Write ``episodes/demo`` from the tiny fixture.

The recording itself is the Phase 1 canonical format (``generate_fixture``).
Clouds, occupancy, and glare are baked from that depth so the UI can open
without a GPU or a simulator. They are not a network prediction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aerosplat.data.adapters.canonical import read_canonical
from aerosplat.data.dataset import MultiCameraDataset
from aerosplat.data.fixture import generate_fixture
from aerosplat.data.records import SyncConfig

from .bake import DEFAULT_SUN, bake_sample, glare_png
from .codec import pack_gaussians
from .config import repo_root


def build_demo(root: Path | None = None) -> Path:
    destination = Path(root) if root is not None else repo_root() / "episodes" / "demo"
    if destination.exists():
        # Regenerating into a dirty tree is fine; the fixture writer overwrites frames.
        pass
    generate_fixture(destination)
    scenes = read_canonical(destination)
    dataset = MultiCameraDataset(
        scenes,
        SyncConfig(
            policy="interpolate_pose",
            reference_camera="front",
            max_image_dt=0.15,
            require_all_cameras=True,
        ),
        load_depth=True,
    )
    sun = DEFAULT_SUN / np.linalg.norm(DEFAULT_SUN)
    timestamps = [float(stamp.timestamp) for stamp in dataset._index]
    for index in range(len(dataset)):
        sample = dataset[index]
        dt = 0.0 if index + 1 >= len(timestamps) else timestamps[index + 1] - timestamps[index]
        baked = bake_sample(sample, source_sun=sun)
        cloud = pack_gaussians(
            baked.packed,
            baked.valid,
            baked.wire_prob,
            timestamp=float(sample.timestamp),
            dt=float(dt if dt > 0 else 0.1),
            source_sun=sun,
        )
        cloud_path = destination / "clouds" / f"{index:05d}.ags"
        cloud_path.parent.mkdir(parents=True, exist_ok=True)
        cloud_path.write_bytes(cloud)
        _write_volume(destination / "occupancy" / f"{index:05d}.npz", baked)
        for name, image in baked.glare.items():
            glare_path = destination / "glare" / f"{index:05d}" / f"{name}.png"
            glare_path.parent.mkdir(parents=True, exist_ok=True)
            glare_path.write_bytes(glare_png(image))
    meta = {
        "title": "Tiny fixture flight",
        "description": (
            "Four-camera fixture flying east at 4 m/s. Splats are unprojected "
            "depth, plus a synthetic cable across the corridor and an orange "
            "actor moving east at 1.5 m/s. POST /api/v1/infer runs the network "
            "instead of this bake."
        ),
        "source_sun": sun.astype(float).tolist(),
        "corridor": [[0.0, 0.0, 3.2], [1.2, 0.0, 3.2], [2.6, 0.0, 3.2]],
        "corridor_radius_m": 0.6,
        "world_frame": "enu_z_up",
        "notes": [
            "SH degree 0 matches the surface color. A linear lobe points at source_sun.",
            "The sun slider rotates that lobe. It does not move the baked glare image.",
            "Predict glare calls POST /api/v1/infer with the new sun.",
            "The cable is absent from the RGB frames. "
            "It exists for the wire overlay and the collision query.",
        ],
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    manifest_path = destination / "dataset.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sync"] = {
        "policy": "interpolate_pose",
        "reference_camera": "front",
        "max_image_dt": 0.15,
        "require_all_cameras": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def _write_volume(path: Path, baked) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    direction = np.transpose(baked.wire_direction, (3, 0, 1, 2))
    offset = np.transpose(baked.wire_offset, (3, 0, 1, 2))
    np.savez_compressed(
        path,
        occupancy_logits=baked.occupancy_logits[None].astype(np.float32),
        wire_logits=baked.wire_logits[None].astype(np.float32),
        wire_direction=direction[None].astype(np.float32),
        wire_offset=offset[None].astype(np.float32),
        origin=baked.origin.reshape(1, 3).astype(np.float32),
        voxel_size=np.float32(baked.voxel_size),
    )


def main() -> None:
    destination = build_demo()
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
