"""Batch samples whose cameras share a name and a resolution.

Images are stacked per camera name, not across the rig. A front camera at
1280x720 and a downward camera at 640x480 stay as two tensors.
"""

from __future__ import annotations

from typing import Any

import torch

from aerosplat.schema import MultiCameraSample


def collate_samples(batch: list[MultiCameraSample]) -> dict[str, Any]:
    if not batch:
        raise ValueError("empty batch")
    names = batch[0].camera_names
    for sample in batch[1:]:
        if sample.camera_names != names:
            raise ValueError(
                "cannot collate samples with different camera sets "
                f"({sample.camera_names} vs {names}). Filter the dataset first."
            )
    cameras: dict[str, dict[str, torch.Tensor]] = {}
    for name in names:
        frames = [sample.cameras[name] for sample in batch]
        image_shape = tuple(frames[0].image.shape)
        for frame in frames[1:]:
            if tuple(frame.image.shape) != image_shape:
                raise ValueError(
                    f"camera {name} changes resolution inside the batch "
                    f"({image_shape} vs {tuple(frame.image.shape)})"
                )
        packed: dict[str, torch.Tensor] = {
            "image": torch.stack([frame.image for frame in frames], dim=0),
            "K": torch.stack([frame.K for frame in frames], dim=0),
            "R": torch.stack([frame.R for frame in frames], dim=0),
            "t": torch.stack([frame.t for frame in frames], dim=0),
            "R_query": torch.stack([frame.R_query for frame in frames], dim=0),
            "t_query": torch.stack([frame.t_query for frame in frames], dim=0),
            "timestamp": torch.tensor([frame.timestamp for frame in frames], dtype=torch.float64),
        }
        depths = [frame.depth for frame in frames]
        if all(depth is None for depth in depths):
            pass
        elif any(depth is None for depth in depths):
            raise ValueError(f"camera {name} has depth for only some samples in the batch")
        else:
            packed["depth"] = torch.stack(depths, dim=0)  # type: ignore[arg-type]
        cameras[name] = packed
    return {
        "scene_id": [sample.scene_id for sample in batch],
        "sample_index": torch.tensor([sample.sample_index for sample in batch], dtype=torch.int64),
        "timestamp": torch.tensor([sample.timestamp for sample in batch], dtype=torch.float64),
        "cameras": cameras,
    }
