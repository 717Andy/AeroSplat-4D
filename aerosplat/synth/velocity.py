"""Per-pixel velocity maps sampled onto Gaussian slots.

``L_velocity`` accepts ``[B, N, 3]``. A synthetic recording stores a
flight-frame velocity image per camera. Slots read that image at their
lift-grid pixel. Background pixels are 0, so a static facade is supervised
as static and a moving actor is supervised as that actor. One broadcast
vector cannot do both.
"""

from __future__ import annotations

import torch
from torch import Tensor

from aerosplat.models.gaussians import GaussianCloud


def slot_velocity_from_maps(
    cloud: GaussianCloud,
    velocity_maps: dict[str, Tensor],
) -> Tensor | None:
    """``[B, N, 3]`` nearest-sampled from per-camera ``[B, 3, H, W]`` maps.

    Returns None when a camera is missing a map or the map size disagrees
    with the image the lifter saw. Callers then keep the sample-level vector.
    """
    if not velocity_maps:
        return None
    if any(name not in velocity_maps for name in cloud.camera_order):
        return None
    batch, count, _ = cloud.packed.shape
    device = cloud.packed.device
    dtype = cloud.packed.dtype
    output = torch.zeros(batch, count, 3, device=device, dtype=dtype)
    pixels = cloud.uv.to(device=device)
    for index, name in enumerate(cloud.camera_order):
        select = cloud.camera_index.to(device) == index
        if not bool(select.any()):
            continue
        image = velocity_maps[name].to(device=device, dtype=dtype)
        if image.ndim != 4 or image.shape[0] != batch or image.shape[1] != 3:
            return None
        height, width = int(image.shape[-2]), int(image.shape[-1])
        chosen = pixels[select]
        columns = chosen[:, 0].round().long().clamp(0, width - 1)
        rows = chosen[:, 1].round().long().clamp(0, height - 1)
        output[:, select] = image[:, :, rows, columns].permute(0, 2, 1).contiguous()
    return output
