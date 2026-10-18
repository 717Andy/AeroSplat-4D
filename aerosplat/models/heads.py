"""Regression heads for the packed Gaussian tensor.

Geometry, appearance, and dynamics are the Phase 3 heads. Their raw outputs
are activated in :class:`aerosplat.models.splats.SplatPredictor` (bounded xyz
offset, anisotropic scale cap, unit quaternion, sigmoid opacity, SH DC from
the image, velocity in m/s) and then packed into ``[B, N, 62]``.

Relighting and occupancy stay side outputs. They are not extra channels in
the 62. Relight consumes image features, the sun direction, and the degree-3
SH field, and it emits an additive glare layer. Occupancy consumes the
Gaussian cloud and is supervised by the cached voxel grid, with a separate
wire channel for cables that are thinner than a voxel.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from aerosplat.models.occupancy_head import OccupancyHead
from aerosplat.models.relight import RelightHead

__all__ = [
    "HEAD_OUTPUTS",
    "AppearanceHead",
    "DynamicsHead",
    "GeometryHead",
    "OccupancyHead",
    "PlaceholderOccupancyHead",
    "RelightHead",
]

# Name -> output fields. Geometry/appearance/dynamics pack into GaussianLayout.
# Relight and occupancy stay beside the 62.
HEAD_OUTPUTS: dict[str, tuple[str, ...]] = {
    "geometry": ("xyz", "scale", "quat"),
    "appearance": ("opacity", "sh"),
    "dynamics": ("velocity",),
    "relight": ("glare_rgb",),
    "occupancy": ("voxel_logits", "wire_logits", "wire_direction", "wire_offset"),
}


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


class GeometryHead(nn.Module):
    """Raw xyz offset, log-scale, and a wxyz quaternion per Gaussian.

    The last layer is zero-initialized except for the quaternion's real part,
    which starts at 1 so the normalized rotation is the identity before any
    wire-tangent alignment. The predictor adds the anchor, the per-slot scale
    bias, and the activations. This module does not see image resolution.
    """

    def __init__(self, embed_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = embed_dim if hidden_dim is None else hidden_dim
        self.proj = _mlp(embed_dim, hidden, 3 + 3 + 4)
        last = self.proj[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        last.bias.data[6] = 1.0

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        raw = self.proj(features)
        return {
            "offset_raw": raw[..., 0:3],
            "log_scale": raw[..., 3:6],
            "quat_raw": raw[..., 6:10],
        }


class AppearanceHead(nn.Module):
    """Opacity logit and a residual on degree-3 RGB spherical harmonics.

    The SH residual is zero at initialization. The predictor adds the DC
    coefficient of the sampled pixel color, so a fresh model splats the image
    color rather than gray. Higher bands stay zero until they train. Opacity
    starts at ``opacity_init`` (default 0.1) instead of a fully opaque cloud.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        opacity_init: float = 0.1,
    ) -> None:
        super().__init__()
        hidden = embed_dim if hidden_dim is None else hidden_dim
        self.proj = _mlp(embed_dim, hidden, 1 + 48)
        last = self.proj[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        probability = min(max(float(opacity_init), 1e-4), 1.0 - 1e-4)
        last.bias.data[0] = math.log(probability / (1.0 - probability))

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        raw = self.proj(features)
        batch, count, _ = raw.shape
        return {
            "opacity_logit": raw[..., 0:1],
            "sh_residual": raw[..., 1:].reshape(batch, count, 16, 3),
        }


class DynamicsHead(nn.Module):
    """World-frame velocity in meters per second.

    Zero-initialized. A static scene is the right starting point: ``L_temp``
    then asks each slot to stay where the next frame puts that same slot,
    and any motion has to be learned.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, 3)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        return {"velocity": self.proj(features)}


class PlaceholderOccupancyHead(nn.Module):
    """One shared logit, broadcast over the Phase 1 cache grid.

    Kept so the Phase 4 loss test still has a tensor that does not read the
    cloud. Training uses :class:`OccupancyHead`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.zeros(()))

    def forward(self, batch: int, dims: tuple[int, int, int]) -> Tensor:
        if batch < 1:
            raise ValueError("batch must be >= 1")
        if len(dims) != 3 or any(int(size) < 1 for size in dims):
            raise ValueError(f"voxel dims must be three positive ints, got {dims}")
        return self.logit.expand(int(batch), int(dims[0]), int(dims[1]), int(dims[2]))
