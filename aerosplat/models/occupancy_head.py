"""Probabilistic occupancy from the Gaussian cloud.

The grid is the local flight-frame volume the Phase 1 cache already uses.
A 3D convolution on that grid exports with a static shape. An implicit query
MLP would answer one point at a time; the collision API wants the whole
corridor, and a voxel lookup is the fast path.

Thin wires do not get their own centimeter volume inside the network. A 2 cm
cell over a 12 m corridor is hundreds of millions of voxels. The head instead
emits four fields on the coarse grid:

* ``occupancy_logits`` — solid surface, the cache's occupied-versus-free label
* ``wire_logits`` — a cable crosses this cell, even when the cell is 0.25 m
* ``wire_direction`` — a unit tangent of that cable
* ``wire_offset`` — where the cable sits inside the cell, in ``[-0.5, 0.5]``
  voxel units along each axis

A 1–3 cm cable is then a line, not a filled cell. The collision query measures
distance to the line. Filling the cell would report a hit for a trajectory
that misses the cable by a hand's width.

Gaussians are splatted into the grid by nearest voxel. ``wire_prob`` is added
onto the wire logit through a learned skip that starts at 1, so a wire-flagged
Gaussian already lights its cell before the convolution trains. The scatter
index is not differentiable; photometric and temporal losses move the means.
The occupancy loss moves opacity, ``wire_prob``, and the convolution.

Lifted features are optional (``extra``). The default inputs are the packed
attributes the export graph already has: position, opacity, scale, the SH DC
color, and ``wire_prob``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from aerosplat.models.gaussians import GaussianCloud


@dataclass
class OccupancyVolume:
    """One local grid. Logits are raw. Probabilities are ``sigmoid`` of them."""

    occupancy_logits: Tensor
    wire_logits: Tensor
    wire_direction: Tensor
    wire_offset: Tensor
    origin: Tensor
    voxel_size: Tensor

    @property
    def occupancy(self) -> Tensor:
        return torch.sigmoid(self.occupancy_logits)

    @property
    def wire(self) -> Tensor:
        return torch.sigmoid(self.wire_logits)


def scatter_gaussians(
    values: Tensor,
    xyz: Tensor,
    origin: Tensor,
    voxel_size: Tensor | float,
    grid_size: tuple[int, int, int],
    weights: Tensor,
) -> Tensor:
    """Nearest-voxel splat. ``values`` ``[B, N, C]`` -> ``[B, C, X, Y, Z]``.

    Points outside the grid are dropped. Their clamped index is multiplied by
    a zero weight so they do not land on the border cell.
    """
    if values.ndim != 3:
        raise ValueError(f"values must be [B, N, C], got {tuple(values.shape)}")
    if xyz.shape[:2] != values.shape[:2] or xyz.shape[-1] != 3:
        raise ValueError(f"xyz {tuple(xyz.shape)} does not match values {tuple(values.shape)}")
    if weights.shape != values.shape[:2]:
        raise ValueError(f"weights {tuple(weights.shape)} != {(values.shape[0], values.shape[1])}")
    x_size, y_size, z_size = (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))
    if min(x_size, y_size, z_size) < 1:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    origin_b = origin.to(dtype=values.dtype, device=values.device)
    if origin_b.ndim == 1:
        origin_b = origin_b.view(1, 1, 3)
    elif origin_b.ndim == 2:
        origin_b = origin_b[:, None, :]
    else:
        raise ValueError(f"origin must be [3] or [B, 3], got {tuple(origin.shape)}")
    size = torch.as_tensor(voxel_size, dtype=values.dtype, device=values.device).clamp_min(1e-8)
    scaled = (xyz - origin_b) / size
    ijk = scaled.floor()
    inside = (
        (ijk[..., 0] >= 0)
        & (ijk[..., 0] < x_size)
        & (ijk[..., 1] >= 0)
        & (ijk[..., 1] < y_size)
        & (ijk[..., 2] >= 0)
        & (ijk[..., 2] < z_size)
    )
    ijk_long = torch.stack(
        (
            ijk[..., 0].clamp(0, x_size - 1),
            ijk[..., 1].clamp(0, y_size - 1),
            ijk[..., 2].clamp(0, z_size - 1),
        ),
        dim=-1,
    ).long()
    flat = ijk_long[..., 0] * (y_size * z_size) + ijk_long[..., 1] * z_size + ijk_long[..., 2]
    flat = flat.clamp(0, x_size * y_size * z_size - 1)
    weight = weights.to(dtype=values.dtype) * inside.to(dtype=values.dtype)
    batch, _, channels = values.shape
    canvas = values.new_zeros(batch, channels, x_size * y_size * z_size)
    index = flat[:, None, :].expand(batch, channels, flat.shape[1])
    canvas.scatter_add_(2, index, (values * weight[..., None]).transpose(1, 2))
    return canvas.view(batch, channels, x_size, y_size, z_size)


class OccupancyHead(nn.Module):
    """Splat the cloud into a grid and run a small 3D convolution."""

    def __init__(self, hidden: int = 16, extra_dim: int = 0) -> None:
        super().__init__()
        if hidden < 1:
            raise ValueError("hidden must be positive")
        if extra_dim < 0:
            raise ValueError("extra_dim must be >= 0")
        self.hidden = int(hidden)
        self.extra_dim = int(extra_dim)
        # opacity, wire_prob, scale (3), SH DC (3), fractional voxel offset (3), optional extra.
        in_dim = 1 + 1 + 3 + 3 + 3 + self.extra_dim
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=False),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=False),
        )
        self.conv = nn.Sequential(
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        # occ, wire, direction xyz, offset xyz.
        self.output = nn.Conv3d(hidden, 8, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.output.bias.data[2] = 1.0
        # A wire-flagged Gaussian lights its cell before the conv learns.
        self.wire_skip = nn.Parameter(torch.tensor(1.0))

    def forward_cloud(
        self,
        cloud: GaussianCloud,
        origin: Tensor,
        voxel_size: Tensor | float,
        grid_size: tuple[int, int, int],
        extra: Tensor | None = None,
    ) -> OccupancyVolume:
        parts = cloud.unpack()
        return self.forward(
            parts["xyz"],
            parts["opacity"].squeeze(-1),
            cloud.wire_prob,
            cloud.valid,
            origin,
            voxel_size,
            grid_size,
            parts["scale"],
            parts["sh"][..., 0, :],
            extra=extra,
        )

    def forward(
        self,
        xyz: Tensor,
        opacity: Tensor,
        wire_prob: Tensor,
        valid: Tensor,
        origin: Tensor,
        voxel_size: Tensor | float,
        grid_size: tuple[int, int, int],
        scale: Tensor,
        sh_dc: Tensor,
        extra: Tensor | None = None,
    ) -> OccupancyVolume:
        if opacity.ndim == 3:
            opacity = opacity.squeeze(-1)
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must be [B, N, 3], got {tuple(xyz.shape)}")
        batch, count, _ = xyz.shape
        for name, tensor in (
            ("opacity", opacity),
            ("wire_prob", wire_prob),
        ):
            if tensor.shape != (batch, count):
                raise ValueError(f"{name} must be {(batch, count)}, got {tuple(tensor.shape)}")
        if valid.shape != (batch, count):
            raise ValueError(f"valid must be {(batch, count)}, got {tuple(valid.shape)}")
        if scale.shape != (batch, count, 3) or sh_dc.shape != (batch, count, 3):
            raise ValueError("scale and sh_dc must be [B, N, 3]")
        origin_b = origin.to(dtype=xyz.dtype, device=xyz.device)
        if origin_b.ndim == 1:
            origin_b = origin_b.view(1, 3).expand(batch, 3)
        size = torch.as_tensor(voxel_size, dtype=xyz.dtype, device=xyz.device).clamp_min(1e-8)
        scaled = (xyz - origin_b[:, None, :]) / size
        fractional = scaled - scaled.floor()
        pieces = [opacity.unsqueeze(-1), wire_prob.unsqueeze(-1), scale, sh_dc, fractional]
        if self.extra_dim:
            if extra is None or extra.shape != (batch, count, self.extra_dim):
                raise ValueError(
                    f"extra must be {(batch, count, self.extra_dim)}, "
                    f"got {None if extra is None else tuple(extra.shape)}"
                )
            pieces.append(extra)
        embedded = self.embed(torch.cat(pieces, dim=-1))
        weights = valid.to(dtype=xyz.dtype) * opacity.clamp(0, 1)
        volume = scatter_gaussians(embedded, xyz, origin_b, size, grid_size, weights)
        wire_mass = scatter_gaussians(
            wire_prob.unsqueeze(-1),
            xyz,
            origin_b,
            size,
            grid_size,
            weights,
        )
        hidden = self.conv(volume)
        raw = self.output(hidden)
        occupancy_logits = raw[:, 0]
        wire_logits = raw[:, 1] + self.wire_skip * wire_mass[:, 0]
        direction = F.normalize(raw[:, 2:5], dim=1, eps=1e-6)
        offset = torch.tanh(raw[:, 5:8]) * 0.5
        return OccupancyVolume(
            occupancy_logits=occupancy_logits,
            wire_logits=wire_logits,
            wire_direction=direction,
            wire_offset=offset,
            origin=origin_b,
            voxel_size=size,
        )
