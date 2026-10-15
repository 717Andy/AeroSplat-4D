"""Packed Gaussian tensor consumed by the Phase 4 rasterizer.

Layout of the last dimension, size 62::

    xyz        3   world-frame position, meters, ENU
    scale      3   positive axis scales, meters (already activated)
    quat       4   rotation quaternion, wxyz, unit length
    opacity    1   alpha in [0, 1] (already activated)
    sh         48  degree-3 RGB spherical harmonics, 16 coeffs x 3
    velocity   3   world-frame velocity, meters per second

Spherical harmonics are stored coefficient-major then RGB, so
``packed[..., GaussianLayout.SH].reshape(B, N, 16, 3)`` matches the
``(N, 16, 3)`` tensor diff-gaussian-rasterization expects.

Velocity is packed beside the rasterizer inputs because the dynamics head
predicts it in the same forward pass. The CUDA rasterizer does not read it.
Slice ``packed[..., :GaussianLayout.RASTER_DIM]`` before binding.

``B`` is a batch of timesteps. It is not a flattened video: the temporal loss
compares two packed tensors (t and t+1), it does not index a time axis inside
the 62 channels. ``N`` may differ across clips; a batch pads to one N and a
future ``valid`` mask (not part of the 62) will mark padding. The spec's
phrase "B is the batch size (video frame sequence)" is treated as this.
"""

from __future__ import annotations

import torch
from torch import Tensor

GAUSSIAN_DIM = 62


class GaussianLayout:
    XYZ = slice(0, 3)
    SCALE = slice(3, 6)
    QUAT = slice(6, 10)
    OPACITY = slice(10, 11)
    SH = slice(11, 59)
    VELOCITY = slice(59, 62)
    DIM = 62
    RASTER_DIM = 59
    SH_DEGREE = 3
    SH_COEFFS = (SH_DEGREE + 1) ** 2

    @staticmethod
    def channel_names() -> tuple[str, ...]:
        names = ["x", "y", "z", "sx", "sy", "sz", "qw", "qx", "qy", "qz", "opacity"]
        for coeff in range(GaussianLayout.SH_COEFFS):
            names.extend(f"sh{coeff}_{channel}" for channel in "rgb")
        names.extend(["vx", "vy", "vz"])
        return tuple(names)


def pack_gaussians(
    xyz: Tensor,
    scale: Tensor,
    quat: Tensor,
    opacity: Tensor,
    sh: Tensor,
    velocity: Tensor,
) -> Tensor:
    """Concatenate physical Gaussian parameters into ``[B, N, 62]``.

    Scales must be positive and opacities must lie in ``[0, 1]``. Quaternions
    are normalized here. Inputs are the activated values a rasterizer consumes,
    not network logits. Use :func:`physical_from_raw` to apply exp / sigmoid.
    """
    xyz = _bn(xyz, 3, "xyz")
    scale = _bn(scale, 3, "scale")
    quat = _bn(quat, 4, "quat")
    opacity = _bn(opacity, 1, "opacity")
    sh = _sh(sh, xyz.shape[0], xyz.shape[1])
    velocity = _bn(velocity, 3, "velocity")
    batch, count = xyz.shape[:2]
    for name, tensor in (
        ("scale", scale),
        ("quat", quat),
        ("opacity", opacity),
        ("sh", sh),
        ("velocity", velocity),
    ):
        if tensor.shape[0] != batch or tensor.shape[1] != count:
            raise ValueError(f"{name} batch/count {tensor.shape[:2]} != xyz {(batch, count)}")
    if not (scale > 0).all():
        raise ValueError("scale must be positive; pack activated scales, not log-scales")
    if not ((opacity >= 0) & (opacity <= 1)).all():
        raise ValueError("opacity must lie in [0, 1]")
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.cat((xyz, scale, quat, opacity, sh, velocity), dim=-1)


def unpack_gaussians(packed: Tensor) -> dict[str, Tensor]:
    """Split ``[B, N, 62]`` into physical parameters. ``sh`` is ``[B, N, 16, 3]``."""
    if packed.ndim != 3 or packed.shape[-1] != GaussianLayout.DIM:
        raise ValueError(f"expected [B, N, {GaussianLayout.DIM}], got {tuple(packed.shape)}")
    sh = packed[..., GaussianLayout.SH].reshape(
        packed.shape[0], packed.shape[1], GaussianLayout.SH_COEFFS, 3
    )
    return {
        "xyz": packed[..., GaussianLayout.XYZ],
        "scale": packed[..., GaussianLayout.SCALE],
        "quat": packed[..., GaussianLayout.QUAT],
        "opacity": packed[..., GaussianLayout.OPACITY],
        "sh": sh,
        "velocity": packed[..., GaussianLayout.VELOCITY],
    }


def rasterizer_view(packed: Tensor) -> Tensor:
    """Drop velocity. The CUDA rasterizer does not consume those three channels."""
    if packed.shape[-1] != GaussianLayout.DIM:
        raise ValueError(f"expected last dim {GaussianLayout.DIM}, got {packed.shape[-1]}")
    return packed[..., : GaussianLayout.RASTER_DIM]


def physical_from_raw(
    xyz: Tensor,
    scale_log: Tensor,
    quat_raw: Tensor,
    opacity_logit: Tensor,
    sh: Tensor,
    velocity: Tensor,
) -> Tensor:
    """Map unbounded head outputs to the packed physical tensor.

    Scale uses exp, opacity uses sigmoid, quaternion is normalized. This is
    the activation convention Phase 3 heads should share. It is not something
    the rasterizer applies itself.
    """
    return pack_gaussians(
        xyz,
        torch.exp(scale_log),
        quat_raw,
        torch.sigmoid(opacity_logit),
        sh,
        velocity,
    )


def _bn(tensor: Tensor, channels: int, name: str) -> Tensor:
    if tensor.ndim != 3 or tensor.shape[-1] != channels:
        raise ValueError(f"{name} must have shape [B, N, {channels}], got {tuple(tensor.shape)}")
    return tensor


def _sh(tensor: Tensor, batch: int, count: int) -> Tensor:
    if tensor.ndim == 4:
        if tensor.shape[-2:] != (GaussianLayout.SH_COEFFS, 3):
            raise ValueError(
                f"sh must be [B, N, {GaussianLayout.SH_COEFFS}, 3], got {tuple(tensor.shape)}"
            )
        flat = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
    elif tensor.ndim == 3 and tensor.shape[-1] == GaussianLayout.SH_COEFFS * 3:
        flat = tensor
    else:
        raise ValueError(f"sh must be [B, N, 16, 3] or [B, N, 48], got {tuple(tensor.shape)}")
    if flat.shape[0] != batch or flat.shape[1] != count:
        raise ValueError(f"sh batch/count {flat.shape[:2]} != {(batch, count)}")
    return flat
