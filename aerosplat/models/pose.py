"""Pose-conditioned token injection.

Two encodings are added to every patch token:

* Plücker rays. A pixel (or patch center) is the ray ``(direction, origin ×
  direction)`` in the flight frame. Direction is a unit vector. The moment
  term carries where the camera sits, so the token knows the rig's place on
  the flight path, not only its viewing direction.
* A sinusoidal embedding of ``K`` and ``R|t``, passed through a learned linear
  map and broadcast across the patches. This is the spec's "sinusoidal or
  learned linear" extrinsic injection. It is a per-camera bias; the Plücker
  term is what varies across the image.

``X_cam = R @ X_world + t`` is the flight-frame convention from Phase 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def transform_w2c(
    R: Tensor, t: Tensor, rotation: Tensor, translation: Tensor
) -> tuple[Tensor, Tensor]:
    """Move world-to-camera extrinsics into a transformed flight frame.

    World points change as ``X' = rotation @ X + translation``. Camera-frame
    coordinates of a physical point stay the same, so

        R' = R @ rotation^T
        t' = t - R @ rotation^T @ translation

    ``rotation`` is ``[3, 3]`` and ``translation`` is ``[3]``. ``R`` / ``t`` may
    have a leading batch.
    """
    rotation = rotation.to(dtype=R.dtype, device=R.device).reshape(3, 3)
    translation = translation.to(dtype=t.dtype, device=t.device).reshape(3)
    rotated_shift = rotation.transpose(0, 1) @ translation
    r_new = torch.matmul(R, rotation.transpose(0, 1))
    t_new = t - torch.matmul(R, rotated_shift)
    return r_new, t_new


def plucker_at_pixels(K: Tensor, R: Tensor, t: Tensor, uv: Tensor) -> Tensor:
    """Plücker coordinates of rays through pixel locations.

    ``K`` is ``[B, 3, 3]``, ``R`` is ``[B, 3, 3]``, ``t`` is ``[B, 3]``,
    ``uv`` is ``[B, N, 2]`` or ``[N, 2]`` in pixel index coordinates (the same
    convention as :func:`aerosplat.geometry.camera.unproject_depth`).

    Returns ``[B, N, 6]`` as ``(direction_world, origin_world × direction)``.
    """
    if uv.ndim == 2:
        uv = uv.unsqueeze(0).expand(K.shape[0], -1, -1)
    fx = K[:, 0, 0][:, None]
    fy = K[:, 1, 1][:, None]
    cx = K[:, 0, 2][:, None]
    cy = K[:, 1, 2][:, None]
    ones = torch.ones(uv.shape[0], uv.shape[1], device=uv.device, dtype=uv.dtype)
    dirs_c = torch.stack(
        ((uv[..., 0] - cx) / fx, (uv[..., 1] - cy) / fy, ones),
        dim=-1,
    )
    dirs_c = dirs_c / dirs_c.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    # Row-vector form of dir_world = R.T @ dir_cam, origin = -R.T @ t.
    dirs_w = torch.einsum("bni,bij->bnj", dirs_c, R)
    origin = -torch.einsum("bi,bij->bj", t, R)
    moment = torch.cross(origin[:, None, :], dirs_w, dim=-1)
    return torch.cat((dirs_w, moment), dim=-1)


def patch_pixel_centers(
    height: int, width: int, patch_size: int, device, dtype
) -> tuple[Tensor, int, int]:
    """Pixel-index coordinate of each patch center, including a partial last patch.

    The grid covers the image padded up to a multiple of ``patch_size`` on the
    bottom and right, which is how the ViT conv sees the image. Coordinates
    follow the Phase 1 convention: the ray for an integer pixel index, not a
    half-pixel center of the raster footprint. A patch center is the mean of
    its pixel indices.
    """
    pad_h = (patch_size - height % patch_size) % patch_size
    pad_w = (patch_size - width % patch_size) % patch_size
    grid_h = (height + pad_h) // patch_size
    grid_w = (width + pad_w) // patch_size
    v = (torch.arange(grid_h, device=device, dtype=dtype) + 0.5) * patch_size - 0.5
    u = (torch.arange(grid_w, device=device, dtype=dtype) + 0.5) * patch_size - 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    uv = torch.stack((uu, vv), dim=-1).reshape(-1, 2)
    return uv, grid_h, grid_w


def sinusoidal_encode(values: Tensor, num_frequencies: int) -> Tensor:
    """NeRF-style sin/cos encoding. ``values`` is ``[..., F]``."""
    if num_frequencies < 1:
        raise ValueError("num_frequencies must be >= 1")
    freq = torch.pow(2.0, torch.arange(num_frequencies, device=values.device, dtype=values.dtype))
    freq = freq * torch.pi
    angles = values.unsqueeze(-1) * freq
    encoded = torch.cat((angles.sin(), angles.cos()), dim=-1)
    return encoded.flatten(start_dim=-2)


# 9 rotation entries, 3 translation entries, 4 intrinsics (log fx, log fy, cx, cy).
POSE_FEATURE_DIM = 16


def pose_feature_vector(K: Tensor, R: Tensor, t: Tensor) -> Tensor:
    """Flatten intrinsics and extrinsics into one vector per camera. ``[B, 16]``."""
    intrinsics = torch.stack((K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]), dim=-1)
    # Log focals so a 1080p camera and a 64px fixture share a scale. Principal
    # points stay in pixels; the sinusoidal frequencies cover that range.
    logged = intrinsics.clone()
    logged[:, 0] = torch.log(logged[:, 0].clamp_min(1e-3))
    logged[:, 1] = torch.log(logged[:, 1].clamp_min(1e-3))
    return torch.cat((R.reshape(R.shape[0], 9), t, logged), dim=-1)


class PoseInjection(nn.Module):
    """Add a pose residual to patch tokens. Last linear layers can be zero-init.

    Zero-init keeps a freshly loaded DINOv2 encoder unchanged until training
    moves these layers. Tests that check the injection actually changes tokens
    leave it off.
    """

    def __init__(
        self,
        dim: int,
        mode: str = "plucker",
        num_frequencies: int = 6,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        if mode not in {"plucker", "sinusoidal", "plucker+sinusoidal"}:
            raise ValueError(
                f"pose mode must be plucker, sinusoidal, or plucker+sinusoidal, got {mode!r}"
            )
        self.mode = mode
        self.num_frequencies = num_frequencies
        self.ray = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))
        sin_dim = POSE_FEATURE_DIM * num_frequencies * 2
        self.extrinsic = nn.Sequential(nn.Linear(sin_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        if zero_init:
            nn.init.zeros_(self.ray[-1].weight)
            nn.init.zeros_(self.ray[-1].bias)
            nn.init.zeros_(self.extrinsic[-1].weight)
            nn.init.zeros_(self.extrinsic[-1].bias)

    def forward(
        self, K: Tensor, R: Tensor, t: Tensor, height: int, width: int, patch_size: int
    ) -> Tensor:
        """Return ``[B, grid_h * grid_w, dim]`` aligned with the padded patch grid."""
        uv, _, _ = patch_pixel_centers(height, width, patch_size, K.device, K.dtype)
        out = torch.zeros(
            K.shape[0], uv.shape[0], self.ray[-1].out_features, device=K.device, dtype=K.dtype
        )
        if "plucker" in self.mode:
            rays = plucker_at_pixels(K, R, t, uv.to(dtype=K.dtype))
            out = out + self.ray(rays)
        if "sinusoidal" in self.mode:
            encoded = sinusoidal_encode(pose_feature_vector(K, R, t), self.num_frequencies)
            out = out + self.extrinsic(encoded)[:, None, :]
        return out
