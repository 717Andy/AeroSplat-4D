"""Plane-sweep lift from multi-camera tokens to flight-frame anchors.

Why a plane sweep, not a learned frustum by itself
---------------------------------------------------
The spec allows a cost volume or a frustum projection. The rig's value is that
neighboring cameras overlap, and a cost volume is the module that uses that
overlap: for each reference pixel it warps the other cameras onto hypothesized
depth planes with the known ``K`` and ``R|t`` and scores the match. A frustum
head that only shoots a ray and regresses a depth never compares the other
view, so it cannot be checked against the fixture without training, and it
throws away the calibration Phase 1 exists to provide.

The sweep assumes fronto-parallel planes in the reference camera (the MVSNet
hypothesis). That is approximate on slanted facades and on the UAM rig, whose
side cameras sit about 90° off the nose. The photometric cost is still a
usable initialization, and a 1x1 residual on the ViT features is added on top
so depth pretraining can move the distribution. The frustum step is the
unprojection that turns the resulting planar depth into 3D points in the
flight frame. Phase 3 consumes those points as Gaussian anchors.

Textureless surfaces make the cost flat along a ray. The mode of the
distribution is the geometric readout. The expected depth is the mean of that
distribution and is only sharp when the match is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from aerosplat.models.backbone import BackboneTokens
from aerosplat.models.gaussians import GAUSSIAN_DIM


@dataclass
class CameraLift:
    """One reference camera after the sweep, at the lift grid."""

    name: str
    features: Tensor
    depth_logits: Tensor
    depth_probs: Tensor
    depth: Tensor
    depth_mode: Tensor
    points: Tensor
    confidence: Tensor
    valid: Tensor
    uv: Tensor
    multiscale: dict[int, Tensor] = field(default_factory=dict)


@dataclass
class LiftedGeometry:
    """What Phase 3 heads consume.

    ``anchors`` is ``[B, N, 3]`` flight-frame points, one per lift-grid cell
    per camera, concatenated in camera order. ``N`` is the Gaussian count if
    the geometry head emits one splat per anchor. The other 59 channels of
    ``[B, N, 62]`` are not predicted here; :meth:`gaussian_xyz` is the slice
    that packs into ``GaussianLayout.XYZ``. ``anchor_valid`` marks cells whose
    winning depth plane was seen by at least one other camera.
    """

    cameras: dict[str, CameraLift]
    camera_names: list[str]
    anchors: Tensor
    anchor_features: Tensor
    anchor_valid: Tensor
    camera_index: Tensor
    planes: Tensor
    gaussian_dim: int = GAUSSIAN_DIM

    @property
    def gaussian_slots(self) -> int:
        return int(self.anchors.shape[1])

    def gaussian_xyz(self) -> Tensor:
        """``[B, N, 3]`` world positions for the packed Gaussian tensor."""
        return self.anchors


def make_depth_planes(
    near: float,
    far: float,
    count: int,
    spacing: str,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if count < 2:
        raise ValueError("need at least two depth planes")
    if not 0 < near < far:
        raise ValueError(f"depth range must satisfy 0 < near < far, got {near}, {far}")
    if spacing == "linear":
        return torch.linspace(near, far, count, device=device, dtype=dtype)
    if spacing == "inverse":
        inverse = torch.linspace(1.0 / far, 1.0 / near, count, device=device, dtype=dtype)
        return 1.0 / inverse
    raise ValueError(f"depth spacing must be linear or inverse, got {spacing!r}")


def lift_pixel_grid(height: int, width: int, stride: int, device, dtype) -> tuple[Tensor, int, int]:
    """Integer-or-center pixel coordinates of the lift grid. ``uv`` is ``[h, w, 2]``."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    grid_h = height // stride
    grid_w = width // stride
    if grid_h < 1 or grid_w < 1:
        raise ValueError(f"stride {stride} is larger than the image {height}x{width}")
    # Stride 1 uses pixel indices 0..W-1, matching the Phase 1 ray convention.
    # Coarser strides use the center index of each bin.
    offset = (stride - 1) / 2.0
    v = torch.arange(grid_h, device=device, dtype=dtype) * stride + offset
    u = torch.arange(grid_w, device=device, dtype=dtype) * stride + offset
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return torch.stack((uu, vv), dim=-1), grid_h, grid_w


def sample_at(image: Tensor, uv: Tensor) -> Tensor:
    """Bilinear sample. ``image`` is ``[B, C, H, W]``, ``uv`` is ``[h, w, 2]`` pixels."""
    _, _, height, width = image.shape
    if width == 1 and height == 1:
        return image.expand(-1, -1, uv.shape[0], uv.shape[1])
    grid_x = _to_grid(uv[..., 0], width)
    grid_y = _to_grid(uv[..., 1], height)
    grid = torch.stack((grid_x, grid_y), dim=-1)
    grid = grid.unsqueeze(0).expand(image.shape[0], -1, -1, -1)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def warp_to_reference(
    source: Tensor,
    K_ref: Tensor,
    R_ref: Tensor,
    t_ref: Tensor,
    K_src: Tensor,
    R_src: Tensor,
    t_src: Tensor,
    depth: Tensor,
    uv: Tensor,
) -> tuple[Tensor, Tensor]:
    """Warp ``source`` onto the reference lift grid at one planar depth.

    ``depth`` is ``[B, h, w]`` meters. Returns the warped image ``[B, C, h, w]``
    and a mask of samples that land in front of the source camera and inside
    its image.
    """
    points_cam = _pixels_to_camera(uv, depth, K_ref)
    world = torch.einsum("bhwi,bij->bhwj", points_cam - t_ref[:, None, None, :], R_ref)
    src_cam = torch.einsum("bhwj,bij->bhwi", world, R_src) + t_src[:, None, None, :]
    z_src = src_cam[..., 2]
    safe_z = torch.where(z_src.abs() < 1e-6, torch.ones_like(z_src), z_src)
    u = K_src[:, 0, 0][:, None, None] * src_cam[..., 0] / safe_z + K_src[:, 0, 2][:, None, None]
    v = K_src[:, 1, 1][:, None, None] * src_cam[..., 1] / safe_z + K_src[:, 1, 2][:, None, None]
    _, _, src_h, src_w = source.shape
    grid = torch.stack((_to_grid(u, src_w), _to_grid(v, src_h)), dim=-1)
    warped = F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    valid = (z_src > 0.05) & (u >= 0) & (v >= 0) & (u <= src_w - 1) & (v <= src_h - 1)
    return warped, valid


def plane_sweep_cost(
    images: list[Tensor],
    K: list[Tensor],
    R: list[Tensor],
    t: list[Tensor],
    planes: Tensor,
    uv: Tensor,
    ref_index: int = 0,
) -> tuple[Tensor, Tensor]:
    """Photometric L1 cost of a plane sweep.

    ``images[i]`` is ``[B, 3, H_i, W_i]`` in a range where 1 is a full mismatch
    (typically ``[0, 1]``). Returns ``cost`` ``[B, D, h, w]`` and ``hit``
    ``[B, D, h, w]`` (True where at least one other camera saw that plane).
    Pixels with no overlap get cost 1.
    """
    ref = sample_at(images[ref_index], uv)
    batch = ref.shape[0]
    grid_h, grid_w = uv.shape[:2]
    device, dtype = ref.device, ref.dtype
    cost_sum = torch.zeros(batch, planes.shape[0], grid_h, grid_w, device=device, dtype=dtype)
    hit_count = torch.zeros(batch, planes.shape[0], grid_h, grid_w, device=device, dtype=dtype)
    for index, source in enumerate(images):
        if index == ref_index:
            continue
        for plane_index, depth_value in enumerate(planes):
            depth = depth_value.expand(batch, grid_h, grid_w)
            warped, valid = warp_to_reference(
                source,
                K[ref_index],
                R[ref_index],
                t[ref_index],
                K[index],
                R[index],
                t[index],
                depth,
                uv,
            )
            diff = (warped - ref).abs().mean(dim=1)
            valid_f = valid.to(dtype)
            cost_sum[:, plane_index] = cost_sum[:, plane_index] + diff * valid_f
            hit_count[:, plane_index] = hit_count[:, plane_index] + valid_f
    hit = hit_count > 0
    mean = cost_sum / hit_count.clamp_min(1.0)
    cost = torch.where(hit, mean, torch.ones_like(mean))
    return cost, hit


def depth_from_cost(
    cost: Tensor,
    planes: Tensor,
    temperature: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Softmax over planes. Returns probs, expected depth, mode depth, confidence.

    Depth tensors are ``[B, 1, H, W]``. ``cost`` is ``[B, D, H, W]``, lower is better.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = -cost / temperature
    probs = torch.softmax(logits, dim=1)
    depth = (probs * planes.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
    mode_index = cost.argmin(dim=1)
    mode = planes[mode_index].unsqueeze(1)
    confidence = probs.max(dim=1, keepdim=True).values
    return probs, depth, mode, confidence


def unproject_depth_map(depth: Tensor, uv: Tensor, K: Tensor, R: Tensor, t: Tensor) -> Tensor:
    """Planar depth at ``uv`` to flight-frame points.

    ``depth`` is ``[B, 1, H, W]`` or ``[B, H, W]``.

    Returns ``[B, 3, H, W]``.
    """
    if depth.ndim == 4:
        depth = depth.squeeze(1)
    points_cam = _pixels_to_camera(uv, depth, K)
    world = torch.einsum("bhwi,bij->bhwj", points_cam - t[:, None, None, :], R)
    return world.permute(0, 3, 1, 2).contiguous()


def plane_sweep_depth(
    images: list[Tensor],
    K: list[Tensor],
    R: list[Tensor],
    t: list[Tensor],
    planes: Tensor,
    *,
    ref_index: int = 0,
    stride: int = 1,
    temperature: float = 0.02,
) -> dict[str, Tensor]:
    """Sweep ``images[ref_index]`` and unproject the depth into the flight frame.

    This is the geometry-only path (no learned residual). ``depth`` is the
    expected planar Z. ``depth_mode`` is the plane with the lowest cost.
    ``points`` uses the expected depth.
    """
    ref = images[ref_index]
    uv, _, _ = lift_pixel_grid(ref.shape[-2], ref.shape[-1], stride, ref.device, ref.dtype)
    cost, hit = plane_sweep_cost(images, K, R, t, planes, uv, ref_index=ref_index)
    probs, depth, mode, confidence = depth_from_cost(cost, planes, temperature)
    mode_index = cost.argmin(dim=1)
    hit_at_mode = hit.gather(1, mode_index.unsqueeze(1)).squeeze(1)
    points = unproject_depth_map(depth, uv, K[ref_index], R[ref_index], t[ref_index])
    return {
        "cost": cost,
        "probs": probs,
        "depth": depth,
        "depth_mode": mode,
        "points": points,
        "confidence": confidence,
        "valid": hit_at_mode,
        "uv": uv,
    }


class EpipolarLift(nn.Module):
    """Plane sweep plus a learned per-plane residual from the reference features.

    ``match="photometric"`` ignores the residual and is the path the fixture
    test uses. ``match="hybrid"`` adds ``Conv2d(C, D)`` applied to the ViT
    features sampled on the lift grid, which is what depth pretraining updates.
    """

    def __init__(
        self,
        embed_dim: int,
        num_planes: int = 32,
        depth_near: float = 0.5,
        depth_far: float = 80.0,
        depth_spacing: str = "linear",
        stride: int = 1,
        temperature: float = 0.02,
        residual_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_planes = num_planes
        self.depth_near = depth_near
        self.depth_far = depth_far
        self.depth_spacing = depth_spacing
        self.stride = stride
        self.temperature = temperature
        self.residual = nn.Conv2d(embed_dim, num_planes, kernel_size=1)
        nn.init.normal_(self.residual.weight, std=residual_std)
        nn.init.zeros_(self.residual.bias)

    def planes(self, device, dtype) -> Tensor:
        return make_depth_planes(
            self.depth_near,
            self.depth_far,
            self.num_planes,
            self.depth_spacing,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        tokens: BackboneTokens,
        cameras: dict[str, dict[str, Tensor]],
        images: dict[str, Tensor] | None = None,
        match: str = "hybrid",
    ) -> LiftedGeometry:
        if match not in {"photometric", "hybrid"}:
            raise ValueError(f"match must be photometric or hybrid, got {match!r}")
        if images is None:
            raise ValueError("EpipolarLift needs the RGB images to build the photometric cost")
        names = list(tokens.names)
        if list(images) != names:
            # Allow the same cameras in any order, but require the same set.
            if set(images) != set(names):
                raise ValueError("images and backbone tokens name different cameras")
        device = images[names[0]].device
        dtype = images[names[0]].dtype
        planes = self.planes(device, dtype)
        image_list = [images[name] for name in names]
        k_list = [cameras[name]["K"] for name in names]
        r_list = [cameras[name]["R"] for name in names]
        t_list = [cameras[name]["t"] for name in names]
        lifted: dict[str, CameraLift] = {}
        anchor_xyz: list[Tensor] = []
        anchor_feat: list[Tensor] = []
        anchor_valid: list[Tensor] = []
        anchor_index: list[Tensor] = []
        for ref_index, name in enumerate(names):
            ref_image = image_list[ref_index]
            uv, grid_h, grid_w = lift_pixel_grid(
                ref_image.shape[-2], ref_image.shape[-1], self.stride, device, dtype
            )
            cost, hit = plane_sweep_cost(
                image_list, k_list, r_list, t_list, planes, uv, ref_index=ref_index
            )
            features = _resize_features(tokens.features[name], grid_h, grid_w)
            logits = -cost / self.temperature
            if match == "hybrid":
                logits = logits + self.residual(features)
            probs = torch.softmax(logits, dim=1)
            depth = (probs * planes.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
            mode_index = torch.argmin(cost, dim=1)
            mode = planes[mode_index].unsqueeze(1)
            confidence = probs.max(dim=1, keepdim=True).values
            valid = hit.gather(1, mode_index.unsqueeze(1)).squeeze(1)
            points = unproject_depth_map(
                depth, uv, k_list[ref_index], r_list[ref_index], t_list[ref_index]
            )
            lifted[name] = CameraLift(
                name=name,
                features=features,
                depth_logits=logits,
                depth_probs=probs,
                depth=depth,
                depth_mode=mode,
                points=points,
                confidence=confidence,
                valid=valid,
                uv=uv,
                multiscale=tokens.multiscale.get(name, {}),
            )
            batch = points.shape[0]
            anchor_xyz.append(points.permute(0, 2, 3, 1).reshape(batch, grid_h * grid_w, 3))
            anchor_feat.append(features.permute(0, 2, 3, 1).reshape(batch, grid_h * grid_w, -1))
            anchor_valid.append(valid.reshape(batch, grid_h * grid_w))
            anchor_index.append(
                torch.full((batch, grid_h * grid_w), ref_index, dtype=torch.long, device=device)
            )
        return LiftedGeometry(
            cameras=lifted,
            camera_names=names,
            anchors=torch.cat(anchor_xyz, dim=1),
            anchor_features=torch.cat(anchor_feat, dim=1),
            anchor_valid=torch.cat(anchor_valid, dim=1),
            camera_index=torch.cat(anchor_index, dim=1),
            planes=planes,
        )


def _pixels_to_camera(uv: Tensor, depth: Tensor, K: Tensor) -> Tensor:
    """``uv`` ``[h, w, 2]``, ``depth`` ``[B, h, w]`` -> camera points ``[B, h, w, 3]``."""
    fx = K[:, 0, 0][:, None, None]
    fy = K[:, 1, 1][:, None, None]
    cx = K[:, 0, 2][:, None, None]
    cy = K[:, 1, 2][:, None, None]
    u = uv[..., 0][None]
    v = uv[..., 1][None]
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return torch.stack((x, y, depth), dim=-1)


def _to_grid(pixel: Tensor, size: int) -> Tensor:
    if size <= 1:
        return torch.zeros_like(pixel)
    return 2.0 * pixel / (size - 1) - 1.0


def _resize_features(features: Tensor, height: int, width: int) -> Tensor:
    if features.shape[-2:] == (height, width):
        return features
    return F.interpolate(features, size=(height, width), mode="bilinear", align_corners=True)
