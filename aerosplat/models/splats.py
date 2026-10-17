"""Phase 3 splat predictor: heads, wire allocation, slot correspondence.

Correspondence for ``L_temp``
-----------------------------
Andy's loss indexes Gaussian ``i`` at ``t`` and at ``t+1``::

    L_temp = (1 / N) sum_i || (x_i^(t) + v_i^(t) Δt) - x_i^(t+1) ||^2

A feedforward net that re-ranks Gaussians every frame would make ``i`` a
different surface point each time. This module does not do that. ``i`` is a
slot::

    (camera name, lift-cell row, lift-cell col, subslot)

packed in that order. Subslot 0 is the base Gaussian on the lift anchor.
Subslots ``1..K`` are the extra, smaller candidates the edge branch may turn
on. The packed tensor is always ``[B, N, 62]`` with
``N = sum_cameras(grid_h * grid_w * slots_per_cell)``. Thresholding and top-k
write the ``valid`` mask. They do not gather, pad, or permute, so an ONNX
export at a fixed rig resolution has a fixed ``N``, and index ``i`` is the
same pixel slot at the next timestep.

``temporal_residual`` is the quantity inside Andy's sum, plus a weight
``valid_t * valid_{t+1}`` so an extra slot that is off in either frame does
not dominate the mean. Phase 4 should reduce with that weight. The prediction
``x + v Δt`` is the loss target, not a second matching stage.

What this identity is not: a tracked world point. Under ego-motion the same
pixel looks at a different surface point next frame, so a static world still
produces a nonzero residual if velocity stays zero. That is a property of
pixel slots plus Andy's index-aligned formula. A world-point tracker would
break the static shape. See the open questions in ``docs/ARCHITECTURE.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from aerosplat.models.edges import EdgeBranch, allocate_valid
from aerosplat.models.gaussians import (
    GaussianCloud,
    activate_scales,
    pack_gaussians,
    quat_mul,
    quat_normalize,
    quat_slerp,
    quaternion_from_x_axis,
    sh_dc_from_rgb,
)
from aerosplat.models.heads import AppearanceHead, DynamicsHead, GeometryHead
from aerosplat.models.lift import LiftedGeometry, sample_at


@dataclass
class SplatConfig:
    """Knobs for the static slot budget and the wire scale cap.

    Distances are meters. ``slots_per_cell`` includes the base Gaussian, so
    the extra budget per lift cell is ``slots_per_cell - 1``.
    ``max_extra_gaussians`` caps how many of those extras may be ``valid``
    across the whole rig. It does not change ``N``.
    """

    slots_per_cell: int = 4
    max_extra_gaussians: int | None = None
    edge_threshold: float = 0.4
    offset_radius_m: float = 0.5
    along_radius_m: float = 0.4
    min_scale_m: float = 1e-4
    max_scale_m: float = 2.0
    wire_transverse_max_m: float = 0.03
    wire_long_max_m: float = 1.0
    base_scale_m: float = 0.2
    extra_thin_scale_m: float = 0.01
    extra_long_scale_m: float = 0.25
    opacity_init: float = 0.1

    def __post_init__(self) -> None:
        if self.slots_per_cell < 1:
            raise ValueError("slots_per_cell must be >= 1")
        if self.max_extra_gaussians is not None and self.max_extra_gaussians < 0:
            raise ValueError("max_extra_gaussians must be >= 0")
        if not 0.0 < self.edge_threshold < 1.0:
            raise ValueError("edge_threshold must lie in (0, 1)")
        if self.min_scale_m <= 0 or self.max_scale_m < self.min_scale_m:
            raise ValueError("scale floor/ceiling must satisfy 0 < min <= max")
        if not 0 < self.wire_transverse_max_m <= self.wire_long_max_m:
            raise ValueError("wire caps must satisfy 0 < transverse <= long")
        if self.min_scale_m > self.wire_transverse_max_m:
            raise ValueError("min_scale_m must stay below the wire transverse cap")
        if self.extra_thin_scale_m <= 0 or self.extra_long_scale_m < self.extra_thin_scale_m:
            raise ValueError("extra scales must satisfy 0 < thin <= long")
        if self.base_scale_m <= 0:
            raise ValueError("base_scale_m must be positive")
        if self.offset_radius_m < 0 or self.along_radius_m < 0:
            raise ValueError("offset radii must be >= 0")


class SplatPredictor(nn.Module):
    """Turn a :class:`LiftedGeometry` into a :class:`GaussianCloud`."""

    def __init__(self, embed_dim: int, config: SplatConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else SplatConfig()
        slots = self.config.slots_per_cell
        self.geometry = GeometryHead(embed_dim)
        self.appearance = AppearanceHead(embed_dim, opacity_init=self.config.opacity_init)
        self.dynamics = DynamicsHead(embed_dim)
        self.edge = EdgeBranch()
        self.slot_embed = nn.Parameter(torch.zeros(slots, embed_dim))
        nn.init.normal_(self.slot_embed, std=0.02)
        self.register_buffer("log_scale_bias", _scale_bias(self.config), persistent=True)
        self.register_buffer("slot_frac", _slot_frac(slots), persistent=True)

    def forward(
        self,
        lifted: LiftedGeometry,
        images: dict[str, Tensor],
        cameras: dict[str, dict[str, Tensor]],
        highres_images: dict[str, Tensor] | None = None,
    ) -> GaussianCloud:
        cfg = self.config
        slots = cfg.slots_per_cell
        feature_chunks: list[Tensor] = []
        anchor_chunks: list[Tensor] = []
        color_chunks: list[Tensor] = []
        tangent_chunks: list[Tensor] = []
        cell_edge_chunks: list[Tensor] = []
        anchor_valid_chunks: list[Tensor] = []
        camera_index_chunks: list[Tensor] = []
        row_chunks: list[Tensor] = []
        col_chunks: list[Tensor] = []
        subslot_chunks: list[Tensor] = []
        uv_chunks: list[Tensor] = []
        slot_chunks: list[Tensor] = []
        slot_offset = 0
        names = list(lifted.camera_names)
        if not names:
            raise ValueError("LiftedGeometry has no cameras")
        for camera_index, name in enumerate(names):
            if name not in images or name not in cameras:
                raise ValueError(f"camera {name!r} is missing from images or cameras")
            view = lifted.cameras[name]
            image = images[name]
            edge_image = image if highres_images is None else highres_images[name]
            batch = image.shape[0]
            grid_h, grid_w = int(view.points.shape[-2]), int(view.points.shape[-1])
            probability, tangent_xy = self.edge.on_grid(edge_image, grid_h, grid_w)
            cell_edge = probability[:, 0]
            cell_edge_chunks.append(cell_edge.reshape(batch, grid_h * grid_w))
            tangent = _tangent_world(tangent_xy, cameras[name]["K"], cameras[name]["R"])
            tangent_chunks.append(_repeat_slots(tangent, slots))
            color = sample_at(image, view.uv).permute(0, 2, 3, 1).reshape(batch, grid_h * grid_w, 3)
            color_chunks.append(_repeat_slots(color, slots))
            anchors = view.points.permute(0, 2, 3, 1).reshape(batch, grid_h * grid_w, 3)
            anchor_chunks.append(_repeat_slots(anchors, slots))
            features = view.features.permute(0, 2, 3, 1).reshape(batch, grid_h * grid_w, -1)
            features = features.unsqueeze(2) + self.slot_embed.view(1, 1, slots, -1)
            feature_chunks.append(features.reshape(batch, grid_h * grid_w * slots, -1))
            anchor_valid_chunks.append(
                _repeat_slots(view.valid.reshape(batch, grid_h * grid_w), slots)
            )
            count = grid_h * grid_w * slots
            device = image.device
            camera_index_chunks.append(
                torch.full((count,), camera_index, dtype=torch.long, device=device)
            )
            rows = torch.arange(grid_h, device=device).view(grid_h, 1).expand(grid_h, grid_w)
            cols = torch.arange(grid_w, device=device).view(1, grid_w).expand(grid_h, grid_w)
            row_chunks.append(_repeat_slots(rows.reshape(-1), slots))
            col_chunks.append(_repeat_slots(cols.reshape(-1), slots))
            sub = torch.arange(slots, device=device).view(1, slots).expand(grid_h * grid_w, slots)
            subslot_chunks.append(sub.reshape(-1))
            uv = view.uv.reshape(grid_h * grid_w, 2)
            uv_chunks.append(_repeat_cells_first(uv, slots))
            slot_chunks.append(torch.arange(count, device=device, dtype=torch.long) + slot_offset)
            slot_offset += count

        features = torch.cat(feature_chunks, dim=1)
        anchors = torch.cat(anchor_chunks, dim=1)
        colors = torch.cat(color_chunks, dim=1).clamp(0, 1)
        tangents = torch.cat(tangent_chunks, dim=1)
        cell_edge = torch.cat(cell_edge_chunks, dim=1)
        wire_prob = _repeat_slots(cell_edge, slots).clamp(0, 1)
        valid = allocate_valid(cell_edge, slots, cfg.edge_threshold, cfg.max_extra_gaussians)
        subslot = torch.cat(subslot_chunks, dim=0)
        geometry = self.geometry(features)
        appearance = self.appearance(features)
        dynamics = self.dynamics(features)
        offset = torch.tanh(geometry["offset_raw"]) * cfg.offset_radius_m
        spread = self.slot_frac[subslot].to(dtype=anchors.dtype)
        tangent_norm = tangents.norm(dim=-1, keepdim=True)
        tangent_unit = tangents / tangent_norm.clamp_min(1e-8)
        has_tangent = (tangent_norm > 1e-4).to(dtype=anchors.dtype)
        extra = (subslot > 0).to(dtype=anchors.dtype)
        along = (spread * cfg.along_radius_m)[:, None] * tangent_unit * has_tangent * extra[:, None]
        xyz = anchors + offset + along
        log_scale = geometry["log_scale"] + self.log_scale_bias[subslot].to(dtype=features.dtype)
        scale = activate_scales(
            log_scale,
            wire_prob,
            min_scale=cfg.min_scale_m,
            max_scale=cfg.max_scale_m,
            wire_transverse_max=cfg.wire_transverse_max_m,
            wire_long_max=cfg.wire_long_max_m,
        )
        residual = quat_normalize(geometry["quat_raw"])
        aligned = quat_mul(quaternion_from_x_axis(tangents), residual)
        quat = quat_slerp(residual, aligned, wire_prob)
        sh = appearance["sh_residual"]
        dc = sh_dc_from_rgb(colors)
        sh = torch.cat((sh[..., :1, :] + dc.unsqueeze(-2), sh[..., 1:, :]), dim=-2)
        opacity = torch.sigmoid(appearance["opacity_logit"])
        opacity = torch.where(valid.unsqueeze(-1), opacity, torch.zeros_like(opacity))
        packed = pack_gaussians(xyz, scale, quat, opacity, sh, dynamics["velocity"])
        return GaussianCloud(
            packed=packed,
            valid=valid,
            wire_prob=wire_prob,
            slot_id=torch.cat(slot_chunks, dim=0),
            camera_index=torch.cat(camera_index_chunks, dim=0),
            row=torch.cat(row_chunks, dim=0),
            col=torch.cat(col_chunks, dim=0),
            subslot=subslot,
            uv=torch.cat(uv_chunks, dim=0),
            camera_order=tuple(names),
            anchor_valid=torch.cat(anchor_valid_chunks, dim=1),
        )


class FeedforwardSplatter(nn.Module):
    """Phase 2 lifter plus the Phase 3 predictor. One forward, one cloud."""

    def __init__(self, lifter: nn.Module, predictor: SplatPredictor) -> None:
        super().__init__()
        self.lifter = lifter
        self.predictor = predictor

    def forward(
        self,
        images: dict[str, Tensor],
        cameras: dict[str, dict[str, Tensor]],
        highres_images: dict[str, Tensor] | None = None,
        match: str = "hybrid",
    ) -> GaussianCloud:
        lifted = self.lifter(images, cameras, match=match)
        return self.predictor(lifted, images, cameras, highres_images=highres_images)


def temporal_residual(
    cloud_t: GaussianCloud,
    cloud_tp1: GaussianCloud,
    dt: float | Tensor,
) -> dict[str, Tensor]:
    """Slot-aligned residual inside Andy's ``L_temp``.

    ``predicted_xyz = x_t + v_t * dt`` and ``residual = predicted_xyz - x_{t+1}``,
    both ``[B, N, 3]``. ``weight`` is ``[B, N]`` and is 1 only where the slot
    is valid at both times. ``dt`` is seconds, a scalar or ``[B]``.

    Raises if the slot layouts differ. Phase 4 should not Hungarian-match
    these tensors; the index is the correspondence.
    """
    if not cloud_t.same_layout(cloud_tp1):
        raise ValueError(
            "L_temp indexes Gaussian i at t and t+1. These clouds do not share a "
            "slot layout: camera order, resolution, or slots_per_cell changed."
        )
    parts_t = cloud_t.unpack()
    parts_next = cloud_tp1.unpack()
    if not torch.is_tensor(dt):
        step = parts_t["xyz"].new_tensor(float(dt))
    else:
        step = dt.to(device=parts_t["xyz"].device, dtype=parts_t["xyz"].dtype)
    while step.ndim < 3:
        step = step.unsqueeze(-1)
    predicted = parts_t["xyz"] + parts_t["velocity"] * step
    residual = predicted - parts_next["xyz"]
    weight = (cloud_t.valid & cloud_tp1.valid).to(dtype=residual.dtype)
    return {"predicted_xyz": predicted, "residual": residual, "weight": weight}


def _scale_bias(config: SplatConfig) -> Tensor:
    bias = torch.empty(config.slots_per_cell, 3)
    bias[0].fill_(math.log(config.base_scale_m))
    for slot in range(1, config.slots_per_cell):
        bias[slot, 0] = math.log(config.extra_long_scale_m)
        bias[slot, 1] = math.log(config.extra_thin_scale_m)
        bias[slot, 2] = math.log(config.extra_thin_scale_m)
    return bias


def _slot_frac(slots: int) -> Tensor:
    """Where extra subslots sit along the wire tangent, in ``[-1, 1]``."""
    frac = torch.zeros(slots)
    extras = slots - 1
    if extras == 1:
        frac[1] = 0.5
    elif extras > 1:
        frac[1:] = torch.linspace(-1.0, 1.0, extras)
    return frac


def _tangent_world(tangent_xy: Tensor, K: Tensor, R: Tensor) -> Tensor:
    """Image-plane tangent ``[B, 2, H, W]`` to a flight-frame direction ``[B, H*W, 3]``.

    A fronto-parallel wire has camera-frame direction ``(du / fx, dv / fy, 0)``.
    ``R`` is world-to-camera, so the world direction is ``R^T`` times that.
    ``K`` and ``R`` are ``[B, 3, 3]``. The same-FOV assumption is that ``tangent_xy``
    was measured on an image this ``K`` describes, or on a uniform resize of it
    (the direction, not the pixel length, is what we keep).
    """
    du = tangent_xy[:, 0]
    dv = tangent_xy[:, 1]
    fx = K[:, 0, 0][:, None, None]
    fy = K[:, 1, 1][:, None, None]
    direction_cam = torch.stack((du / fx, dv / fy, torch.zeros_like(du)), dim=-1)
    direction_world = torch.einsum("bji,bhwj->bhwi", R, direction_cam)
    batch, grid_h, grid_w, _ = direction_world.shape
    return direction_world.reshape(batch, grid_h * grid_w, 3)


def _repeat_slots(tensor: Tensor, slots: int) -> Tensor:
    """Repeat each cell so the subslot is the fast index.

    ``[cells]`` -> ``[cells * slots]``. ``[B, cells]`` -> ``[B, cells * slots]``.
    ``[B, cells, D]`` -> ``[B, cells * slots, D]``. A ``[cells, D]`` uv uses
    :func:`_repeat_cells_first` instead; both layouts are rank 2.
    """
    if tensor.ndim == 1:
        return tensor.unsqueeze(1).expand(tensor.shape[0], slots).reshape(-1)
    if tensor.ndim == 2:
        expanded = tensor.unsqueeze(2).expand(tensor.shape[0], tensor.shape[1], slots)
        return expanded.reshape(tensor.shape[0], tensor.shape[1] * slots)
    if tensor.ndim != 3:
        raise ValueError(f"cannot repeat slots of shape {tuple(tensor.shape)}")
    expanded = tensor.unsqueeze(2).expand(tensor.shape[0], tensor.shape[1], slots, tensor.shape[2])
    return expanded.reshape(tensor.shape[0], tensor.shape[1] * slots, tensor.shape[2])


def _repeat_cells_first(tensor: Tensor, slots: int) -> Tensor:
    """``[cells, D]`` -> ``[cells * slots, D]`` with the subslot as the fast index."""
    expanded = tensor.unsqueeze(1).expand(tensor.shape[0], slots, tensor.shape[-1])
    return expanded.reshape(-1, tensor.shape[-1])
