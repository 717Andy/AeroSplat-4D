"""Corridor queries against an occupancy volume.

This is the function ``GET /api/v1/occupancy`` will call. It does not run the
ViT. A trajectory is a polyline in the flight frame. The corridor is a tube
of ``corridor_radius_m`` around it.

``risk`` is the highest occupancy or wire probability of any cell the tube
touches, in ``[0, 1]``. ``nearest_distance_m`` is the distance from the
polyline to the nearest obstacle. Solid cells use the distance to the voxel
box. Wire cells use the distance to the predicted line segment, so a cable
a few centimeters off the voxel center is not treated as a solid 0.25 m cube.

The search is limited to cells near the trajectory (the corridor, plus a
margin for the nearest-obstacle distance). A 32³ grid and a few dozen
waypoints is a sub-millisecond lookup on CPU. The ViT that fills the grid
is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from aerosplat.models.occupancy_head import OccupancyVolume


@dataclass
class CollisionReport:
    """One query. Distances are meters. Risk is a probability in ``[0, 1]``."""

    risk: Tensor
    nearest_distance_m: Tensor
    wire_risk: Tensor
    solid_risk: Tensor

    def item(self) -> dict[str, float]:
        return {
            "risk": float(self.risk.detach().reshape(-1)[0].cpu()),
            "nearest_distance_m": float(self.nearest_distance_m.detach().reshape(-1)[0].cpu()),
            "wire_risk": float(self.wire_risk.detach().reshape(-1)[0].cpu()),
            "solid_risk": float(self.solid_risk.detach().reshape(-1)[0].cpu()),
        }


def query_collision(
    volume: OccupancyVolume,
    trajectory: Tensor,
    corridor_radius_m: float,
    *,
    occ_threshold: float = 0.5,
    wire_threshold: float = 0.5,
    search_margin_m: float | None = None,
) -> CollisionReport:
    """Risk and clearance for a tube around ``trajectory``.

    ``trajectory`` is ``[T, 3]`` or ``[B, T, 3]`` in the same frame as
    ``volume.origin``. ``B`` must match the volume. A batch of trajectories
    returns a batch of reports.
    """
    if corridor_radius_m < 0:
        raise ValueError("corridor_radius_m must be >= 0")
    if not 0.0 < occ_threshold < 1.0 or not 0.0 < wire_threshold < 1.0:
        raise ValueError("thresholds must lie in (0, 1)")
    logits = volume.occupancy_logits
    points = trajectory.to(dtype=logits.dtype, device=logits.device)
    if points.ndim == 2:
        points = points.unsqueeze(0)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"trajectory must be [T, 3] or [B, T, 3], got {tuple(trajectory.shape)}")
    batch = volume.occupancy_logits.shape[0]
    if points.shape[0] != batch:
        raise ValueError(f"trajectory batch {points.shape[0]} != volume batch {batch}")
    voxel = volume.voxel_size.to(dtype=points.dtype, device=points.device)
    # ``search_margin_m`` is accepted so a caller can ask for a local crop.
    # The default scans every cell above threshold. A 32³ volume has few
    # occupied cells; the lookup stays well under a millisecond.
    del search_margin_m
    occupancy = volume.occupancy
    wire = volume.wire
    # Strictly above the threshold. A zero logit is sigmoid 0.5, and an
    # untrained head must not report the whole grid as blocked.
    solid = occupancy > occ_threshold
    cable = wire > wire_threshold
    risks = []
    wire_risks = []
    solid_risks = []
    distances = []
    for index in range(batch):
        report = _one_query(
            points[index],
            occupancy[index],
            wire[index],
            solid[index],
            cable[index],
            volume.wire_direction[index],
            volume.wire_offset[index],
            volume.origin[index],
            float(voxel),
            float(corridor_radius_m),
        )
        risks.append(report[0])
        wire_risks.append(report[1])
        solid_risks.append(report[2])
        distances.append(report[3])
    return CollisionReport(
        risk=torch.stack(risks),
        nearest_distance_m=torch.stack(distances),
        wire_risk=torch.stack(wire_risks),
        solid_risk=torch.stack(solid_risks),
    )


def _one_query(
    points: Tensor,
    occupancy: Tensor,
    wire: Tensor,
    solid: Tensor,
    cable: Tensor,
    direction: Tensor,
    offset: Tensor,
    origin: Tensor,
    voxel: float,
    radius: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    interesting = solid | cable
    dtype = points.dtype
    if not bool(interesting.any()):
        zero = points.new_zeros(())
        far = points.new_tensor(float("inf"))
        return zero, zero, zero, far
    coords = interesting.nonzero(as_tuple=False)
    gx = coords[:, 0]
    gy = coords[:, 1]
    gz = coords[:, 2]
    voxel_origin = origin + torch.stack((gx, gy, gz), dim=-1).to(dtype) * voxel
    box_min = voxel_origin
    box_max = voxel_origin + voxel
    # Distance from every waypoint to every selected box. T and K stay small
    # because only cells above the threshold are kept.
    delta_low = box_min[None, :, :] - points[:, None, :]
    delta_high = points[:, None, :] - box_max[None, :, :]
    outside = torch.maximum(delta_low, delta_high).clamp_min(0)
    box_distance = outside.norm(dim=-1)
    is_cable = cable[gx, gy, gz]
    is_solid = solid[gx, gy, gz]
    centers = voxel_origin + 0.5 * voxel
    # offset and direction are [3, X, Y, Z]. Index every spatial axis.
    line_point = centers + offset[:, gx, gy, gz].transpose(0, 1) * voxel
    line_dir = direction[:, gx, gy, gz].transpose(0, 1)
    line_dir = line_dir / line_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    half = 0.5 * voxel * (3.0**0.5)
    cable_distance = _point_segment_distance(points, line_point, line_dir, half)
    # A wire cell is the cable, not a solid cube. Measuring the box would
    # block the whole 0.25 m cell around a 2 cm line.
    obstacle = torch.where(is_cable[None, :], cable_distance, box_distance)
    nearest = obstacle.min()
    touched = obstacle <= radius
    if not bool(touched.any()):
        zero = points.new_zeros(())
        return zero, zero, zero, nearest
    occ_sel = occupancy[gx, gy, gz]
    wire_sel = wire[gx, gy, gz]
    solid_keep = touched & is_solid[None, :]
    wire_keep = touched & is_cable[None, :]
    solid_risk = occ_sel[None, :].expand_as(touched).masked_fill(~solid_keep, 0).max()
    wire_risk = wire_sel[None, :].expand_as(touched).masked_fill(~wire_keep, 0).max()
    risk = torch.maximum(solid_risk, wire_risk)
    return risk, wire_risk, solid_risk, nearest


def _point_segment_distance(
    points: Tensor,
    center: Tensor,
    direction: Tensor,
    half_length: float,
) -> Tensor:
    """``points`` ``[T, 3]``, segment centers ``[K, 3]``. Return ``[T, K]``."""
    relative = points[:, None, :] - center[None, :, :]
    along = (relative * direction[None, :, :]).sum(dim=-1).clamp(-half_length, half_length)
    closest = center[None, :, :] + along[..., None] * direction[None, :, :]
    return (points[:, None, :] - closest).norm(dim=-1)


def wire_direction_smoothness(wire_logits: Tensor, direction: Tensor) -> Tensor:
    """Neighboring wire cells should agree on tangent.

    A hanging cable is locally straight at the scale of one coarse voxel, and
    its tangent turns slowly from cell to cell. That is the line prior. It is
    not a single catenary fitted across the corridor: a crane cable can be
    vertical, and a power line can sag, and both are a chain of segments.
    The weight is the product of the two probabilities, detached, so the loss
    aligns directions where the head already believes there is a wire.
    """
    probability = torch.sigmoid(wire_logits).detach()
    # ``wire_logits`` is ``[B, X, Y, Z]``. ``direction`` is ``[B, 3, X, Y, Z]``.
    if direction.ndim != 5 or direction.shape[1] != 3:
        raise ValueError(f"wire direction must be [B, 3, X, Y, Z], got {tuple(direction.shape)}")
    if tuple(direction.shape[2:]) != tuple(probability.shape[1:]):
        raise ValueError(
            f"wire direction grid {tuple(direction.shape[2:])} != "
            f"logits {tuple(probability.shape[1:])}"
        )
    loss = wire_logits.new_zeros(())
    weight = wire_logits.new_zeros(())
    for axis in range(3):
        spatial = axis + 1
        if probability.shape[spatial] < 2:
            continue
        channel_axis = axis + 2
        left = direction.narrow(channel_axis, 0, direction.shape[channel_axis] - 1)
        right = direction.narrow(channel_axis, 1, direction.shape[channel_axis] - 1)
        cosine = (left * right).sum(dim=1).clamp(-1.0, 1.0)
        left_p = probability.narrow(spatial, 0, probability.shape[spatial] - 1)
        right_p = probability.narrow(spatial, 1, probability.shape[spatial] - 1)
        pair = left_p * right_p
        loss = loss + ((1.0 - cosine) * pair).sum()
        weight = weight + pair.sum()
    if float(weight.detach()) == 0.0:
        return wire_logits.sum() * 0.0
    return loss / weight.clamp_min(1e-8)


def wire_point_loss(volume: OccupancyVolume, points: Tensor) -> Tensor:
    """Mean distance, in voxels, from known cable points to the predicted line.

    ``points`` is ``[M, 3]`` in the flight frame. Points outside the grid are
    ignored. This is how a simulator wire sample supervises the sub-voxel
    offset: the coarse cell can be 0.25 m and the point still pulls the line
    onto the cable.
    """
    if points.numel() == 0:
        return volume.wire_logits.sum() * 0.0
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"wire points must be [M, 3], got {tuple(points.shape)}")
    origin = volume.origin[0]
    voxel = volume.voxel_size
    dims = volume.occupancy_logits.shape[-3:]
    scaled = (points.to(dtype=origin.dtype, device=origin.device) - origin) / voxel
    ijk = scaled.floor().long()
    inside = (
        (ijk[:, 0] >= 0)
        & (ijk[:, 0] < dims[0])
        & (ijk[:, 1] >= 0)
        & (ijk[:, 1] < dims[1])
        & (ijk[:, 2] >= 0)
        & (ijk[:, 2] < dims[2])
    )
    if not bool(inside.any()):
        return volume.wire_logits.sum() * 0.0
    ijk = ijk[inside]
    kept = points.to(dtype=origin.dtype, device=origin.device)[inside]
    center = origin + (ijk.to(dtype=origin.dtype) + 0.5) * voxel
    shift = volume.wire_offset[0, :, ijk[:, 0], ijk[:, 1], ijk[:, 2]].transpose(0, 1) * voxel
    line_point = center + shift
    direction = volume.wire_direction[0, :, ijk[:, 0], ijk[:, 1], ijk[:, 2]].transpose(0, 1)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    relative = kept - line_point
    along = (relative * direction).sum(dim=-1, keepdim=True)
    perpendicular = (relative - along * direction).norm(dim=-1)
    return (perpendicular / voxel).mean()
