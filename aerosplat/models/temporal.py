"""Reprojected correspondence for ``L_temp``.

Andy's sum indexes Gaussian ``i`` at ``t`` and at ``t+1``::

    L_temp = (1 / N_splats) sum_i || (x_i^(t) + v_i^(t) Δt) - x_i^(t+1) ||_2^2

A feedforward cloud has no tracked id. Phase 3's slot index is a pixel: the
same ``(camera, row, col, subslot)`` next frame. Under ego-motion that pixel
sees a different surface point, so a static world and ``v = 0`` still produce
a residual. :func:`aerosplat.models.splats.temporal_residual` is that
pixel-slot quantity. It is not the training loss.

The training loss builds the missing correspondence by reprojection:

1. Move the time-``t`` point in the flight frame, ``p = x_t + v_t Δt``.
2. Project ``p`` into the time-``t+1`` cameras with the known poses.
3. Read the time-``t+1`` predicted point at that pixel by bilinear sampling
   of the base-slot point map, falling back to the nearest valid slot when
   the bilinear neighborhood is empty.
4. Drop the splat when it is out of view, occluded (a closer surface sits in
   front of ``p``), or invalid at either time.
5. Average the squared L2 over the splats that remain.

The owning camera is preferred when it still sees ``p``. Another camera is
used only when the owner is masked out, so a point that leaves one frustum
can still land on a neighbor's map.
"""

from __future__ import annotations

import torch
from torch import Tensor

from aerosplat.models.gaussians import GaussianCloud


def velocity_from_positions(position_t: Tensor, position_tp1: Tensor, dt: float | Tensor) -> Tensor:
    """World-frame m/s from two logged actor origins.

    Cosys-AirSim and Colosseum can record ``simGetObjectPose`` for a named
    actor. Convert that origin into the flight frame first. The returned
    vector is one rigid velocity. Assigning it to individual Gaussians needs
    a mask (segmentation, or "this actor is the whole dynamic object"). A
    static world is the zero vector. ``dt`` is seconds and must be nonzero.
    """
    step = dt if torch.is_tensor(dt) else position_t.new_tensor(float(dt))
    if float(step.reshape(-1)[0]) == 0.0:
        raise ValueError("dt must be nonzero to difference actor poses")
    origin_t = torch.as_tensor(position_t, device=step.device, dtype=step.dtype).reshape(3)
    origin_next = torch.as_tensor(position_tp1, device=step.device, dtype=step.dtype).reshape(3)
    return (origin_next - origin_t) / step.reshape(-1)[0]


def point_maps(cloud: GaussianCloud) -> dict[str, dict[str, Tensor]]:
    """Base-slot xyz and validity on each camera's lift grid.

    ``xyz`` is ``[B, H, W, 3]`` in the flight frame. ``valid`` is ``[B, H, W]``.
    ``uv`` is ``[H, W, 2]`` pixel centers of those cells, the same coordinates
    the lifter unprojected. Subslots other than 0 are not a second map: a wire
    extra lives on the same cell as its base Gaussian.
    """
    parts = cloud.unpack()
    xyz = parts["xyz"]
    maps: dict[str, dict[str, Tensor]] = {}
    for camera_index, name in enumerate(cloud.camera_order):
        base = (cloud.camera_index == camera_index) & (cloud.subslot == 0)
        if not bool(base.any()):
            continue
        rows = cloud.row[base]
        cols = cloud.col[base]
        grid_h = int(rows.max().item()) + 1
        grid_w = int(cols.max().item()) + 1
        batch = xyz.shape[0]
        xyz_map = xyz.new_zeros(batch, grid_h, grid_w, 3)
        valid_map = torch.zeros(batch, grid_h, grid_w, dtype=torch.bool, device=xyz.device)
        uv_map = cloud.uv.new_zeros(grid_h, grid_w, 2)
        slot = base.nonzero(as_tuple=False).squeeze(-1)
        xyz_map[:, rows, cols] = xyz[:, slot]
        valid_map[:, rows, cols] = cloud.valid[:, slot]
        uv_map[rows, cols] = cloud.uv[slot]
        maps[name] = {
            "xyz": xyz_map,
            "valid": valid_map,
            "uv": uv_map,
            "camera_index": camera_index,
        }
    if not maps:
        raise ValueError("cloud has no base slots to build a point map from")
    return maps


def reprojected_temporal_loss(
    cloud_t: GaussianCloud,
    cloud_tp1: GaussianCloud,
    cameras_tp1: dict[str, dict[str, Tensor]],
    dt: float | Tensor,
    *,
    depth_abs: float = 0.5,
    depth_rel: float = 0.1,
    max_search: float = 1.0,
    near: float = 0.05,
) -> dict[str, Tensor]:
    """Squared L2 after reprojecting ``x_t + v_t Δt`` into the next frame.

    ``cameras_tp1`` maps a camera name to ``K`` ``[B, 3, 3]``, ``R`` ``[B, 3, 3]``,
    ``t`` ``[B, 3]``, and ``height`` / ``width``. Those poses are the ``t+1``
    capture poses, the same ones the ``t+1`` forward was conditioned on.

    Returns ``loss`` (a scalar, mean over surviving splats, or 0 when none
    survive), ``residual_sq`` ``[B, N]``, and ``weight`` ``[B, N]`` (1 where
    the splat entered the mean). Slots invalid at ``t`` or with no valid
    destination at ``t+1`` have weight 0.
    """
    if not cloud_t.same_layout(cloud_tp1):
        raise ValueError(
            "L_temp compares two clouds from the same rig. Camera order, "
            "resolution, or slots_per_cell changed between t and t+1."
        )
    parts_t = cloud_t.unpack()
    predicted = parts_t["xyz"] + parts_t["velocity"] * _dt_broadcast(dt, parts_t["xyz"])
    maps = point_maps(cloud_tp1)
    batch, count, _ = predicted.shape
    chosen = predicted.new_zeros(batch, count)
    chosen_ok = torch.zeros(batch, count, dtype=torch.bool, device=predicted.device)
    source_valid = cloud_t.valid
    for name, point_map in maps.items():
        if name not in cameras_tp1:
            continue
        camera = cameras_tp1[name]
        height, width = _hw(camera, point_map["uv"])
        u, v, depth = _project(predicted, camera["K"], camera["R"], camera["t"])
        sampled, sample_ok = _sample_point_map(
            point_map["xyz"],
            point_map["valid"],
            point_map["uv"],
            u,
            v,
            max_search=max_search,
        )
        in_image = (depth > near) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        sampled_depth = _camera_depth(sampled, camera["R"], camera["t"])
        margin = torch.maximum(
            predicted.new_full(depth.shape, float(depth_abs)),
            float(depth_rel) * depth.clamp_min(near),
        )
        # One-sided: a surface in front of p hides it. A farther sample is a
        # disagreement the residual is allowed to train, not an occlusion.
        occluded = sampled_depth < (depth - margin)
        ok = source_valid & sample_ok & in_image & ~occluded
        residual_sq = (predicted - sampled).pow(2).sum(dim=-1)
        owning = cloud_t.camera_index == int(point_map["camera_index"])
        take = (ok & ~chosen_ok) | (owning.unsqueeze(0) & ok)
        chosen = torch.where(take, residual_sq, chosen)
        chosen_ok = chosen_ok | ok
    weight = chosen_ok.to(dtype=chosen.dtype)
    if not bool(chosen_ok.any()):
        loss = predicted.sum() * 0.0
    else:
        loss = chosen[chosen_ok].mean()
    return {"loss": loss, "residual_sq": chosen, "weight": weight, "predicted_xyz": predicted}


def _dt_broadcast(dt: float | Tensor, xyz: Tensor) -> Tensor:
    if not torch.is_tensor(dt):
        step = xyz.new_tensor(float(dt))
    else:
        step = dt.to(device=xyz.device, dtype=xyz.dtype)
    while step.ndim < xyz.ndim:
        step = step.unsqueeze(-1)
    return step


def _hw(camera: dict, uv: Tensor) -> tuple[int, int]:
    if "height" in camera and "width" in camera:
        return int(camera["height"]), int(camera["width"])
    # The lift grid does not name the image. A missing size is recovered from
    # the outermost cell center plus one cell of padding, which matches a grid
    # built by lift_pixel_grid on that image.
    if uv.shape[0] == 1 and uv.shape[1] == 1:
        span_v = float(uv[0, 0, 1]) + 1.0
        span_u = float(uv[0, 0, 0]) + 1.0
    else:
        step_u = float(uv[0, min(1, uv.shape[1] - 1), 0] - uv[0, 0, 0]) or 1.0
        step_v = float(uv[min(1, uv.shape[0] - 1), 0, 1] - uv[0, 0, 1]) or 1.0
        span_u = float(uv[0, -1, 0]) + 0.5 * abs(step_u) + 1.0
        span_v = float(uv[-1, 0, 1]) + 0.5 * abs(step_v) + 1.0
    return max(int(span_v), 1), max(int(span_u), 1)


def _project(points: Tensor, K: Tensor, R: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    camera = torch.einsum("bij,bnj->bni", R, points) + t[:, None, :]
    depth = camera[..., 2]
    safe = torch.where(depth.abs() < 1e-6, torch.ones_like(depth), depth)
    fx = K[:, 0, 0][:, None]
    fy = K[:, 1, 1][:, None]
    cx = K[:, 0, 2][:, None]
    cy = K[:, 1, 2][:, None]
    u = fx * camera[..., 0] / safe + cx
    v = fy * camera[..., 1] / safe + cy
    return u, v, depth


def _camera_depth(points: Tensor, R: Tensor, t: Tensor) -> Tensor:
    camera = torch.einsum("bij,bnj->bni", R, points) + t[:, None, :]
    return camera[..., 2]


def _sample_point_map(
    xyz_map: Tensor,
    valid_map: Tensor,
    uv: Tensor,
    u: Tensor,
    v: Tensor,
    *,
    max_search: float,
) -> tuple[Tensor, Tensor]:
    """Bilinear sample in grid-index space, then a nearest-valid fallback.

    ``u``, ``v`` are pixels. ``uv`` is the pixel coordinate of each cell.
    Cells are equally spaced, which is the lift grid. Returns ``[B, N, 3]``
    and a bool ``[B, N]``.
    """
    grid_x, grid_y = _pixel_to_grid(uv, u, v)
    sampled, weights, bilinear_ok = _bilinear(xyz_map, valid_map, grid_x, grid_y)
    nearest, nearest_ok = _nearest_valid(xyz_map, valid_map, grid_x, grid_y, max_search)
    use_bilinear = bilinear_ok
    point = torch.where(use_bilinear.unsqueeze(-1), sampled, nearest)
    ok = use_bilinear | nearest_ok
    # ``weights`` keeps the bilinear path on the graph when it is selected.
    # A pure nearest fallback has no gradient through the index, which is the
    # same hard decision as the valid mask.
    del weights
    return point, ok


def _pixel_to_grid(uv: Tensor, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    if uv.shape[1] <= 1:
        origin_u = uv[0, 0, 0]
        step_u = uv.new_tensor(1.0)
    else:
        origin_u = uv[0, 0, 0]
        step_u = uv[0, 1, 0] - uv[0, 0, 0]
    if uv.shape[0] <= 1:
        origin_v = uv[0, 0, 1]
        step_v = uv.new_tensor(1.0)
    else:
        origin_v = uv[0, 0, 1]
        step_v = uv[1, 0, 1] - uv[0, 0, 1]
    step_u = torch.where(step_u.abs() < 1e-8, torch.ones_like(step_u), step_u)
    step_v = torch.where(step_v.abs() < 1e-8, torch.ones_like(step_v), step_v)
    return (u - origin_u) / step_u, (v - origin_v) / step_v


def _bilinear(
    xyz_map: Tensor,
    valid_map: Tensor,
    grid_x: Tensor,
    grid_y: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    batch, grid_h, grid_w, _ = xyz_map.shape
    x0 = torch.floor(grid_x).long()
    y0 = torch.floor(grid_y).long()
    x1 = x0 + 1
    y1 = y0 + 1
    wx = (grid_x - x0.to(dtype=grid_x.dtype)).clamp(0, 1)
    wy = (grid_y - y0.to(dtype=grid_y.dtype)).clamp(0, 1)

    def at(ix: Tensor, iy: Tensor) -> tuple[Tensor, Tensor]:
        inside = (ix >= 0) & (ix < grid_w) & (iy >= 0) & (iy < grid_h)
        ix_c = ix.clamp(0, max(grid_w - 1, 0))
        iy_c = iy.clamp(0, max(grid_h - 1, 0))
        batch_index = torch.arange(batch, device=xyz_map.device)[:, None]
        point = xyz_map[batch_index, iy_c, ix_c]
        valid = valid_map[batch_index, iy_c, ix_c] & inside
        return point, valid

    p00, v00 = at(x0, y0)
    p10, v10 = at(x1, y0)
    p01, v01 = at(x0, y1)
    p11, v11 = at(x1, y1)
    w00 = (1 - wx) * (1 - wy) * v00.to(dtype=xyz_map.dtype)
    w10 = wx * (1 - wy) * v10.to(dtype=xyz_map.dtype)
    w01 = (1 - wx) * wy * v01.to(dtype=xyz_map.dtype)
    w11 = wx * wy * v11.to(dtype=xyz_map.dtype)
    weight = w00 + w10 + w01 + w11
    mixed = (
        w00.unsqueeze(-1) * p00
        + w10.unsqueeze(-1) * p10
        + w01.unsqueeze(-1) * p01
        + w11.unsqueeze(-1) * p11
    )
    ok = weight > 1e-6
    sampled = mixed / weight.clamp_min(1e-6).unsqueeze(-1)
    return sampled, weight, ok


def _nearest_valid(
    xyz_map: Tensor,
    valid_map: Tensor,
    grid_x: Tensor,
    grid_y: Tensor,
    max_search: float,
) -> tuple[Tensor, Tensor]:
    batch, grid_h, grid_w, _ = xyz_map.shape
    ys = torch.arange(grid_h, device=xyz_map.device, dtype=grid_x.dtype)
    xs = torch.arange(grid_w, device=xyz_map.device, dtype=grid_x.dtype)
    dist_x = grid_x[:, :, None] - xs.view(1, 1, grid_w)
    dist_y = grid_y[:, :, None] - ys.view(1, 1, grid_h)
    # [B, N, H, W]
    dist = dist_y.unsqueeze(-1).pow(2) + dist_x.unsqueeze(-2).pow(2)
    dist = dist.masked_fill(~valid_map[:, None], float("inf"))
    flat = dist.reshape(batch, grid_x.shape[1], grid_h * grid_w)
    nearest_dist, nearest = flat.min(dim=-1)
    row = torch.div(nearest, grid_w, rounding_mode="floor")
    col = nearest.remainder(grid_w)
    batch_index = torch.arange(batch, device=xyz_map.device)[:, None]
    point = xyz_map[batch_index, row, col]
    ok = nearest_dist <= float(max_search) ** 2
    return point, ok
