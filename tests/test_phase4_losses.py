"""Phase 4 losses against closed-form values, plus the ego-motion correspondence."""

import math

import numpy as np
import torch

from aerosplat.geometry.occupancy import OCCUPIED, UNKNOWN, OccupancyGrid
from aerosplat.models.gaussians import GaussianCloud, pack_gaussians
from aerosplat.models.heads import OccupancyHead, PlaceholderOccupancyHead
from aerosplat.models.losses import (
    LossWeights,
    composite_loss,
    occupancy_bce,
    occupancy_target_from_grid,
    photometric_l1,
    sparsity_loss,
    ssim_loss,
    velocity_supervision,
)
from aerosplat.models.splats import temporal_residual
from aerosplat.models.temporal import reprojected_temporal_loss, velocity_from_positions


def test_photo_is_a_per_camera_mean_of_per_pixel_l1():
    rendered = {
        "front": torch.zeros(1, 3, 1, 1),
        "down": torch.zeros(1, 3, 2, 2),
    }
    targets = {
        "front": torch.ones(1, 3, 1, 1),
        "down": torch.zeros(1, 3, 2, 2),
    }
    # Front is wrong by 1 on its only pixel. Down is exact. A pixel-pooled
    # mean would be 3/15 = 0.2 because down has more pixels. The loss is the
    # unweighted mean of the two cameras.
    assert abs(float(photometric_l1(rendered, targets)) - 0.5) < 1e-6


def test_ssim_of_flat_fields_matches_the_closed_form():
    identical = torch.full((1, 3, 8, 8), 0.4)
    assert float(ssim_loss({"front": identical}, {"front": identical.clone()})) < 1e-3
    prediction = torch.full((1, 3, 8, 8), 0.2)
    target = torch.full((1, 3, 8, 8), 0.8)
    # Variances vanish. C1 = 1e-4, C2 cancels.
    # SSIM = (2*0.2*0.8 + 1e-4) / (0.2^2 + 0.8^2 + 1e-4) = 0.3201 / 0.6801
    expected = 1.0 - (0.3201 / 0.6801)
    value = float(ssim_loss({"front": prediction}, {"front": target}, window_size=5))
    assert abs(value - expected) < 1e-4


def test_sparsity_penalizes_oversized_opaque_gaussians_and_ignores_wire_prob():
    # Volumes 2, 0.5, 10. tau = 1. Opacities 1, 0.5, 0.
    # Hinge: 1*(2-1) + 0.5*0 + 0*(10-1) = 1. Divide by N=3.
    scales = torch.tensor([[[2.0, 1.0, 1.0], [0.5, 1.0, 1.0], [10.0, 1.0, 1.0]]])
    opacity = torch.tensor([[[1.0], [0.5], [0.0]]])
    packed = _pack(scales, opacity)
    loss = sparsity_loss(packed, tau_max=1.0)
    assert abs(float(loss) - (1.0 / 3.0)) < 1e-5
    tiny = _pack(torch.full((1, 2, 3), 0.1), torch.ones(1, 2, 1))
    assert float(sparsity_loss(tiny, tau_max=1.0)) == 0.0
    huge = _pack(torch.full((1, 1, 3), 2.0), torch.ones(1, 1, 1))
    assert float(sparsity_loss(huge, tau_max=1.0)) > 0.0
    # The packed tensor has no wire channel. Two clouds that differ only in
    # wire_prob therefore share the loss: the hinge is ungated.
    first = _cloud(torch.zeros(1, 1, 3), torch.zeros(1, 1, 3), wire=0.0, scale=2.0, opacity=1.0)
    second = _cloud(torch.zeros(1, 1, 3), torch.zeros(1, 1, 3), wire=1.0, scale=2.0, opacity=1.0)
    assert float(sparsity_loss(first.packed, 1.0)) == float(sparsity_loss(second.packed, 1.0))


def test_occupancy_bce_matches_the_two_voxel_closed_form_and_masks_unknown():
    # Logit 0 -> p = 1/2. Known cells: one occupied, one free. Unknown dropped.
    logits = torch.zeros(1, 2, 2)
    target = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    known = torch.tensor([[[True, True], [False, False]]])
    ln2 = math.log(2.0)
    assert abs(float(occupancy_bce(logits, target, known)) - ln2) < 1e-5
    assert abs(float(occupancy_bce(logits, target, known, reduction="sum")) - 2 * ln2) < 1e-5
    # pos_weight 2 multiplies only the positive term: (2*ln2 + ln2) / 2.
    weighted = occupancy_bce(logits, target, known, pos_weight=2.0)
    assert abs(float(weighted) - 1.5 * ln2) < 1e-5
    # gamma 2, p_t = 1/2, (1 - p_t)^2 = 1/4.
    focal = occupancy_bce(logits, target, known, focal_gamma=2.0)
    assert abs(float(focal) - 0.25 * ln2) < 1e-5
    state = np.zeros((1, 2, 2), dtype=np.uint8)
    state[0, 0, 0] = int(OCCUPIED)
    state[0, 0, 1] = 1
    state[0, 1, 0] = int(UNKNOWN)
    state[0, 1, 1] = int(UNKNOWN)
    grid = OccupancyGrid(
        state=state,
        hit_count=np.zeros_like(state, dtype=np.uint16),
        origin=(0.0, 0.0, 0.0),
        voxel_size=0.5,
    )
    grid_target, grid_known = occupancy_target_from_grid(grid)
    assert torch.equal(grid_known, known)
    assert torch.equal(grid_target, target)
    head = PlaceholderOccupancyHead()
    logits_b = head(2, (1, 2, 2))
    assert logits_b.shape == (2, 1, 2, 2)
    assert logits_b.requires_grad
    broadcast_target = target.unsqueeze(1).expand_as(logits_b)
    broadcast_known = known.unsqueeze(1).expand_as(logits_b)
    # Equal counts of free and occupied cancel at logit 0. A positive weight does not.
    grad = torch.autograd.grad(
        occupancy_bce(logits_b, broadcast_target, broadcast_known, pos_weight=2.0),
        head.logit,
    )[0]
    assert float(grad.abs()) > 0
    raised = False
    try:
        OccupancyHead()(torch.zeros(1, 4))
    except NotImplementedError:
        raised = True
    assert raised


def test_velocity_supervision_is_squared_l2_on_valid_slots():
    cloud = _cloud(torch.zeros(1, 2, 3), torch.tensor([[[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]]]))
    cloud.valid[:, 1] = False
    loss = velocity_supervision(cloud, torch.zeros(3))
    assert abs(float(loss) - 1.0) < 1e-5
    actor = velocity_from_positions(
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([0.4, 0.0, 0.0]),
        dt=0.1,
    )
    assert torch.allclose(actor, torch.tensor([4.0, 0.0, 0.0]))


def test_static_scene_with_a_moving_camera_has_zero_temporal_loss_at_rest():
    current, nxt, cameras, _ = _ego_pair(shift=(0.0, 0.0, 0.0))
    out = reprojected_temporal_loss(current, nxt, cameras, dt=0.1)
    assert int(out["weight"].sum()) > 0
    assert float(out["loss"]) < 1e-4
    # The same pixels are not the same surface points. The slot residual is
    # the quantity Phase 4 refused to train.
    slot = temporal_residual(current, nxt, dt=0.1)
    slot_sq = slot["residual"].pow(2).sum(-1)[slot["weight"] > 0].mean()
    assert float(slot_sq) > 0.01
    # A point that actually moved by (3, 4, 0) is a squared L2 of 25, not 5.
    moved, moved_next, moved_cameras, _ = _ego_pair(shift=(3.0, 4.0, 0.0))
    moved_loss = reprojected_temporal_loss(moved, moved_next, moved_cameras, dt=0.5)
    assert abs(float(moved_loss["loss"]) - 25.0) < 1e-3
    # The velocity that explains that motion drives the residual back to zero.
    explained = _with_velocity(moved, (3.0 / 0.5, 4.0 / 0.5, 0.0))
    fixed = reprojected_temporal_loss(explained, moved_next, moved_cameras, dt=0.5)
    assert float(fixed["loss"]) < 1e-3


def test_invalid_slots_and_occluders_drop_out_of_temporal_loss():
    current, nxt, cameras, _ = _ego_pair(shift=(0.0, 0.0, 0.0))
    current.valid[:] = False
    dead = reprojected_temporal_loss(current, nxt, cameras, dt=0.1)
    assert float(dead["loss"]) == 0.0
    assert int(dead["weight"].sum()) == 0
    current.valid[:] = True
    nxt.valid[:] = False
    dead_next = reprojected_temporal_loss(current, nxt, cameras, dt=0.1)
    assert int(dead_next["weight"].sum()) == 0
    # A closer surface in front of every correspondence is an occlusion.
    _, closer, closer_cameras, _ = _ego_pair(shift=(0.0, 0.0, 0.0), depth_next=1.0)
    hidden = reprojected_temporal_loss(current, closer, closer_cameras, dt=0.1, depth_abs=0.25)
    assert int(hidden["weight"].sum()) == 0


def test_composite_uses_the_photometric_weights_and_leaves_occ_at_zero():
    rendered = {"front": torch.zeros(1, 3, 4, 4)}
    targets = {"front": torch.ones(1, 3, 4, 4)}
    weights = LossWeights(photo=0.8, ssim=0.0, temp=0.0, occ=0.0, sparse=0.0, velocity=0.0)
    out = composite_loss(rendered, targets, weights=weights)
    photo = float(out.photo)
    assert abs(photo - 1.0) < 1e-5
    assert abs(float(out.total) - 0.8) < 1e-5
    assert float(out.occ) == 0.0
    assert float(out.temp) == 0.0


def _pack(scale, opacity):
    batch, count, _ = scale.shape
    quat = torch.zeros(batch, count, 4)
    quat[..., 0] = 1
    return pack_gaussians(
        torch.zeros(batch, count, 3),
        scale,
        quat,
        opacity,
        torch.zeros(batch, count, 16, 3),
        torch.zeros(batch, count, 3),
    )


def _cloud(xyz, velocity, *, wire=0.0, scale=0.1, opacity=0.4, valid=None):
    batch, count, _ = xyz.shape
    if valid is None:
        valid = torch.ones(batch, count, dtype=torch.bool)
    quat = torch.zeros(batch, count, 4)
    quat[..., 0] = 1
    return GaussianCloud.from_activated(
        xyz,
        torch.full((batch, count, 3), scale),
        quat,
        torch.full((batch, count, 1), opacity),
        torch.zeros(batch, count, 16, 3),
        velocity,
        valid=valid,
        wire_prob=torch.full((batch, count), wire),
        slot_id=torch.arange(count),
        camera_index=torch.zeros(count, dtype=torch.long),
        row=torch.arange(count) // 4,
        col=torch.arange(count) % 4,
        subslot=torch.zeros(count, dtype=torch.long),
        uv=torch.zeros(count, 2),
        camera_order=("front",),
        anchor_valid=torch.ones(batch, count, dtype=torch.bool),
    )


def _with_velocity(cloud, velocity):
    parts = cloud.unpack()
    speed = parts["velocity"].clone()
    speed[:] = torch.tensor(velocity, dtype=speed.dtype)
    return GaussianCloud.from_activated(
        parts["xyz"],
        parts["scale"],
        parts["quat"],
        parts["opacity"],
        parts["sh"],
        speed,
        valid=cloud.valid.clone(),
        wire_prob=cloud.wire_prob.clone(),
        slot_id=cloud.slot_id.clone(),
        camera_index=cloud.camera_index.clone(),
        row=cloud.row.clone(),
        col=cloud.col.clone(),
        subslot=cloud.subslot.clone(),
        uv=cloud.uv.clone(),
        camera_order=cloud.camera_order,
        anchor_valid=cloud.anchor_valid.clone(),
    )


def _ego_pair(shift, depth_next=4.0):
    """Fronto-parallel plane. The camera translates. ``shift`` moves the surface."""
    grid = 4
    stride = 2.0
    offset = 0.5
    coords = offset + stride * torch.arange(grid)
    vv, uu = torch.meshgrid(coords, coords, indexing="ij")
    uv = torch.stack((uu, vv), dim=-1)
    height = width = 8
    focal = 8.0
    intrinsics = torch.tensor(
        [[focal, 0.0, (width - 1) / 2], [0.0, focal, (height - 1) / 2], [0.0, 0.0, 1.0]]
    )
    identity = torch.eye(3)
    translation_t = torch.zeros(3)
    # Camera center moves +0.4 m along world X. t = -R^T center.
    translation_next = torch.tensor([-0.4, 0.0, 0.0])
    points_t = _unproject(uv, 4.0, intrinsics, identity, translation_t)
    points_next = _unproject(uv, depth_next, intrinsics, identity, translation_next)
    points_next = points_next + torch.tensor(shift)
    current = _grid_cloud(points_t, uv)
    nxt = _grid_cloud(points_next, uv)
    # Border cells slide out of the bilinear neighborhood when the camera
    # translates. Keep them in the map (so interior samples have neighbors)
    # and drop them from the source mask.
    border = (current.row == 0) | (current.col == 0)
    border = border | (current.row == grid - 1) | (current.col == grid - 1)
    current.valid[:, border] = False
    cameras = {
        "front": {
            "K": intrinsics.view(1, 3, 3),
            "R": identity.view(1, 3, 3),
            "t": translation_next.view(1, 3),
            "height": height,
            "width": width,
        }
    }
    return current, nxt, cameras, uv


def _unproject(uv, depth, K, R, t):
    u = uv[..., 0]
    v = uv[..., 1]
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    camera = torch.stack((x, y, torch.full_like(x, depth)), dim=-1)
    return torch.einsum("ij,...j->...i", R.T, camera - t)


def _grid_cloud(points, uv):
    grid_h, grid_w, _ = points.shape
    count = grid_h * grid_w
    xyz = points.reshape(1, count, 3)
    row = torch.arange(grid_h).view(grid_h, 1).expand(grid_h, grid_w).reshape(-1)
    col = torch.arange(grid_w).view(1, grid_w).expand(grid_h, grid_w).reshape(-1)
    quat = torch.zeros(1, count, 4)
    quat[..., 0] = 1
    return GaussianCloud.from_activated(
        xyz,
        torch.full((1, count, 3), 0.05),
        quat,
        torch.full((1, count, 1), 0.5),
        torch.zeros(1, count, 16, 3),
        torch.zeros(1, count, 3),
        valid=torch.ones(1, count, dtype=torch.bool),
        wire_prob=torch.zeros(1, count),
        slot_id=torch.arange(count),
        camera_index=torch.zeros(count, dtype=torch.long),
        row=row,
        col=col,
        subslot=torch.zeros(count, dtype=torch.long),
        uv=uv.reshape(count, 2),
        camera_order=("front",),
        anchor_valid=torch.ones(1, count, dtype=torch.bool),
    )
