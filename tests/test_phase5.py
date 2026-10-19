"""Phase 5: glare separation, wire occupancy, collision queries, ONNX parity."""

from pathlib import Path

import numpy as np
import pytest
import torch

from aerosplat.data.flare import (
    FlareAugmenter,
    FlareLibrary,
    paired_glare_target,
    write_synthetic_flare,
)
from aerosplat.export import OccupancyOnnx, RelightOnnx, export_onnx
from aerosplat.geometry.occupancy import OCCUPIED, OccupancyGrid, fuse_occupancy
from aerosplat.geometry.sun import estimate_sun_from_image, sun_direction_enu, sun_in_camera
from aerosplat.models.collision import query_collision, wire_direction_smoothness
from aerosplat.models.losses import occupancy_bce
from aerosplat.models.occupancy_head import OccupancyHead, OccupancyVolume
from aerosplat.models.rasterizer import eval_sh
from aerosplat.models.relight import (
    RelightHead,
    composite_glare,
    evaluate_relit_sh,
    rotate_sh_coeffs,
)


def test_sun_ephemeris_is_a_unit_vector_above_the_horizon_at_local_noon():
    # 2020-06-21 12:00 UTC, equator, longitude 0. The sun is high.
    vector = sun_direction_enu(1592733600.0, 0.0, 0.0)
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-6
    assert vector[2] > 0.5
    camera = sun_in_camera(vector, np.eye(3))
    assert np.allclose(camera, vector)


def test_brightest_pixel_estimates_a_forward_sun():
    image = torch.zeros(3, 16, 16)
    image[:, 8, 8] = 1.0
    intrinsics = torch.tensor([[8.0, 0.0, 8.0], [0.0, 8.0, 8.0], [0.0, 0.0, 1.0]])
    rotation = torch.eye(3)
    sun = estimate_sun_from_image(image, intrinsics, rotation)
    assert sun.shape == (3,)
    assert abs(float(sun.norm()) - 1.0) < 1e-5
    assert float(sun[2]) > 0.9


def test_sh_slider_swings_the_lobe_onto_the_new_sun():
    sh = torch.zeros(2, 16, 3)
    sh[:, 2, 0] = 1.0
    old = torch.tensor([0.0, 0.0, 1.0])
    new = torch.tensor([1.0, 0.0, 0.0])
    relit = evaluate_relit_sh(sh, new, old, new)
    direct = eval_sh(sh, old.expand(2, 3)) + 0.5
    assert torch.allclose(relit, direct, atol=1e-5)
    rotated = rotate_sh_coeffs(sh, old, new, samples=32)
    replay = eval_sh(rotated, new.expand(2, 3)) + 0.5
    assert torch.allclose(replay, direct, atol=1e-3)


def test_flare_paste_is_additive_and_masked(tmp_path: Path):
    write_synthetic_flare(tmp_path)
    library = FlareLibrary(tmp_path)
    augmenter = FlareAugmenter(library, probability=1.0, gain=(1.0, 1.0))
    clean = torch.full((3, 32, 32), 0.2)
    composite = augmenter.composite(clean, index=0, gain=1.0)
    assert composite.image.shape == clean.shape
    assert torch.allclose(composite.image, (clean + composite.glare).clamp(0, 1))
    assert float(composite.mask.sum()) > 0
    assert float(composite.glare[composite.mask.expand_as(composite.glare) > 0].mean()) > 0.05
    observed = clean + 0.4
    glare, mask = paired_glare_target(observed.clamp(0, 1), clean)
    assert float(mask.mean()) == 1.0
    assert torch.allclose(glare, torch.full_like(glare, 0.4))


def test_glare_head_separates_a_synthetic_blob():
    torch.manual_seed(0)
    clean = torch.full((1, 3, 32, 32), 0.15)
    yy = torch.linspace(-1, 1, 32)
    xx = torch.linspace(-1, 1, 32)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    blob = torch.exp(-((grid_x - 0.2) ** 2 + (grid_y + 0.15) ** 2) / 0.03)
    color = torch.tensor([1.0, 0.7, 0.3]).view(1, 3, 1, 1)
    glare_gt = blob[None, None].expand(1, 3, 32, 32) * color
    observed = composite_glare(clean, glare_gt)
    model = RelightHead(feature_dim=3, hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    sh = torch.zeros(1, 16, 3)
    sun = torch.tensor([[0.0, 0.2, 1.0]])
    # The separation loss is masked. An unmasked mean is dominated by the dark
    # pixels and the head collapses to a black layer.
    mask = (glare_gt.mean(dim=1, keepdim=True) > 0.15).to(dtype=glare_gt.dtype)
    start = None
    for _ in range(50):
        optimizer.zero_grad()
        predicted = model(observed, sh, sun, (32, 32))
        weight = mask.expand_as(predicted)
        inside_error = ((predicted - glare_gt).abs() * weight).sum()
        inside_error = inside_error / weight.sum().clamp_min(1.0)
        outside = (predicted * (1.0 - weight)).sum() / (1.0 - weight).sum().clamp_min(1.0)
        loss = inside_error + 0.02 * outside
        loss.backward()
        optimizer.step()
        if start is None:
            start = float(loss.detach())
    predicted = model(observed, sh, sun, (32, 32)).detach()
    weight = mask.expand_as(predicted) > 0.5
    inside_mean = float(predicted[weight].mean())
    outside_mean = float(predicted[~weight].mean())
    assert float(loss.detach()) < start * 0.5
    assert inside_mean > 0.2
    assert inside_mean > outside_mean + 0.1
    # The clean scene is the rasterizer output. Photometry sees the sum.
    assert torch.allclose(composite_glare(clean, predicted), (clean + predicted).clamp(0, 1))


def test_occupancy_recalls_a_synthetic_thin_wire():
    torch.manual_seed(0)
    count = 12
    xs = torch.linspace(0.4, 1.6, count)
    xyz = torch.stack((xs, torch.full_like(xs, 0.40), torch.full_like(xs, 0.40)), dim=-1)
    xyz = xyz.unsqueeze(0)
    opacity = torch.full((1, count), 0.9)
    wire = torch.ones(1, count)
    valid = torch.ones(1, count, dtype=torch.bool)
    scale = torch.tensor([0.08, 0.01, 0.01]).view(1, 1, 3).expand(1, count, 3).contiguous()
    sh_dc = torch.zeros(1, count, 3)
    origin = torch.zeros(3)
    voxel = 0.25
    grid = (8, 8, 8)
    target = torch.zeros(1, *grid)
    # The line at y=z=0.40 falls in voxel index 1. x spans indices 1..6.
    target[0, 1:7, 1, 1] = 1.0
    known = torch.ones_like(target, dtype=torch.bool)
    head = OccupancyHead(hidden=8)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.05)
    first = None
    for _ in range(15):
        optimizer.zero_grad()
        volume = head(xyz, opacity, wire, valid, origin, voxel, grid, scale, sh_dc)
        loss = occupancy_bce(
            volume.wire_logits,
            target,
            known,
            pos_weight=8.0,
            focal_gamma=2.0,
        )
        loss.backward()
        optimizer.step()
        if first is None:
            first = float(loss.detach())
    volume = head(xyz, opacity, wire, valid, origin, voxel, grid, scale, sh_dc)
    probability = volume.wire.detach()
    hits = probability[0, 1:7, 1, 1]
    recall = float((hits > 0.5).float().mean())
    assert recall >= 0.8
    assert float(probability[0, 0, 0, 0]) <= 0.5
    assert float(loss.detach()) <= first + 1e-4


def test_wire_direction_smoothness_uses_the_channel_axis():
    logits = torch.zeros(1, 4, 5, 6)
    direction = torch.zeros(1, 3, 4, 5, 6)
    direction[:, 0] = 1.0
    loss = wire_direction_smoothness(logits, direction)
    assert torch.isfinite(loss)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_collision_query_on_a_known_box_and_a_known_cable():
    grid = (4, 4, 4)
    occupancy = torch.full((1, *grid), -8.0)
    wire = torch.full((1, *grid), -8.0)
    occupancy[0, 1, 1, 1] = 8.0
    wire[0, 2, 0, 0] = 8.0
    direction = torch.zeros(1, 3, *grid)
    direction[0, 0] = 1.0
    offset = torch.zeros(1, 3, *grid)
    offset[0, 1, 2, 0, 0] = 0.4
    volume = OccupancyVolume(
        occupancy_logits=occupancy,
        wire_logits=wire,
        wire_direction=direction,
        wire_offset=offset,
        origin=torch.zeros(1, 3),
        voxel_size=torch.tensor(1.0),
    )
    through = torch.tensor([[1.5, 1.5, 1.5]])
    hit = query_collision(volume, through, corridor_radius_m=0.1)
    assert float(hit.risk) > 0.5
    assert float(hit.nearest_distance_m) < 1e-4
    assert float(hit.solid_risk) > 0.5
    clear = torch.tensor([[1.5, 1.5, 4.0]])
    miss = query_collision(volume, clear, corridor_radius_m=0.2)
    assert float(miss.risk) == 0.0
    assert abs(float(miss.nearest_distance_m) - 2.0) < 1e-4
    # Cell (2, 0, 0) is a cable shifted +0.4 m in Y. The cell center is a miss.
    center = torch.tensor([[2.5, 0.5, 0.5]])
    cable = query_collision(volume, center, corridor_radius_m=0.1)
    assert abs(float(cable.nearest_distance_m) - 0.4) < 1e-3
    assert float(cable.risk) == 0.0
    wide = query_collision(volume, center, corridor_radius_m=0.5)
    assert float(wide.wire_risk) > 0.5
    assert float(wide.solid_risk) == 0.0


def test_fuse_occupancy_keeps_a_wire_hit_and_its_point():
    height = width = 16
    intrinsics = np.array([[20.0, 0.0, 8.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]])
    depth = np.zeros((height, width), dtype=np.float64)
    depth[8, 8] = 2.0
    mask = np.zeros((height, width), dtype=bool)
    mask[8, 8] = True
    origin = np.array([-1.0, -1.0, 0.0])
    grid = fuse_occupancy(
        [depth],
        [intrinsics],
        [np.eye(3)],
        [np.zeros(3)],
        origin=origin,
        voxel_size=0.5,
        dims=(4, 4, 8),
        max_depth=8.0,
        wire_masks=[mask],
    )
    # World (0, 0, 2) -> voxel ((0+1)/0.5, (0+1)/0.5, 2/0.5) = (2, 2, 4).
    assert grid.state[2, 2, 4] == OCCUPIED
    assert grid.wire[2, 2, 4] == 1
    assert grid.wire_points.shape[0] >= 1
    assert np.allclose(grid.wire_points[0], [0.0, 0.0, 2.0], atol=1e-5)
    path_grid = OccupancyGrid(
        state=grid.state,
        hit_count=grid.hit_count,
        origin=grid.origin,
        voxel_size=grid.voxel_size,
        wire=grid.wire,
        wire_points=grid.wire_points,
    )
    assert int(path_grid.wire_mask.sum()) == int(grid.wire_mask.sum())


def test_old_occupancy_npz_loads_without_a_wire_channel(tmp_path: Path):
    state = np.zeros((2, 2, 2), dtype=np.uint8)
    state[0, 0, 0] = int(OCCUPIED)
    path = tmp_path / "old.npz"
    np.savez_compressed(
        path,
        state=state,
        hit_count=np.zeros_like(state, dtype=np.uint16),
        origin=np.zeros(3),
        voxel_size=np.float64(0.5),
        timestamp=np.float64(0.0),
        scene_id=np.asarray("scene"),
    )
    loaded = OccupancyGrid.load(path)
    assert loaded.wire.shape == state.shape
    assert int(loaded.wire.sum()) == 0
    assert loaded.wire_points.shape == (0, 3)


def test_onnx_heads_match_pytorch(tmp_path: Path):
    onnxruntime = pytest.importorskip("onnxruntime")
    torch.manual_seed(1)
    relight = RelightOnnx(4, 16, 16, hidden=8)
    features = torch.rand(1, 4, 16, 16)
    sh = torch.randn(1, 16, 3) * 0.1
    sun = torch.tensor([[0.1, 0.2, 0.9]])
    relight_path = tmp_path / "relight.onnx"
    export_onnx(
        relight,
        (features, sh, sun),
        relight_path,
        ["features", "sh", "sun_cam"],
        ["glare"],
    )
    session = onnxruntime.InferenceSession(str(relight_path), providers=["CPUExecutionProvider"])
    (glare_onnx,) = session.run(
        None,
        {"features": features.numpy(), "sh": sh.numpy(), "sun_cam": sun.numpy()},
    )
    with torch.no_grad():
        glare_torch = relight(features, sh, sun).numpy()
    assert glare_onnx.shape == glare_torch.shape
    assert np.allclose(glare_onnx, glare_torch, atol=1e-4, rtol=1e-4)

    grid = (8, 8, 8)
    occupancy = OccupancyOnnx(grid, hidden=8)
    xyz = torch.rand(1, 5, 3) * 2
    opacity = torch.rand(1, 5)
    wire = torch.rand(1, 5)
    valid = torch.ones(1, 5)
    origin = torch.zeros(1, 3)
    voxel = torch.tensor(0.5)
    scale = torch.full((1, 5, 3), 0.1)
    sh_dc = torch.zeros(1, 5, 3)
    occ_path = tmp_path / "occupancy.onnx"
    export_onnx(
        occupancy,
        (xyz, opacity, wire, valid, origin, voxel, scale, sh_dc),
        occ_path,
        ["xyz", "opacity", "wire_prob", "valid", "origin", "voxel_size", "scale", "sh_dc"],
        ["occupancy_logits", "wire_logits", "wire_direction", "wire_offset"],
    )
    session = onnxruntime.InferenceSession(str(occ_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(
        None,
        {
            "xyz": xyz.numpy(),
            "opacity": opacity.numpy(),
            "wire_prob": wire.numpy(),
            "valid": valid.numpy(),
            "origin": origin.numpy(),
            "voxel_size": voxel.numpy(),
            "scale": scale.numpy(),
            "sh_dc": sh_dc.numpy(),
        },
    )
    with torch.no_grad():
        torch_out = occupancy(xyz, opacity, wire, valid, origin, voxel, scale, sh_dc)
    for exported, reference in zip(onnx_out, torch_out, strict=True):
        assert exported.shape == reference.shape
        assert np.allclose(exported, reference.numpy(), atol=1e-4, rtol=1e-4)
