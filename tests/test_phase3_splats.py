"""Phase 3 heads, wire allocation, slot correspondence, and the CPU splatter."""

import torch

from aerosplat.config import load_config
from aerosplat.models.edges import allocate_valid, structure_prior
from aerosplat.models.factory import ModelConfig, build_lifter, build_model, splat_config_from
from aerosplat.models.gaussians import (
    SH_C0,
    GaussianCloud,
    activate_scales,
    pack_gaussians,
    quat_normalize,
    quat_to_matrix,
    quaternion_from_x_axis,
    scale_volume,
    sh_dc_from_rgb,
)
from aerosplat.models.lift import CameraLift, LiftedGeometry, lift_pixel_grid
from aerosplat.models.rasterizer import (
    GaussianRasterizer,
    diff_gaussian_camera_tensors,
    eval_sh,
)
from aerosplat.models.splats import SplatConfig, SplatPredictor, temporal_residual


def test_activate_scales_caps_wire_volume_and_keeps_a_numerical_floor():
    huge = torch.tensor([[[10.0, 12.0, 9.0]]])
    wire = torch.tensor([[1.0]])
    scale = activate_scales(
        huge,
        wire,
        min_scale=1e-4,
        max_scale=2.0,
        wire_transverse_max=0.03,
        wire_long_max=1.0,
    )
    ordered = scale.sort(dim=-1).values
    assert ordered[0, 0, 0] <= 0.03 + 1e-5
    assert ordered[0, 0, 1] <= 0.03 + 1e-5
    assert ordered[0, 0, 2] <= 1.0 + 1e-5
    assert scale_volume(scale)[0, 0] <= 0.03 * 0.03 * 1.0 + 1e-6
    # The long axis is the one the logits asked for, not a shuffled copy.
    assert int(scale[0, 0].argmax()) == 1
    loose = activate_scales(
        huge,
        torch.zeros(1, 1),
        min_scale=1e-4,
        max_scale=2.0,
        wire_transverse_max=0.03,
        wire_long_max=1.0,
    )
    assert torch.allclose(loose, torch.full_like(loose, 2.0))
    floored = activate_scales(
        torch.full((1, 2, 3), -40.0),
        torch.ones(1, 2),
        min_scale=1e-4,
        max_scale=2.0,
        wire_transverse_max=0.03,
        wire_long_max=1.0,
    )
    assert torch.allclose(floored, torch.full_like(floored, 1e-4))


def test_quaternion_maps_local_x_onto_the_tangent_and_is_unit():
    tangent = torch.tensor([[0.0, 2.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    quat = quaternion_from_x_axis(tangent)
    assert torch.allclose(quat.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert bool((quat[:, 0] >= 0).all())
    matrix = quat_to_matrix(quat)
    assert torch.allclose(matrix[0, :, 0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)
    assert torch.allclose(matrix[1, :, 0], torch.tensor([-1.0, 0.0, 0.0]), atol=1e-5)
    assert torch.allclose(matrix[2], torch.eye(3), atol=1e-5)
    flipped = quat_normalize(torch.tensor([[-3.0, 0.4, -0.2, 0.1]]))
    assert flipped[0, 0] > 0
    assert torch.allclose(flipped.norm(), torch.ones(1), atol=1e-5)


def test_allocate_topk_does_not_change_the_tensor_length():
    edge = torch.zeros(2, 6)
    edge[0, 0] = 1.0
    edge[0, 1] = 0.9
    edge[1, 3] = 0.8
    full = allocate_valid(edge, slots_per_cell=3, threshold=0.4, max_extra=None)
    capped = allocate_valid(edge, slots_per_cell=3, threshold=0.4, max_extra=2)
    empty = allocate_valid(torch.zeros(2, 6), slots_per_cell=3, threshold=0.4, max_extra=2)
    assert full.shape == capped.shape == empty.shape == (2, 18)
    # Subslot 0 of every cell stays valid. Shape is the export shape.
    assert int(capped[0, 0::3].sum()) == 6
    assert int(empty[0, 1::3].sum() + empty[0, 2::3].sum()) == 0
    # Cell 0's two extras outrank cell 1 once the budget is 2.
    assert bool(capped[0, 1] and capped[0, 2] and not capped[0, 4])
    assert int(capped[0, 1::3].sum() + capped[0, 2::3].sum()) == 2
    assert int(full[0, 1::3].sum() + full[0, 2::3].sum()) == 4


def test_thin_line_gets_more_gaussians_and_a_thinner_scale():
    image = torch.full((1, 3, 32, 32), 0.15)
    image[:, :, :, 18] = 1.0
    prior, tangent = structure_prior(image)
    # A 1 px line has its Sobel peak on the pixels beside the line, not on it.
    assert float(prior[0, 0, 8:24, 16:21].max()) > 0.8
    assert float(prior[0, 0, :, 1].mean()) < 0.05
    # Vertical line: the tangent beside it runs along +v or -v, not across it.
    direction = tangent[0, :, 16, 17]
    assert abs(float(direction[1])) > 0.8
    assert abs(float(direction[0])) < 0.3

    predictor = SplatPredictor(
        8,
        SplatConfig(
            slots_per_cell=4,
            max_extra_gaussians=None,
            edge_threshold=0.45,
            base_scale_m=1.0,
            offset_radius_m=0.0,
            along_radius_m=0.0,
            wire_transverse_max_m=0.05,
            wire_long_max_m=1.0,
            max_scale_m=2.0,
            min_scale_m=1e-4,
        ),
    )
    predictor.eval()
    lifted = _lifted(image, embed_dim=8, stride=4)
    cameras = _pose(image)
    cloud = predictor(lifted, {"front": image}, cameras)
    blank = predictor(lifted, {"front": torch.full_like(image, 0.15)}, cameras)
    assert cloud.packed.shape == blank.packed.shape == (1, 8 * 8 * 4, 62)
    assert torch.equal(cloud.slot_id, blank.slot_id)
    assert cloud.wire_prob.shape == (1, cloud.count)
    assert cloud.wire_prob is not cloud.packed
    grid = 8
    slots = 4
    wire = cloud.wire_prob[0].view(grid, grid, slots)
    valid = cloud.valid[0].view(grid, grid, slots)
    hot = wire[:, :, 0] > 0.5
    assert int(hot.sum()) >= grid // 2
    assert int(hot.sum(dim=0).gt(0).sum()) <= 3
    assert float(valid[hot][:, 1:].float().mean()) > float(valid[~hot][:, 1:].float().mean())
    assert int(valid[~hot][:, 1:].sum()) == 0
    assert int(blank.valid[0].view(grid, grid, slots)[:, :, 1:].sum()) == 0
    scales = cloud.unpack()["scale"][0].view(grid, grid, slots, 3)
    hot_base = scales[:, :, 0, :][hot]
    ordered = hot_base.sort(dim=-1).values
    # wire_prob is sigmoid(logit(prior)) and sits just under 1, so the blended
    # transverse cap is a hair above wire_transverse_max. It is still microscopic
    # next to the isotropic base scale of 1 m on the blank cells.
    assert float(ordered[:, 1].detach().max()) < 0.1
    assert float(ordered[:, 2].detach().max()) <= 1.0 + 1e-4
    far = scales[:, 0, 0, :]
    assert torch.allclose(far, torch.ones_like(far), atol=1e-4)
    quat = cloud.unpack()["quat"]
    assert torch.allclose(quat.norm(dim=-1), torch.ones(cloud.count), atol=1e-5)
    assert bool((quat[..., 0] >= -1e-6).all())
    rotation = quat_to_matrix(quat[0].view(grid, grid, slots, 4)[:, :, 1][hot])
    axis = rotation[:, :, 0].detach()
    vertical = axis.new_tensor([0.0, 1.0, 0.0])
    alignment = axis.matmul(vertical).abs()
    assert float(alignment.min()) > 0.8


def test_appearance_dc_matches_the_pixel_and_velocity_starts_at_rest():
    color = torch.tensor([0.2, 0.5, 0.9])
    image = color.view(1, 3, 1, 1).expand(1, 3, 16, 16).contiguous()
    predictor = SplatPredictor(4, SplatConfig(slots_per_cell=2, max_extra_gaussians=0))
    predictor.eval()
    cloud = predictor(_lifted(image, embed_dim=4, stride=8), {"front": image}, _pose(image))
    parts = cloud.unpack()
    expected = sh_dc_from_rgb(color)
    dc = parts["sh"][0, :, 0, :]
    assert torch.allclose(dc, expected.expand_as(dc), atol=1e-5)
    assert torch.allclose(parts["sh"][0, :, 1:, :], torch.zeros_like(parts["sh"][0, :, 1:, :]))
    assert torch.allclose(parts["velocity"], torch.zeros_like(parts["velocity"]))
    base = cloud.subslot == 0
    opacity = parts["opacity"][0, base]
    assert torch.allclose(opacity, torch.full_like(opacity, 0.1), atol=1e-5)
    assert torch.equal(parts["opacity"][0, ~base], torch.zeros(int((~base).sum()), 1))
    assert torch.allclose(SH_C0 * expected + 0.5, color, atol=1e-5)
    # Base slots sit on the lift anchors when the offset radius is the default
    # and the last geometry layer is zero. Default radius is not zero, but the
    # zero-init head emits a zero offset, so xyz equals the anchor plus a
    # tangent spread that is zero on subslot 0.
    anchors = _lifted(image, embed_dim=4, stride=8).anchors
    assert torch.allclose(parts["xyz"][0, base], anchors[0], atol=1e-5)


def test_temporal_residual_uses_the_slot_index():
    xyz = torch.zeros(2, 4, 3)
    velocity = torch.zeros(2, 4, 3)
    velocity[..., 0] = 2.0
    current = _cloud(xyz, velocity)
    nxt = _cloud(xyz + velocity * 0.25, torch.zeros_like(velocity))
    out = temporal_residual(current, nxt, dt=0.25)
    assert torch.allclose(out["residual"], torch.zeros_like(xyz), atol=1e-5)
    assert torch.equal(out["weight"], torch.ones(2, 4))
    shifted = _cloud(xyz + velocity * 0.25 + torch.tensor([1.0, 0.0, 0.0]), velocity)
    moved = temporal_residual(current, shifted, dt=torch.tensor([0.25, 0.25]))
    assert torch.allclose(moved["residual"][..., 0], torch.full((2, 4), -1.0))
    current.valid[:, 1] = False
    weighted = temporal_residual(current, nxt, 0.25)
    assert torch.equal(weighted["weight"][:, 1], torch.zeros(2))
    other = _cloud(xyz, velocity, names=("port",))
    try:
        temporal_residual(current, other, 0.1)
    except ValueError as exc:
        assert "slot layout" in str(exc)
    else:
        raise AssertionError("mismatched camera order should not be treated as the same slot")


def test_reference_renderer_paints_a_gaussian_and_composites_in_depth():
    height = width = 21
    cameras = {"front": _batched_pose(1, height, width, fx=20.0)}
    red = _one_gaussian(xyz=(0.0, 0.0, 2.0), rgb=(1.0, 0.0, 0.0), scale=0.15, opacity=0.95)
    renderer = GaussianRasterizer(backend="reference")
    image = renderer(red, cameras)["front"]
    assert image.shape == (1, 3, height, width)
    center = image[0, :, 10, 10]
    corner = image[0, :, 0, 0]
    assert float(center[0]) > 0.5
    assert float(center[1]) < 0.15
    assert float(corner.abs().max()) < 0.05
    green = _one_gaussian(xyz=(0.0, 0.0, 5.0), rgb=(0.0, 1.0, 0.0), scale=0.3, opacity=0.95)
    both = _cat_gaussians(red, green)
    composite = renderer(both, cameras)["front"]
    assert float(composite[0, 0, 10, 10]) > float(composite[0, 1, 10, 10]) + 0.4
    behind = _one_gaussian(xyz=(0.0, 0.0, -2.0), rgb=(1.0, 0.0, 0.0), scale=0.3, opacity=1.0)
    assert float(renderer(behind, cameras)["front"].abs().max()) < 1e-5
    # Degree-0 SH round trip through the same evaluator the renderer uses.
    sh = torch.zeros(1, 1, 16, 3)
    sh[..., 0, :] = sh_dc_from_rgb(torch.tensor([0.25, 0.5, 0.75]))
    dirs = torch.tensor([[[0.2, -0.4, 0.8]]])
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    rgb = eval_sh(sh, dirs, degree=0) + 0.5
    assert torch.allclose(rgb, torch.tensor([[[0.25, 0.5, 0.75]]]), atol=1e-5)


def test_diff_gaussian_view_matrix_is_the_transposed_w2c():
    rotation = torch.eye(3)
    translation = torch.tensor([0.2, -0.4, -5.0])
    intrinsics = torch.tensor([[10.0, 0.0, 8.0], [0.0, 12.0, 6.0], [0.0, 0.0, 1.0]])
    packed = diff_gaussian_camera_tensors(intrinsics, rotation, translation, height=12, width=16)
    w2c = torch.eye(4)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = translation
    assert torch.allclose(packed["viewmatrix"], w2c.T)
    center = -rotation.T @ translation
    assert torch.allclose(packed["campos"], center, atol=1e-5)
    assert abs(packed["tanfovx"] - (8.0 / 10.0)) < 1e-6
    assert abs(packed["tanfovy"] - (6.0 / 12.0)) < 1e-6


def test_tiny_lifter_export_shape_stays_fixed_and_trains():
    model_cfg = ModelConfig(
        img_size=16,
        patch_size=8,
        embed_dim=16,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        out_indices=(1,),
        cross_view_layers=1,
        pose_encoding="plucker",
        pose_zero_init=False,
        normalize="none",
        gradient_checkpointing=False,
        depth_planes=4,
        depth_near=0.5,
        depth_far=8.0,
        depth_spacing="linear",
        lift_stride=8,
    )
    from aerosplat.models.splats import FeedforwardSplatter

    model = FeedforwardSplatter(
        build_lifter(model_cfg),
        SplatPredictor(model_cfg.embed_dim, SplatConfig(slots_per_cell=2, max_extra_gaussians=3)),
    )
    images, cameras = _rig(batch=1, size=16)
    first = model(images, cameras)
    second_images = {name: tensor.roll(1, dims=-1) for name, tensor in images.items()}
    second = model(second_images, cameras)
    # 2 cameras, 2x2 lift cells, 2 slots.
    assert first.packed.shape == second.packed.shape == (1, 16, 62)
    assert first.same_layout(second)
    assert int(first.valid[:, first.subslot > 0].sum()) <= 3
    assert int(second.valid[:, second.subslot > 0].sum()) <= 3
    loss = first.packed.sum() + first.wire_prob.sum()
    loss.backward()
    assert model.predictor.edge.net[-1].weight.grad is not None
    parts = first.unpack()
    assert torch.allclose(parts["quat"].norm(dim=-1), torch.ones(1, 16), atol=1e-4)


def test_laptop_configs_keep_the_8_to_16_gb_budget():
    small = load_config(["model=vit_small"])
    base = load_config(["model=vit_base"])
    tiny = load_config(["model=vit_tiny"])
    assert small.model.memory.min_vram_gb == 8
    assert small.model.memory.max_vram_gb == 16
    assert base.model.memory.min_vram_gb == 8
    assert base.model.memory.max_vram_gb == 16
    assert int(small.model.splats.slots_per_cell) == 4
    assert int(base.model.splats.max_extra_gaussians) == 1024
    assert "8" in str(small.model.splats.note)
    assert "16" in str(base.model.splats.note)
    splat = splat_config_from(tiny)
    assert splat.slots_per_cell == 4
    assert splat.max_extra_gaussians == 256
    assert splat.min_scale_m < splat.wire_transverse_max_m
    built = build_model(tiny)
    assert built.predictor.config.slots_per_cell == 4


def _lifted(image, embed_dim, stride, depth=4.0):
    batch, _, height, width = image.shape
    grid_h, grid_w = height // stride, width // stride
    uv, _, _ = lift_pixel_grid(height, width, stride, image.device, image.dtype)
    points = torch.zeros(batch, 3, grid_h, grid_w)
    points[:, 0] = (uv[..., 0] - width / 2) * depth / 16.0
    points[:, 1] = (uv[..., 1] - height / 2) * depth / 16.0
    points[:, 2] = depth
    features = torch.zeros(batch, embed_dim, grid_h, grid_w)
    zeros = torch.zeros(batch, 1, grid_h, grid_w)
    view = CameraLift(
        name="front",
        features=features,
        depth_logits=zeros,
        depth_probs=zeros,
        depth=zeros + depth,
        depth_mode=zeros + depth,
        points=points,
        confidence=torch.ones_like(zeros),
        valid=torch.ones(batch, grid_h, grid_w, dtype=torch.bool),
        uv=uv,
    )
    flat = points.permute(0, 2, 3, 1).reshape(batch, grid_h * grid_w, 3)
    return LiftedGeometry(
        cameras={"front": view},
        camera_names=["front"],
        anchors=flat,
        anchor_features=torch.zeros(batch, grid_h * grid_w, embed_dim),
        anchor_valid=torch.ones(batch, grid_h * grid_w, dtype=torch.bool),
        camera_index=torch.zeros(batch, grid_h * grid_w, dtype=torch.long),
        planes=torch.linspace(0.5, 8.0, 4),
    )


def _pose(image, fx=16.0):
    _, _, height, width = image.shape
    intrinsics = torch.eye(3).view(1, 3, 3)
    intrinsics[0, 0, 0] = fx
    intrinsics[0, 1, 1] = fx
    intrinsics[0, 0, 2] = width / 2
    intrinsics[0, 1, 2] = height / 2
    return {
        "front": {
            "K": intrinsics,
            "R": torch.eye(3).view(1, 3, 3),
            "t": torch.zeros(1, 3),
        }
    }


def _batched_pose(batch, height, width, fx):
    intrinsics = torch.eye(3).view(1, 3, 3).repeat(batch, 1, 1)
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fx
    intrinsics[:, 0, 2] = (width - 1) / 2
    intrinsics[:, 1, 2] = (height - 1) / 2
    return {
        "K": intrinsics,
        "R": torch.eye(3).view(1, 3, 3).repeat(batch, 1, 1),
        "t": torch.zeros(batch, 3),
        "height": height,
        "width": width,
    }


def _cloud(xyz, velocity, names=("front",), valid=None):
    batch, count, _ = xyz.shape
    if valid is None:
        valid = torch.ones(batch, count, dtype=torch.bool)
    quat = torch.zeros(batch, count, 4)
    quat[..., 0] = 1
    return GaussianCloud.from_activated(
        xyz,
        torch.full((batch, count, 3), 0.1),
        quat,
        torch.full((batch, count, 1), 0.4),
        torch.zeros(batch, count, 16, 3),
        velocity,
        valid=valid,
        wire_prob=torch.zeros(batch, count),
        slot_id=torch.arange(count),
        camera_index=torch.zeros(count, dtype=torch.long),
        row=torch.zeros(count, dtype=torch.long),
        col=torch.arange(count),
        subslot=torch.zeros(count, dtype=torch.long),
        uv=torch.zeros(count, 2),
        camera_order=tuple(names),
        anchor_valid=torch.ones(batch, count, dtype=torch.bool),
    )


def _one_gaussian(xyz, rgb, scale, opacity):
    sh = torch.zeros(1, 1, 16, 3)
    sh[0, 0, 0] = sh_dc_from_rgb(torch.tensor(rgb))
    quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    means = torch.tensor(xyz, dtype=torch.float32).view(1, 1, 3)
    return pack_gaussians(
        means,
        torch.full((1, 1, 3), scale),
        quat,
        torch.tensor([[[opacity]]]),
        sh,
        torch.zeros(1, 1, 3),
    )


def _cat_gaussians(first, second):
    return torch.cat((first, second), dim=1)


def _rig(batch, size):
    images = {}
    cameras = {}
    for index, name in enumerate(("front", "port")):
        images[name] = torch.rand(batch, 3, size, size)
        intrinsics = torch.eye(3).view(1, 3, 3).repeat(batch, 1, 1)
        intrinsics[:, 0, 0] = 12
        intrinsics[:, 1, 1] = 12
        intrinsics[:, 0, 2] = size / 2
        intrinsics[:, 1, 2] = size / 2
        rotation = torch.eye(3).view(1, 3, 3).repeat(batch, 1, 1)
        rotation[:, 0, 0] = 0.0 if index else 1.0
        rotation[:, 0, 2] = 1.0 if index else 0.0
        rotation[:, 2, 0] = -1.0 if index else 0.0
        rotation[:, 2, 2] = 0.0 if index else 1.0
        translation = torch.zeros(batch, 3)
        translation[:, 0] = 0.4 * index
        cameras[name] = {"K": intrinsics, "R": rotation, "t": translation}
    return images, cameras
