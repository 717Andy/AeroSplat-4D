"""ViT shapes, pose injection, and plane-sweep depth on the fixture scene."""

import numpy as np
import torch

from aerosplat.data.fixture import BOXES, camera_pose
from aerosplat.geometry.camera import camera_center
from aerosplat.geometry.coordinates import intrinsics_from_hfov
from aerosplat.geometry.raycast import render_boxes
from aerosplat.models.backbone import ViTBackbone
from aerosplat.models.factory import ModelConfig, build_lifter
from aerosplat.models.gaussians import pack_gaussians
from aerosplat.models.lift import make_depth_planes, plane_sweep_depth
from aerosplat.models.pose import plucker_at_pixels, transform_w2c
from aerosplat.models.pretrained import load_pretrained_weights


def _cameras(batch, names, height, width, seed=0):
    gen = torch.Generator().manual_seed(seed)
    images = {}
    cameras = {}
    for index, name in enumerate(names):
        images[name] = torch.rand(batch, 3, height, width, generator=gen)
        K = torch.eye(3).unsqueeze(0).repeat(batch, 1, 1)
        K[:, 0, 0] = 40
        K[:, 1, 1] = 40
        K[:, 0, 2] = width / 2
        K[:, 1, 2] = height / 2
        yaw = index * 0.4
        R = torch.eye(3).unsqueeze(0).repeat(batch, 1, 1)
        R[:, 0, 0] = torch.cos(torch.tensor(yaw))
        R[:, 0, 2] = torch.sin(torch.tensor(yaw))
        R[:, 2, 0] = -torch.sin(torch.tensor(yaw))
        R[:, 2, 2] = torch.cos(torch.tensor(yaw))
        t = torch.zeros(batch, 3)
        t[:, 0] = 0.2 * index
        cameras[name] = {"K": K, "R": R, "t": t}
    return images, cameras


def _tiny(**kwargs):
    defaults = dict(
        img_size=32,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        out_indices=(1,),
        cross_view_layers=1,
        pose_encoding="plucker",
        pose_zero_init=False,
        normalize="none",
        gradient_checkpointing=False,
    )
    defaults.update(kwargs)
    return ViTBackbone(**defaults)


def test_multiscale_tokens_follow_the_patch_grid():
    model = _tiny()
    model.eval()
    images, cameras = _cameras(2, ["front", "port"], 32, 32)
    tokens = model(images, cameras)
    assert tokens.names == ["front", "port"]
    assert tokens.patch_size == 8
    for name in tokens.names:
        feature = tokens.features[name]
        assert feature.shape == (2, 32, 4, 4)
        assert set(tokens.multiscale[name]) == {1, 2, 4}
        assert tokens.multiscale[name][2].shape[-2:] == (2, 2)
        assert tokens.multiscale[name][4].shape[-2:] == (1, 1)
        assert torch.isfinite(feature).all()


def test_cameras_may_differ_in_resolution():
    model = _tiny()
    model.eval()
    front, front_cam = _cameras(1, ["front"], 32, 32)
    side, side_cam = _cameras(1, ["side"], 24, 40, seed=1)
    images = {"front": front["front"], "side": side["side"]}
    cameras = {"front": front_cam["front"], "side": side_cam["side"]}
    tokens = model(images, cameras)
    assert tokens.features["front"].shape[-2:] == (4, 4)
    assert tokens.features["side"].shape[-2:] == (3, 5)
    # 3x5 does not divide by 2, so the pyramid stays at the patch grid.
    assert list(tokens.multiscale["side"]) == [1]


def test_permuting_cameras_permutes_tokens():
    model = _tiny()
    model.eval()
    images, cameras = _cameras(1, ["a", "b"], 32, 32)
    forward = model(images, cameras)
    backward = model(
        {"b": images["b"], "a": images["a"]},
        {"b": cameras["b"], "a": cameras["a"]},
    )
    assert torch.allclose(forward.features["a"], backward.features["a"], atol=1e-5)
    assert torch.allclose(forward.features["b"], backward.features["b"], atol=1e-5)


def test_pose_injection_changes_tokens_and_identical_poses_match():
    model = _tiny(pose_encoding="plucker+sinusoidal")
    model.eval()
    images, cameras = _cameras(1, ["front"], 32, 32)
    first = model(images, cameras)
    same = model(images, cameras)
    assert torch.allclose(first.features["front"], same.features["front"])
    moved = {
        name: {key: value.clone() for key, value in pose.items()} for name, pose in cameras.items()
    }
    moved["front"]["t"] = moved["front"]["t"] + torch.tensor([[1.5, -0.4, 0.7]])
    second = model(images, moved)
    assert not torch.allclose(first.features["front"], second.features["front"], atol=1e-4)


def test_plucker_rays_follow_a_world_frame_change():
    K = torch.tensor([[[50.0, 0.0, 8.0], [0.0, 50.0, 6.0], [0.0, 0.0, 1.0]]])
    R = torch.eye(3).unsqueeze(0)
    t = torch.tensor([[0.0, 0.0, -3.0]])
    uv = torch.tensor([[[8.0, 6.0], [10.0, 4.0]]])
    rays = plucker_at_pixels(K, R, t, uv)
    angle = torch.tensor(0.4)
    rotation = torch.tensor(
        [
            [torch.cos(angle), 0.0, torch.sin(angle)],
            [0.0, 1.0, 0.0],
            [-torch.sin(angle), 0.0, torch.cos(angle)],
        ]
    )
    translation = torch.tensor([0.5, -1.0, 0.25])
    R2, t2 = transform_w2c(R, t, rotation, translation)
    rays2 = plucker_at_pixels(K, R2, t2, uv)
    direction = rays[0, :, :3]
    origin = -torch.einsum("bi,bij->bj", t, R)
    new_dir = torch.einsum("ij,nj->ni", rotation, direction)
    new_origin = torch.einsum("ij,nj->ni", rotation, origin.expand_as(direction)) + translation
    new_moment = torch.cross(new_origin, new_dir, dim=-1)
    expected = torch.cat((new_dir, new_moment), dim=-1)
    assert torch.allclose(rays2[0], expected, atol=1e-5)


def test_pretrained_loader_accepts_dust3r_and_dinov2_keys(tmp_path):
    model = _tiny(img_size=16, depth=1, out_indices=(0,))
    encoder = model.encoder
    state = encoder.state_dict()
    dust = {}
    for key, value in state.items():
        renamed = key.replace("blocks.", "enc_blocks.").replace("norm.", "enc_norm.")
        dust[renamed] = value.clone()
    # A decoder tensor must be ignored, and a wrong-shaped tensor must be skipped.
    dust["dec_blocks.0.norm1.weight"] = torch.zeros(4)
    dust["blocks.0.attn.qkv.weight"] = torch.zeros(3, 3)
    path = tmp_path / "aerial_dust3r.pt"
    torch.save({"model": dust}, path)
    sentinel = encoder.blocks[0].norm1.weight.detach().clone()
    encoder.blocks[0].norm1.weight.data.add_(0.5)
    report = load_pretrained_weights(model, path)
    assert report.loaded_count > 0
    assert any("dec_blocks" in item or "no parameter" in item for item in report.skipped)
    assert torch.allclose(encoder.blocks[0].norm1.weight, sentinel)
    # A square pos_embed of a different grid is interpolated rather than dropped.
    other = torch.zeros(1, 1 + 4 * 4, encoder.embed_dim)
    other[:, 0] = 1
    pos_path = tmp_path / "pos.pt"
    torch.save({"pos_embed": other}, pos_path)
    report = load_pretrained_weights(encoder, pos_path)
    assert any("pos_embed" in item for item in report.loaded)
    assert torch.isfinite(encoder.pos_embed).all()


def test_lift_anchors_match_the_gaussian_slot_count():
    cfg = ModelConfig(
        img_size=32,
        patch_size=8,
        embed_dim=32,
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
        lift_stride=8,
        init_values=1.0,
    )
    model = build_lifter(cfg)
    model.eval()
    images, cameras = _cameras(1, ["front", "starboard"], 32, 32)
    with torch.no_grad():
        lifted = model(images, cameras, match="photometric")
    # stride 8 on 32px is a 4x4 grid, two cameras: N = 32 anchors.
    assert lifted.anchors.shape == (1, 32, 3)
    assert lifted.anchor_features.shape[:2] == (1, 32)
    assert lifted.gaussian_slots == 32
    assert lifted.gaussian_dim == 62
    batch, count, _ = lifted.anchors.shape
    packed = pack_gaussians(
        lifted.gaussian_xyz(),
        torch.full((batch, count, 3), 0.1),
        torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4).expand(batch, count, 4),
        torch.full((batch, count, 1), 0.5),
        torch.zeros(batch, count, 16, 3),
        torch.zeros(batch, count, 3),
    )
    assert packed.shape == (1, 32, 62)
    front = lifted.cameras["front"]
    assert front.depth_probs.shape[1] == 4
    assert torch.allclose(front.depth_probs.sum(dim=1), torch.ones_like(front.depth), atol=1e-4)


def _textured_fixture_pair():
    """Fixture scene, front pose, plus a 0.5 m stereo partner, with a world texture.

    The checked-in albedo is flat on each face, so a photometric cost has no
    unique minimum. The texture is a function of the hit point, which is what
    a textured surface would have given the sweep. Geometry, K, and extrinsics
    are the fixture's.
    """
    R, t = camera_pose("front", 0.0)
    height, width = 36, 48
    K_np = intrinsics_from_hfov(width, height, 60.0)
    center = camera_center(R, t)
    right = R.T @ np.array([1.0, 0.0, 0.0])
    t2 = -R @ (center + 0.5 * right)
    _, depth = render_boxes(K_np, R, t, width, height, BOXES)
    _, depth2 = render_boxes(K_np, R, t2, width, height, BOXES)
    rgb, rgb2 = _world_texture(depth, K_np, R, t), _world_texture(depth2, K_np, R, t2)
    return {
        "K": K_np,
        "R": R,
        "t": t,
        "t2": t2,
        "depth": depth,
        "depth2": depth2,
        "rgb": rgb,
        "rgb2": rgb2,
    }


def _world_texture(depth, K, R, t):
    import numpy as np

    height, width = depth.shape
    vs, us = np.mgrid[0:height, 0:width]
    z = depth.astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts = np.stack([(us - cx) * z / fx, (vs - cy) * z / fy, z], axis=-1)
    world = (pts - t) @ R
    # World-locked and smooth enough that the photometric cost has one basin.
    # A faster oscillation aliases across the depth planes and the argmin
    # lands on a false match. Frequency 4 still changes across a pixel at
    # the fixture's ~2 m range.
    color = np.stack(
        [
            0.5 + 0.5 * np.sin(world[..., 0] * 4.0),
            0.5 + 0.5 * np.sin(world[..., 1] * 4.0 + 1.3),
            0.5 + 0.5 * np.sin(world[..., 2] * 4.0 + world[..., 0] * 0.5),
        ],
        axis=-1,
    )
    color[depth <= 0] = 0
    return color.astype(np.float32)


def _mutual_mask(pair):
    import numpy as np

    depth, K, R, t = pair["depth"], pair["K"], pair["R"], pair["t"]
    height, width = depth.shape
    vs, us = np.mgrid[0:height, 0:width]
    z = depth.astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts = np.stack([(us - cx) * z / fx, (vs - cy) * z / fy, z], axis=-1)
    world = (pts - t) @ R
    cam = world @ R.T + pair["t2"]
    zz = cam[..., 2]
    u = K[0, 0] * cam[..., 0] / np.where(zz == 0, np.nan, zz) + K[0, 2]
    v = K[1, 1] * cam[..., 1] / np.where(zz == 0, np.nan, zz) + K[1, 2]
    finite = np.isfinite(u) & np.isfinite(v)
    ui = np.zeros(u.shape, dtype=int)
    vi = np.zeros(v.shape, dtype=int)
    ui[finite] = np.rint(u[finite]).astype(int)
    vi[finite] = np.rint(v[finite]).astype(int)
    inside = (
        finite & (depth > 0.3) & (zz > 0.05) & (ui >= 0) & (vi >= 0) & (ui < width) & (vi < height)
    )
    sampled = pair["depth2"][np.clip(vi, 0, height - 1), np.clip(ui, 0, width - 1)]
    return inside & (np.abs(zz - sampled) < 0.15)


def test_plane_sweep_recovers_fixture_depth_and_is_se3_equivariant():
    import numpy as np

    pair = _textured_fixture_pair()
    images, K, R, t = _torch_pair(pair)
    planes = make_depth_planes(0.4, 12.0, 64, "linear")
    with torch.no_grad():
        swept = plane_sweep_depth(images, K, R, t, planes, temperature=0.02)
    mask = _mutual_mask(pair)
    assert int(mask.sum()) > 100
    mode = swept["depth_mode"][0, 0].numpy()
    expected = swept["depth"][0, 0].numpy()
    gt = pair["depth"]
    mode_err = np.abs(mode - gt)[mask]
    exp_err = np.abs(expected - gt)[mask]
    assert np.median(mode_err) < 0.25
    assert np.median(mode_err / gt[mask]) < 0.12
    assert np.median(exp_err / gt[mask]) < 0.2

    angle = 0.35
    rotation = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ],
        dtype=torch.float32,
    )
    translation = torch.tensor([0.4, -0.2, 1.1])
    R2 = []
    t2 = []
    for rot, trans in zip(R, t, strict=True):
        r_new, t_new = transform_w2c(rot, trans, rotation, translation)
        R2.append(r_new)
        t2.append(t_new)
    with torch.no_grad():
        moved = plane_sweep_depth(images, K, R2, t2, planes, temperature=0.02)
    # The winning plane is a function of relative pose, so it is unchanged on
    # the overlap. Expected depth is a softmax mean and a few low-contrast
    # pixels drift by a plane under float32; the median does not.
    overlap = torch.from_numpy(mask)
    assert torch.equal(moved["depth_mode"][0, 0][overlap], swept["depth_mode"][0, 0][overlap])
    assert (moved["depth"] - swept["depth"]).abs()[0, 0][overlap].median() < 1e-4
    points = swept["points"][0]
    transformed = torch.einsum("ij,jhw->ihw", rotation, points) + translation.view(3, 1, 1)
    assert (moved["points"][0] - transformed).abs()[:, overlap].median() < 1e-3


def _torch_pair(pair):
    def pack(rgb, K, R, t):
        image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        return (
            image,
            torch.from_numpy(K).float().unsqueeze(0),
            torch.from_numpy(R).float().unsqueeze(0),
            torch.from_numpy(t).float().unsqueeze(0),
        )

    ref = pack(pair["rgb"], pair["K"], pair["R"], pair["t"])
    src = pack(pair["rgb2"], pair["K"], pair["R"], pair["t2"])
    images = [ref[0], src[0]]
    return images, [ref[1], src[1]], [ref[2], src[2]], [ref[3], src[3]]
