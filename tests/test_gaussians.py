"""Packed [B, N, 62] Gaussian contract."""

from pathlib import Path

import pytest
import torch

from aerosplat.models.gaussians import (
    GAUSSIAN_DIM,
    GaussianLayout,
    pack_gaussians,
    physical_from_raw,
    rasterizer_view,
    unpack_gaussians,
)
from aerosplat.models.heads import (
    HEAD_OUTPUTS,
    AppearanceHead,
    DynamicsHead,
    GeometryHead,
    OccupancyHead,
    RelightHead,
)
from aerosplat.models.rasterizer import GaussianRasterizer


def _raw(batch=2, count=5):
    xyz = torch.randn(batch, count, 3)
    scale = torch.rand(batch, count, 3) + 0.05
    quat = torch.randn(batch, count, 4)
    opacity = torch.rand(batch, count, 1)
    sh = torch.randn(batch, count, 16, 3)
    velocity = torch.randn(batch, count, 3)
    return xyz, scale, quat, opacity, sh, velocity


def test_layout_adds_up_to_62():
    names = GaussianLayout.channel_names()
    assert len(names) == 62
    assert GAUSSIAN_DIM == 62
    assert GaussianLayout.SH_COEFFS == 16
    assert names[0:3] == ("x", "y", "z")
    assert names[6:10] == ("qw", "qx", "qy", "qz")
    assert names[11] == "sh0_r"
    assert names[13] == "sh0_b"
    assert names[-3:] == ("vx", "vy", "vz")
    packed_width = (
        (GaussianLayout.XYZ.stop - GaussianLayout.XYZ.start)
        + (GaussianLayout.SCALE.stop - GaussianLayout.SCALE.start)
        + (GaussianLayout.QUAT.stop - GaussianLayout.QUAT.start)
        + (GaussianLayout.OPACITY.stop - GaussianLayout.OPACITY.start)
        + (GaussianLayout.SH.stop - GaussianLayout.SH.start)
        + (GaussianLayout.VELOCITY.stop - GaussianLayout.VELOCITY.start)
    )
    assert packed_width == 62
    assert GaussianLayout.RASTER_DIM == 59


def test_pack_unpack_roundtrip_normalizes_quaternion():
    xyz, scale, quat, opacity, sh, velocity = _raw()
    packed = pack_gaussians(xyz, scale, quat, opacity, sh, velocity)
    assert packed.shape == (2, 5, 62)
    parts = unpack_gaussians(packed)
    assert torch.allclose(parts["xyz"], xyz)
    assert torch.allclose(parts["scale"], scale)
    assert torch.allclose(parts["velocity"], velocity)
    assert torch.allclose(parts["sh"], sh)
    assert torch.allclose(parts["quat"].norm(dim=-1), torch.ones(2, 5), atol=1e-5)
    view = rasterizer_view(packed)
    assert view.shape[-1] == 59
    assert torch.equal(view, packed[..., :59])


def test_physical_from_raw_applies_exp_and_sigmoid():
    xyz, _, quat, _, sh, velocity = _raw(1, 2)
    scale_log = torch.zeros(1, 2, 3)
    opacity_logit = torch.zeros(1, 2, 1)
    packed = physical_from_raw(xyz, scale_log, quat, opacity_logit, sh.reshape(1, 2, 48), velocity)
    parts = unpack_gaussians(packed)
    assert torch.allclose(parts["scale"], torch.ones_like(parts["scale"]))
    assert torch.allclose(parts["opacity"], torch.full_like(parts["opacity"], 0.5))


def test_pack_rejects_logits_and_bad_shapes():
    xyz, scale, quat, opacity, sh, velocity = _raw()
    with pytest.raises(ValueError):
        pack_gaussians(xyz, scale * 0, quat, opacity, sh, velocity)
    with pytest.raises(ValueError):
        pack_gaussians(xyz, scale, quat, opacity + 2, sh, velocity)
    with pytest.raises(ValueError):
        pack_gaussians(xyz[0], scale[0], quat[0], opacity[0], sh[0], velocity[0])


def test_phase3_heads_run_phase5_stays_stubbed_and_cuda_stays_optional():
    features = torch.zeros(2, 3, 8)
    geometry = GeometryHead(8)(features)
    assert geometry["offset_raw"].shape == (2, 3, 3)
    assert geometry["log_scale"].shape == (2, 3, 3)
    assert geometry["quat_raw"].shape == (2, 3, 4)
    appearance = AppearanceHead(8)(features)
    assert appearance["opacity_logit"].shape == (2, 3, 1)
    assert appearance["sh_residual"].shape == (2, 3, 16, 3)
    assert DynamicsHead(8)(features)["velocity"].shape == (2, 3, 3)
    for head in (RelightHead(), OccupancyHead()):
        with pytest.raises(NotImplementedError):
            head(features)
    packed = pack_gaussians(*_raw())
    raster = GaussianRasterizer(backend="diff_gaussian")
    camera = {
        "K": torch.eye(3).view(1, 3, 3),
        "R": torch.eye(3).view(1, 3, 3),
        "t": torch.zeros(1, 3),
        "height": 4,
        "width": 4,
    }
    with pytest.raises(RuntimeError):
        raster(packed, cameras={"front": camera})
    assert HEAD_OUTPUTS["geometry"] == ("xyz", "scale", "quat")
    assert HEAD_OUTPUTS["appearance"] == ("opacity", "sh")
    assert HEAD_OUTPUTS["dynamics"] == ("velocity",)
    packed_fields = HEAD_OUTPUTS["geometry"] + HEAD_OUTPUTS["appearance"] + HEAD_OUTPUTS["dynamics"]
    assert "voxel_logits" not in packed_fields
    assert "glare_rgb" not in packed_fields
    source = (Path(__file__).parents[1] / "aerosplat" / "models" / "rasterizer.py").read_text()
    assert "import diff_gaussian_rasterization" not in source
    assert "import gsplat" not in source
    assert "importlib" in source
