"""Image-space glare, and an approximate SH sun slider.

Lens flare is light added by the lens, not radiance from the scene. Degree-3
spherical harmonics stay on the Gaussians and are trained against the clean
render. This head predicts a non-negative RGB layer in each camera:

    composited = clamp(clean_render + glare, 0, 1)

The photometric loss compares ``composited`` to the observed frame. A
downstream consumer that wants the scene without the sun in the lens uses
``clean_render``. The addition clips at 1 because a real sensor saturates.
The separation loss, when a paired flare is known, supervises the pre-clip
layer.

The head reads three things, per camera:

* a feature map (the lift features in the full model, or any image-space map)
* the camera-frame sun direction
* the degree-3 SH field, pooled over the Gaussians that belong to that camera
  and also evaluated in the sun direction

The last layer is biased so a fresh head emits a near-zero veil
(``sigmoid(-4) ≈ 0.018``) and still has gradient. An exact zero would be a
ReLU dead zone, and ``softplus(0)`` would be a bright gray veil.

``evaluate_relit_sh`` is the utility the frontend sun slider can call later.
It rotates the view direction by the rotation that carries the old sun onto
the new one, then evaluates the same coefficients. That swings the existing
view-dependent lobe. It does not recompute shadows, interreflections, or the
glare layer. Degree 3 cannot represent a hard solar disk. ``rotate_sh_coeffs``
is the same map expressed as new coefficients, for a shader that only knows
how to evaluate SH.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from aerosplat.models.gaussians import GaussianCloud, GaussianLayout
from aerosplat.models.rasterizer import eval_sh


def composite_glare(clean: Tensor, glare: Tensor) -> Tensor:
    """Additive flare, clipped to the sensor range ``[0, 1]``."""
    if clean.shape != glare.shape:
        raise ValueError(f"clean {tuple(clean.shape)} != glare {tuple(glare.shape)}")
    return (clean + glare).clamp(0, 1)


def pool_sh(sh: Tensor, weight: Tensor | None = None) -> Tensor:
    """Weighted mean of ``sh`` over the Gaussian axis.

    ``sh`` is ``[B, 16, 3]`` (already pooled) or ``[B, N, 16, 3]``. ``weight``
    is ``[B, N]``. A row whose weights sum to zero returns zeros.
    """
    if sh.ndim == 3 and sh.shape[-2:] == (GaussianLayout.SH_COEFFS, 3):
        return sh
    if sh.ndim != 4 or sh.shape[-2:] != (GaussianLayout.SH_COEFFS, 3):
        raise ValueError(f"sh must be [B, 16, 3] or [B, N, 16, 3], got {tuple(sh.shape)}")
    if weight is None:
        return sh.mean(dim=1)
    w = weight.to(dtype=sh.dtype).clamp_min(0)
    total = w.sum(dim=1).clamp_min(1e-8)
    acc = (sh * w[:, :, None, None]).sum(dim=1)
    empty = w.sum(dim=1) <= 0
    acc = torch.where(empty[:, None, None], torch.zeros_like(acc), acc / total[:, None, None])
    return acc


def _normalize(vector: Tensor) -> Tensor:
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-8)


class RelightHead(nn.Module):
    """Per-camera additive glare from features, sun direction, and degree-3 SH."""

    def __init__(self, feature_dim: int, hidden: int = 32) -> None:
        super().__init__()
        if feature_dim < 1 or hidden < 1:
            raise ValueError("feature_dim and hidden must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden = int(hidden)
        self.features = nn.Sequential(
            nn.Conv2d(feature_dim, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        # sun (3) + SH evaluated at the sun (3) + pooled coefficients (48).
        self.condition = nn.Linear(3 + 3 + GaussianLayout.SH_COEFFS * 3, hidden)
        self.output = nn.Conv2d(hidden, 3, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, -4.0)

    def forward(
        self,
        features: Tensor,
        sh: Tensor,
        sun_cam: Tensor,
        out_hw: tuple[int, int],
        weight: Tensor | None = None,
    ) -> Tensor:
        """``features`` ``[B, C, h, w]``, ``sun_cam`` ``[B, 3]``, glare ``[B, 3, H, W]``."""
        if features.ndim != 4 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must be [B, {self.feature_dim}, h, w], got {tuple(features.shape)}"
            )
        if sun_cam.shape != (features.shape[0], 3):
            raise ValueError(
                f"sun_cam must be [{features.shape[0]}, 3], got {tuple(sun_cam.shape)}"
            )
        if len(out_hw) != 2 or out_hw[0] < 1 or out_hw[1] < 1:
            raise ValueError(f"out_hw must be positive (H, W), got {out_hw}")
        pooled = pool_sh(sh, weight)
        if pooled.shape[0] != features.shape[0]:
            raise ValueError("sh batch does not match features")
        sun = _normalize(sun_cam.to(dtype=features.dtype))
        sun_color = eval_sh(pooled, sun)
        conditioning = torch.cat(
            (sun, sun_color, pooled.reshape(pooled.shape[0], -1)),
            dim=-1,
        )
        spatial = self.features(features)
        bias = self.condition(conditioning).unsqueeze(-1).unsqueeze(-1)
        raw = self.output(spatial + bias)
        glare = torch.sigmoid(raw)
        if glare.shape[-2:] != (int(out_hw[0]), int(out_hw[1])):
            glare = F.interpolate(
                glare,
                size=(int(out_hw[0]), int(out_hw[1])),
                mode="bilinear",
                align_corners=False,
            )
        return glare


def glare_for_cameras(
    head: RelightHead,
    feature_maps: dict[str, Tensor],
    cloud: GaussianCloud,
    cameras: dict[str, dict[str, Tensor]],
    sun_world: Tensor,
    image_hw: dict[str, tuple[int, int]],
) -> dict[str, Tensor]:
    """One glare layer per camera. ``sun_world`` is ``[B, 3]`` in the flight frame."""
    parts = cloud.unpack()
    sh = parts["sh"]
    opacity = parts["opacity"].squeeze(-1)
    weight = cloud.valid.to(dtype=sh.dtype) * opacity
    if sun_world.ndim != 2 or sun_world.shape[-1] != 3:
        raise ValueError(f"sun_world must be [B, 3], got {tuple(sun_world.shape)}")
    order = list(cloud.camera_order)
    glare: dict[str, Tensor] = {}
    for name in feature_maps:
        if name not in cameras:
            raise KeyError(f"camera {name!r} has features but no pose")
        if name not in order:
            raise KeyError(f"camera {name!r} is not in the Gaussian camera order {order}")
        index = order.index(name)
        rotation = cameras[name]["R"]
        sun_cam = torch.matmul(rotation, sun_world.unsqueeze(-1)).squeeze(-1)
        owned = cloud.camera_index == index
        camera_weight = weight * owned.to(dtype=weight.dtype).unsqueeze(0)
        height, width = image_hw[name]
        glare[name] = head(
            feature_maps[name],
            sh,
            sun_cam,
            (height, width),
            weight=camera_weight,
        )
    return glare


def rotation_between(source: Tensor, target: Tensor) -> Tensor:
    """Rotation matrices taking unit vectors ``source`` onto ``target``.

    Both end in 3. The return ends in ``[3, 3]``. Opposite vectors rotate 180
    degrees about an arbitrary orthogonal axis. A zero vector raises.
    """
    if source.shape[-1] != 3 or target.shape[-1] != 3:
        raise ValueError("rotation_between expects vectors ending in 3")
    src = _normalize(source)
    dst = _normalize(target)
    cross = torch.cross(src, dst, dim=-1)
    dot = (src * dst).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    sin = cross.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis = cross / sin
    # Rodrigues. Near-parallel inputs use the identity (or 180 degrees).
    angle = torch.acos(dot)
    eye = torch.eye(3, device=source.device, dtype=source.dtype).expand(src.shape[:-1] + (3, 3))
    k = _skew(axis)
    parallel = (dot.squeeze(-1) > 0.9995).unsqueeze(-1).unsqueeze(-1)
    opposite = (dot.squeeze(-1) < -0.9995).unsqueeze(-1).unsqueeze(-1)
    sin_term = torch.sin(angle).unsqueeze(-1) * k
    cos_term = (1.0 - torch.cos(angle)).unsqueeze(-1) * (k @ k)
    rot = eye + sin_term + cos_term
    ortho = _orthogonal_axis(src)
    half_turn = 2.0 * ortho.unsqueeze(-1) * ortho.unsqueeze(-2) - eye
    rot = torch.where(parallel, eye, rot)
    rot = torch.where(opposite, half_turn, rot)
    return rot


def _skew(axis: Tensor) -> Tensor:
    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    row0 = torch.stack((zero, -z, y), dim=-1)
    row1 = torch.stack((z, zero, -x), dim=-1)
    row2 = torch.stack((-y, x, zero), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _orthogonal_axis(vector: Tensor) -> Tensor:
    other = torch.zeros_like(vector)
    other[..., 1] = 1.0
    # If the vector is nearly +Y, use +X instead so the cross product exists.
    use_x = vector[..., 1].abs() > 0.9
    other = torch.where(use_x.unsqueeze(-1), torch.zeros_like(vector), other)
    other_x = torch.zeros_like(vector)
    other_x[..., 0] = 1.0
    other = torch.where(use_x.unsqueeze(-1), other_x, other)
    return _normalize(torch.cross(vector, other, dim=-1))


def evaluate_relit_sh(
    sh: Tensor,
    view_dir: Tensor,
    source_sun: Tensor,
    target_sun: Tensor,
) -> Tensor:
    """SH color under a rotated sun, plus the rasterizer's ``+0.5``.

    ``sh`` is ``[..., 16, 3]``. ``view_dir`` is the direction the color is
    evaluated along (Gaussian to camera, or any unit direction). ``source_sun``
    and ``target_sun`` are the old and new sun directions in the same frame.
    The lobe that used to sit on ``source_sun`` sits on ``target_sun``. This
    is an environment rotation, not a lighting solve.
    """
    rotation = rotation_between(source_sun, target_sun)
    # f_new(d) = f(R^T d), with R mapping the old sun onto the new one.
    # Row-vector form of R^T d is d @ R.
    while rotation.ndim < view_dir.ndim + 1:
        rotation = rotation.unsqueeze(0)
    rotated = torch.matmul(view_dir.unsqueeze(-2), rotation).squeeze(-2)
    return eval_sh(sh, _normalize(rotated)) + 0.5


def rotate_sh_coeffs(
    sh: Tensor,
    source_sun: Tensor,
    target_sun: Tensor,
    samples: int = 48,
) -> Tensor:
    """Coefficients whose evaluation matches :func:`evaluate_relit_sh` without ``+0.5``.

    Degree 3 is 16-dimensional. The linear map is fit by least squares on a
    Fibonacci sphere so a shader that only evaluates SH can upload the result.
    ``source_sun`` and ``target_sun`` are a single pair of directions (the
    slider), not a per-Gaussian sun. ``samples`` must be at least 16. The fit
    is not part of training.
    """
    if samples < GaussianLayout.SH_COEFFS:
        raise ValueError(f"need at least {GaussianLayout.SH_COEFFS} samples, got {samples}")
    if sh.shape[-2:] != (GaussianLayout.SH_COEFFS, 3):
        raise ValueError(f"sh must end in [16, 3], got {tuple(sh.shape)}")
    directions = _fibonacci(samples, sh.device, sh.dtype)
    columns = []
    for index in range(GaussianLayout.SH_COEFFS):
        coeff = torch.zeros(samples, GaussianLayout.SH_COEFFS, 3, device=sh.device, dtype=sh.dtype)
        coeff[:, index, 0] = 1.0
        columns.append(eval_sh(coeff, directions)[:, 0])
    design = torch.stack(columns, dim=-1)
    solver = torch.linalg.pinv(design)
    rotation = rotation_between(source_sun.reshape(3), target_sun.reshape(3))
    rotated = directions @ rotation
    flat = sh.reshape(-1, GaussianLayout.SH_COEFFS, 3)
    expanded = flat[:, None, :, :].expand(-1, samples, -1, -1).contiguous()
    dirs = rotated.unsqueeze(0).expand(flat.shape[0], -1, -1)
    values = eval_sh(expanded, dirs)
    fitted = torch.einsum("is,bsc->bic", solver, values)
    return fitted.reshape(sh.shape)


def _fibonacci(count: int, device, dtype) -> Tensor:
    golden = (1.0 + 5.0**0.5) / 2.0
    index = torch.arange(count, device=device, dtype=dtype)
    z = 1.0 - 2.0 * (index + 0.5) / count
    radius = torch.sqrt((1.0 - z * z).clamp_min(0))
    theta = 2.0 * math.pi * index / golden
    return torch.stack((radius * torch.cos(theta), radius * torch.sin(theta), z), dim=-1)
