"""Packed Gaussian tensor consumed by the rasterizer and the temporal loss.

Layout of the last dimension, size 62::

    xyz        3   world-frame position, meters, ENU
    scale      3   positive axis scales, meters (already activated)
    quat       4   rotation quaternion, wxyz, unit length
    opacity    1   alpha in [0, 1] (already activated)
    sh         48  degree-3 RGB spherical harmonics, 16 coeffs x 3
    velocity   3   world-frame velocity, meters per second

What ``diff-gaussian-rasterization`` actually multiplies
---------------------------------------------------------
The Inria CUDA kernel does **not** apply ``exp`` or ``sigmoid``. The training
loop in 3D Gaussian Splatting activates parameters in Python and then passes:

* ``means3D`` — world xyz, meters. Our ``xyz`` slice.
* ``scales`` — positive per-axis scales, meters. The kernel builds
  ``R diag(s) diag(s) R^T`` from these values directly. Our packed scales are
  therefore **post-exp**, never log-scales. Passing a log-scale makes the
  covariance the exponent of the number the kernel treats as a length.
* ``rotations`` — quaternion with the real part first, ``(w, x, y, z)``.
  The kernel reads ``q.x`` as ``w``. It normalizes again; we still store a
  unit quaternion, and Phase 3 also flips the sign so ``w >= 0`` (the double
  cover is one rotation). ``pack_gaussians`` normalizes and does not flip
  the sign, so a caller that already canonicalized stays canonical.
* ``opacities`` — alpha in ``[0, 1]``, post-sigmoid, shape ``[N, 1]``.
* ``shs`` — raw coefficients ``[N, 16, 3]``, **not** RGB. The kernel evaluates
  real spherical harmonics and adds ``0.5``. The constant basis is
  ``SH_C0 = 0.28209479177387814``, so a view-independent color satisfies
  ``rgb = SH_C0 * sh[..., 0, :] + 0.5``. Higher bands are not RGB either.

Velocity is packed beside the rasterizer inputs because the dynamics head
predicts it in the same forward pass. The CUDA rasterizer does not read it.
Slice ``packed[..., :GaussianLayout.RASTER_DIM]`` before binding.

``B`` is a batch of timesteps. It is not a flattened video: the temporal loss
compares two packed tensors (t and t+1) at the same slot index. ``N`` is the
static slot budget for one rig resolution (see :class:`GaussianCloud`). A
validity mask and a per-Gaussian wire probability ride **beside** the 62
channels; they are not extra packed channels. The spec's phrase "B is the
batch size (video frame sequence)" is treated as this timestep batch.

Scale bounds live in :func:`activate_scales`. ``exp`` is the activation the
rasterizer expects. On top of that, wire-flagged Gaussians get an anisotropic
upper cap (two thin axes, one long axis). A tiny positive floor keeps the
covariance invertible. That floor is not a minimum size for the scene.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

GAUSSIAN_DIM = 62


class GaussianLayout:
    XYZ = slice(0, 3)
    SCALE = slice(3, 6)
    QUAT = slice(6, 10)
    OPACITY = slice(10, 11)
    SH = slice(11, 59)
    VELOCITY = slice(59, 62)
    DIM = 62
    RASTER_DIM = 59
    SH_DEGREE = 3
    SH_COEFFS = (SH_DEGREE + 1) ** 2

    @staticmethod
    def channel_names() -> tuple[str, ...]:
        names = ["x", "y", "z", "sx", "sy", "sz", "qw", "qx", "qy", "qz", "opacity"]
        for coeff in range(GaussianLayout.SH_COEFFS):
            names.extend(f"sh{coeff}_{channel}" for channel in "rgb")
        names.extend(["vx", "vy", "vz"])
        return tuple(names)


def pack_gaussians(
    xyz: Tensor,
    scale: Tensor,
    quat: Tensor,
    opacity: Tensor,
    sh: Tensor,
    velocity: Tensor,
) -> Tensor:
    """Concatenate physical Gaussian parameters into ``[B, N, 62]``.

    Scales must be positive and opacities must lie in ``[0, 1]``. Quaternions
    are normalized here. Inputs are the activated values a rasterizer consumes,
    not network logits. Use :func:`physical_from_raw` to apply exp / sigmoid.
    """
    xyz = _bn(xyz, 3, "xyz")
    scale = _bn(scale, 3, "scale")
    quat = _bn(quat, 4, "quat")
    opacity = _bn(opacity, 1, "opacity")
    sh = _sh(sh, xyz.shape[0], xyz.shape[1])
    velocity = _bn(velocity, 3, "velocity")
    batch, count = xyz.shape[:2]
    for name, tensor in (
        ("scale", scale),
        ("quat", quat),
        ("opacity", opacity),
        ("sh", sh),
        ("velocity", velocity),
    ):
        if tensor.shape[0] != batch or tensor.shape[1] != count:
            raise ValueError(f"{name} batch/count {tensor.shape[:2]} != xyz {(batch, count)}")
    if not (scale > 0).all():
        raise ValueError("scale must be positive; pack activated scales, not log-scales")
    if not ((opacity >= 0) & (opacity <= 1)).all():
        raise ValueError("opacity must lie in [0, 1]")
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.cat((xyz, scale, quat, opacity, sh, velocity), dim=-1)


def unpack_gaussians(packed: Tensor) -> dict[str, Tensor]:
    """Split ``[B, N, 62]`` into physical parameters. ``sh`` is ``[B, N, 16, 3]``."""
    if packed.ndim != 3 or packed.shape[-1] != GaussianLayout.DIM:
        raise ValueError(f"expected [B, N, {GaussianLayout.DIM}], got {tuple(packed.shape)}")
    sh = packed[..., GaussianLayout.SH].reshape(
        packed.shape[0], packed.shape[1], GaussianLayout.SH_COEFFS, 3
    )
    return {
        "xyz": packed[..., GaussianLayout.XYZ],
        "scale": packed[..., GaussianLayout.SCALE],
        "quat": packed[..., GaussianLayout.QUAT],
        "opacity": packed[..., GaussianLayout.OPACITY],
        "sh": sh,
        "velocity": packed[..., GaussianLayout.VELOCITY],
    }


def rasterizer_view(packed: Tensor) -> Tensor:
    """Drop velocity. The CUDA rasterizer does not consume those three channels."""
    if packed.shape[-1] != GaussianLayout.DIM:
        raise ValueError(f"expected last dim {GaussianLayout.DIM}, got {packed.shape[-1]}")
    return packed[..., : GaussianLayout.RASTER_DIM]


def physical_from_raw(
    xyz: Tensor,
    scale_log: Tensor,
    quat_raw: Tensor,
    opacity_logit: Tensor,
    sh: Tensor,
    velocity: Tensor,
) -> Tensor:
    """Map unbounded head outputs to the packed physical tensor.

    Scale uses exp, opacity uses sigmoid, quaternion is normalized. The
    rasterizer does not do this itself. Phase 3 does not call this helper:
    wire Gaussians need :func:`activate_scales` (anisotropic upper cap plus a
    numerical floor) before :func:`pack_gaussians`.
    """
    return pack_gaussians(
        xyz,
        torch.exp(scale_log),
        quat_raw,
        torch.sigmoid(opacity_logit),
        sh,
        velocity,
    )


# Y_0^0 of the real spherical harmonics. The rasterizer does rgb = SH_C0 * dc + 0.5.
SH_C0 = 0.28209479177387814


def sh_dc_from_rgb(rgb: Tensor) -> Tensor:
    """Degree-0 SH coefficient whose rasterized color is ``rgb``.

    ``rgb`` is in the same units as the training images, typically ``[0, 1]``.
    The rasterizer adds ``0.5`` after the SH dot product, which is why the
    stored DC term is ``(rgb - 0.5) / SH_C0`` and not the RGB value itself.
    """
    return (rgb - 0.5) / SH_C0


def scale_volume(scale: Tensor) -> Tensor:
    """Product ``s_x s_y s_z``. This is the quantity ``L_sparse`` compares to ``tau_max``."""
    if scale.shape[-1] != 3:
        raise ValueError(f"scale must end in 3, got {tuple(scale.shape)}")
    return scale.prod(dim=-1)


def activate_scales(
    log_scale: Tensor,
    wire_prob: Tensor,
    *,
    min_scale: float,
    max_scale: float,
    wire_transverse_max: float,
    wire_long_max: float,
) -> Tensor:
    """Map log-scales to positive meters, then apply the wire cap.

    ``diff-gaussian-rasterization`` consumes the returned meters directly.

    Every Gaussian is floored at ``min_scale`` so a covariance eigenvalue cannot
    hit zero. That floor is a numerical guard (default a tenth of a millimeter),
    not the spec's "lower bound on scale". The spec's lower bound would forbid
    the microscopic splats the wire case needs; Andy's ``L_sparse`` instead
    penalizes volume above ``tau_max``, which is an upper bound on
    ``s_x s_y s_z``.

    The cap implemented here is that upper bound, made anisotropic for
    wire-flagged Gaussians. ``wire_prob`` blends the per-axis ceiling from
    ``max_scale`` (every axis) toward a wire ceiling: the largest predicted
    axis may extend to ``wire_long_max`` (along the wire) and the other two
    are held to ``wire_transverse_max``. At ``wire_prob = 1`` the volume is at
    most ``wire_transverse_max ** 2 * wire_long_max``. The long axis is chosen
    with a straight-through one-hot so the forward cap is hard and the
    log-scale logits still receive gradient.

    ``log_scale`` is ``[B, N, 3]``. ``wire_prob`` is ``[B, N]`` in ``[0, 1]``.
    """
    if log_scale.ndim != 3 or log_scale.shape[-1] != 3:
        raise ValueError(f"log_scale must be [B, N, 3], got {tuple(log_scale.shape)}")
    if wire_prob.shape != log_scale.shape[:2]:
        raise ValueError(
            f"wire_prob shape {tuple(wire_prob.shape)} != {tuple(log_scale.shape[:2])}"
        )
    if min_scale <= 0:
        raise ValueError("min_scale must be positive")
    if wire_transverse_max <= 0 or wire_long_max < wire_transverse_max:
        raise ValueError("need 0 < wire_transverse_max <= wire_long_max")
    if min_scale > wire_transverse_max:
        raise ValueError("min_scale is above the wire transverse cap and would undo it")
    if max_scale < min_scale:
        raise ValueError("max_scale must be >= min_scale")
    raw = torch.exp(log_scale).clamp_min(min_scale)
    soft = torch.softmax(log_scale, dim=-1)
    hard = F.one_hot(log_scale.argmax(dim=-1), num_classes=3).to(dtype=log_scale.dtype)
    long = hard + soft - soft.detach()
    probability = wire_prob.clamp(0, 1).unsqueeze(-1)
    transverse_cap = (1.0 - probability) * max_scale + probability * wire_transverse_max
    long_cap = (1.0 - probability) * max_scale + probability * wire_long_max
    cap = (1.0 - long) * transverse_cap + long * long_cap
    return torch.minimum(raw, cap).clamp_min(min_scale)


def quat_normalize(quat: Tensor) -> Tensor:
    """Unit wxyz quaternion with ``w >= 0``. The sign flip is the same rotation."""
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    negative = quat[..., :1] < 0
    sign = torch.where(negative, -torch.ones_like(quat[..., :1]), torch.ones_like(quat[..., :1]))
    # w == 0 stays put. A negative zero is not a second hemisphere we care about.
    sign = torch.where(quat[..., :1] == 0, torch.ones_like(sign), sign)
    return quat * sign


def quat_mul(lhs: Tensor, rhs: Tensor) -> Tensor:
    """Hamilton product, ``lhs * rhs``, wxyz. Applies ``rhs`` first, then ``lhs``."""
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quat_to_matrix(quat: Tensor) -> Tensor:
    """Rotation matrix whose columns are the images of the local xyz axes."""
    quat = quat_normalize(quat)
    w, x, y, z = quat.unbind(dim=-1)
    matrix = quat.new_empty(quat.shape[:-1] + (3, 3))
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def quaternion_from_x_axis(tangent: Tensor) -> Tensor:
    """Unit wxyz rotation taking local +X onto ``tangent``.

    A zero tangent returns the identity. The opposite of +X is 180 degrees
    about Y, because the cross-product formula vanishes there.
    """
    if tangent.shape[-1] != 3:
        raise ValueError(f"tangent must end in 3, got {tuple(tangent.shape)}")
    norm = tangent.norm(dim=-1, keepdim=True)
    direction = tangent / norm.clamp_min(1e-8)
    identity = tangent.new_zeros(tangent.shape[:-1] + (4,))
    identity[..., 0] = 1
    x_axis = tangent.new_zeros(tangent.shape)
    x_axis[..., 0] = 1
    cross = torch.cross(x_axis, direction, dim=-1)
    dot = (x_axis * direction).sum(dim=-1, keepdim=True)
    quat = torch.cat((1 + dot, cross), dim=-1)
    opposite = dot.squeeze(-1) < -0.999
    fallback = tangent.new_zeros(quat.shape)
    fallback[..., 2] = 1
    quat = torch.where(opposite.unsqueeze(-1), fallback, quat)
    quat = quat_normalize(quat)
    missing = norm.squeeze(-1) < 1e-6
    return torch.where(missing.unsqueeze(-1), identity, quat)


def quat_slerp(start: Tensor, end: Tensor, weight: Tensor) -> Tensor:
    """Interpolate rotations. ``weight`` is ``[... ]`` and broadcasts onto ``start``."""
    if weight.ndim == start.ndim - 1:
        weight = weight.unsqueeze(-1)
    dot = (start * end).sum(dim=-1, keepdim=True)
    end = torch.where(dot < 0, -end, end)
    dot = (start * end).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    near = dot > 0.9995
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega).clamp_min(1e-6)
    start_weight = torch.sin((1.0 - weight) * omega) / sin_omega
    end_weight = torch.sin(weight * omega) / sin_omega
    blended = quat_normalize(start_weight * start + end_weight * end)
    linear = quat_normalize((1.0 - weight) * start + weight * end)
    return torch.where(near, linear, blended)


@dataclass
class GaussianCloud:
    """One timestep of Gaussians plus the side channels the 62-vector omits.

    ``packed`` is ``[B, N, 62]`` of activated parameters. ``N`` is the static
    slot budget: it does not shrink when a wire is absent. ``valid`` marks
    slots the allocator turned on. ``wire_prob`` is the edge probability Phase
    4 multiplies into ``L_sparse`` and Phase 5 can read for the occupancy head.
    Neither mask is a packed channel.

    Slot identity is ``(camera_order[camera_index], row, col, subslot)``.
    Index ``i`` is that same pixel slot at every timestep, which is the
    correspondence ``L_temp`` assumes. See :func:`aerosplat.models.splats.temporal_residual`.
    """

    packed: Tensor
    valid: Tensor
    wire_prob: Tensor
    slot_id: Tensor
    camera_index: Tensor
    row: Tensor
    col: Tensor
    subslot: Tensor
    uv: Tensor
    camera_order: tuple[str, ...]
    anchor_valid: Tensor

    def __post_init__(self) -> None:
        if self.packed.ndim != 3 or self.packed.shape[-1] != GaussianLayout.DIM:
            raise ValueError(
                f"packed must be [B, N, {GaussianLayout.DIM}], got {tuple(self.packed.shape)}"
            )
        batch, count, _ = self.packed.shape
        for name, tensor in (
            ("valid", self.valid),
            ("wire_prob", self.wire_prob),
            ("anchor_valid", self.anchor_valid),
        ):
            if tuple(tensor.shape) != (batch, count):
                raise ValueError(f"{name} must be {(batch, count)}, got {tuple(tensor.shape)}")
        if self.valid.dtype != torch.bool or self.anchor_valid.dtype != torch.bool:
            raise ValueError("valid and anchor_valid must be bool")
        for name, tensor in (
            ("slot_id", self.slot_id),
            ("camera_index", self.camera_index),
            ("row", self.row),
            ("col", self.col),
            ("subslot", self.subslot),
        ):
            if tuple(tensor.shape) != (count,):
                raise ValueError(f"{name} must be [{count}], got {tuple(tensor.shape)}")
        if tuple(self.uv.shape) != (count, 2):
            raise ValueError(f"uv must be [{count}, 2], got {tuple(self.uv.shape)}")
        if not torch.isfinite(self.packed).all():
            raise ValueError("packed Gaussians contain non-finite values")
        if not torch.isfinite(self.wire_prob).all():
            raise ValueError("wire_prob contains non-finite values")
        if len(self.camera_order) == 0 and count != 0:
            raise ValueError("camera_order is empty but N is not")

    @property
    def batch(self) -> int:
        return int(self.packed.shape[0])

    @property
    def count(self) -> int:
        return int(self.packed.shape[1])

    def unpack(self) -> dict[str, Tensor]:
        return unpack_gaussians(self.packed)

    def rasterizer_inputs(self) -> Tensor:
        return rasterizer_view(self.packed)

    def same_layout(self, other: GaussianCloud) -> bool:
        """True when index ``i`` is the same slot in both clouds."""
        if self.camera_order != other.camera_order:
            return False
        if self.slot_id.shape != other.slot_id.shape:
            return False
        return bool(
            torch.equal(self.slot_id, other.slot_id)
            and torch.equal(self.camera_index, other.camera_index)
            and torch.equal(self.row, other.row)
            and torch.equal(self.col, other.col)
            and torch.equal(self.subslot, other.subslot)
        )

    @staticmethod
    def from_activated(
        xyz: Tensor,
        scale: Tensor,
        quat: Tensor,
        opacity: Tensor,
        sh: Tensor,
        velocity: Tensor,
        *,
        valid: Tensor,
        wire_prob: Tensor,
        slot_id: Tensor,
        camera_index: Tensor,
        row: Tensor,
        col: Tensor,
        subslot: Tensor,
        uv: Tensor,
        camera_order: tuple[str, ...],
        anchor_valid: Tensor,
    ) -> GaussianCloud:
        """Pack activated parameters. Scales must already be positive meters."""
        packed = pack_gaussians(xyz, scale, quat, opacity, sh, velocity)
        return GaussianCloud(
            packed=packed,
            valid=valid,
            wire_prob=wire_prob,
            slot_id=slot_id,
            camera_index=camera_index,
            row=row,
            col=col,
            subslot=subslot,
            uv=uv,
            camera_order=tuple(camera_order),
            anchor_valid=anchor_valid,
        )


def _bn(tensor: Tensor, channels: int, name: str) -> Tensor:
    if tensor.ndim != 3 or tensor.shape[-1] != channels:
        raise ValueError(f"{name} must have shape [B, N, {channels}], got {tuple(tensor.shape)}")
    return tensor


def _sh(tensor: Tensor, batch: int, count: int) -> Tensor:
    if tensor.ndim == 4:
        if tensor.shape[-2:] != (GaussianLayout.SH_COEFFS, 3):
            raise ValueError(
                f"sh must be [B, N, {GaussianLayout.SH_COEFFS}, 3], got {tuple(tensor.shape)}"
            )
        flat = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
    elif tensor.ndim == 3 and tensor.shape[-1] == GaussianLayout.SH_COEFFS * 3:
        flat = tensor
    else:
        raise ValueError(f"sh must be [B, N, 16, 3] or [B, N, 48], got {tuple(tensor.shape)}")
    if flat.shape[0] != batch or flat.shape[1] != count:
        raise ValueError(f"sh batch/count {flat.shape[:2]} != {(batch, count)}")
    return flat
