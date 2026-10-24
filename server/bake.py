"""Bake a debug cloud from a synchronized sample.

The network is not involved. Depth is unprojected with the capture pose, so
the splats sit on the fixture surfaces. Two extras are added so the workbench
controls have something to show on a recording that has neither:

* a static cable crossing the flight corridor (``wire_prob = 1``)
* a rigid actor with a constant world velocity

Degree-0 SH matches the pixel color (``rgb = SH_C0 * dc + 0.5``). A linear
lobe is added along ``source_sun`` so the sun slider, which is an environment
rotation, moves a visible highlight. Glare is a separate image-space blob
where that sun projects into the camera. It is not part of the SH.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from aerosplat.geometry.camera import project_points, unproject_depth
from aerosplat.models.gaussians import SH_C0, pack_gaussians, quaternion_from_x_axis
from aerosplat.schema import CameraFrame, MultiCameraSample

# Linear SH constants matching aerosplat.models.rasterizer.eval_sh.
_C1 = 0.4886025119029199

WIRE_START = np.array([0.2, -2.0, 3.3], dtype=np.float64)
WIRE_END = np.array([2.0, 2.0, 3.0], dtype=np.float64)
ACTOR_ORIGIN = np.array([1.0, 1.6, 2.4], dtype=np.float64)
ACTOR_VELOCITY = np.array([1.5, 0.0, 0.0], dtype=np.float64)
DEFAULT_SUN = np.array([0.25, 0.12, 0.96], dtype=np.float64)
# logit(0.92) and logit(0.02). 0.92 is a hit; 0.02 is not (threshold is > 0.5).
_OCC_LOGIT = float(np.log(0.92 / 0.08))
_EMPTY_LOGIT = float(np.log(0.02 / 0.98))


@dataclass
class BakedFrame:
    packed: np.ndarray
    valid: np.ndarray
    wire_prob: np.ndarray
    occupancy_logits: np.ndarray
    wire_logits: np.ndarray
    wire_direction: np.ndarray
    wire_offset: np.ndarray
    origin: np.ndarray
    voxel_size: float
    glare: dict[str, np.ndarray]


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("sun vector vanished")
    return np.asarray(vector, dtype=np.float64) / norm


def linear_sun_lobe(sun: np.ndarray, gain: float = 0.45) -> np.ndarray:
    """Degree-1 SH that peaks when the view direction equals ``sun``.

    ``eval_sh`` contributes ``-C1 y c1 + C1 z c2 - C1 x c3``. Solving that
    for ``gain * dot(dir, sun)`` gives the three coefficients. The return is
    ``[16, 3]`` with the lobe copied onto RGB; DC is left at zero for the caller.
    """
    direction = _unit(sun)
    coeff = np.zeros((16, 3), dtype=np.float32)
    # c1 = -gain * sy / C1, c2 = gain * sz / C1, c3 = -gain * sx / C1
    coeff[1] = np.float32(-gain * direction[1] / _C1)
    coeff[2] = np.float32(gain * direction[2] / _C1)
    coeff[3] = np.float32(-gain * direction[0] / _C1)
    return coeff


def _sh_for_colors(colors: np.ndarray, sun: np.ndarray, lobe: float) -> np.ndarray:
    """``colors`` is ``[N, 3]`` in ``[0, 1]``. Return ``[N, 16, 3]``."""
    dc = (colors.astype(np.float32) - 0.5) / np.float32(SH_C0)
    sh = np.zeros((colors.shape[0], 16, 3), dtype=np.float32)
    sh[:, 0, :] = dc
    if lobe != 0.0:
        sh += linear_sun_lobe(sun, gain=lobe)[None, :, :]
    return sh


def _collect_surface(sample: MultiCameraSample, stride: int) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for frame in sample.cameras.values():
        if frame.depth is None:
            continue
        depth = frame.depth.detach().cpu().numpy()
        image = frame.image.detach().cpu().numpy()
        k = frame.K.detach().cpu().numpy()
        rotation = frame.R.detach().cpu().numpy()
        translation = frame.t.detach().cpu().numpy()
        xyz = unproject_depth(depth, k, rotation, translation, stride=stride, max_depth=40.0)
        if xyz.shape[0] == 0:
            continue
        uv, _z = project_points(xyz, k, rotation, translation)
        height, width = depth.shape
        ui = np.clip(np.rint(uv[:, 0]).astype(np.int32), 0, width - 1)
        vi = np.clip(np.rint(uv[:, 1]).astype(np.int32), 0, height - 1)
        color = image[:, vi, ui].T
        points.append(xyz.astype(np.float32))
        colors.append(color.astype(np.float32))
    if not points:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(points, axis=0), np.concatenate(colors, axis=0)


def _dedup(points: np.ndarray, colors: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 0:
        return points, colors
    keys = np.floor(points / cell).astype(np.int32)
    order: dict[tuple[int, int, int], int] = {}
    for index, key in enumerate(keys):
        token = (int(key[0]), int(key[1]), int(key[2]))
        if token not in order:
            order[token] = index
    keep = np.fromiter(order.values(), dtype=np.int64, count=len(order))
    return points[keep], colors[keep]


def _wire_points(spacing: float = 0.08) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = WIRE_END - WIRE_START
    length = float(np.linalg.norm(delta))
    tangent = delta / length
    count = max(int(length / spacing), 2)
    samples = np.linspace(0.0, 1.0, count, dtype=np.float64)
    points = (WIRE_START[None, :] + samples[:, None] * delta[None, :]).astype(np.float32)
    quat = quaternion_from_x_axis(torch.tensor(tangent, dtype=torch.float32).view(1, 3))
    quat_np = quat.detach().cpu().numpy().astype(np.float32).reshape(1, 4)
    quats = np.repeat(quat_np, points.shape[0], axis=0)
    return points, quats, tangent.astype(np.float32)


def _actor_points(timestamp: float) -> np.ndarray:
    center = ACTOR_ORIGIN + ACTOR_VELOCITY * float(timestamp)
    axis = np.linspace(-0.12, 0.12, 5, dtype=np.float32)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    return (center.astype(np.float32) + grid).astype(np.float32)


def _glare_image(frame: CameraFrame, sun: np.ndarray) -> np.ndarray:
    image = frame.image.detach().cpu().numpy()
    _channels, height, width = image.shape
    rotation = frame.R.detach().cpu().numpy()
    intrinsics = frame.K.detach().cpu().numpy()
    direction = rotation @ _unit(sun)
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    if direction[2] <= 0.05:
        return canvas
    u = float(intrinsics[0, 0] * direction[0] / direction[2] + intrinsics[0, 2])
    v = float(intrinsics[1, 1] * direction[1] / direction[2] + intrinsics[1, 2])
    sigma = max(height, width) * 0.07
    yy, xx = np.mgrid[0:height, 0:width]
    blob = np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    warm = np.array([1.0, 0.86, 0.55], dtype=np.float32)
    canvas = blob[:, :, None] * warm[None, None, :] * 0.85
    return np.clip(canvas, 0.0, 1.0)


def glare_png(glare_hwc: np.ndarray) -> bytes:
    import io

    array = np.clip(np.asarray(glare_hwc), 0.0, 1.0)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"glare image must be [H, W, 3], got {array.shape}")
    image = Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def bake_sample(
    sample: MultiCameraSample,
    *,
    source_sun: np.ndarray | None = None,
    stride: int = 1,
    dedup_m: float = 0.045,
    voxel_size: float = 0.25,
    origin: np.ndarray | None = None,
    grid_size: tuple[int, int, int] = (48, 48, 28),
) -> BakedFrame:
    """One timestep of splats, a coarse occupancy volume, and per-camera glare."""
    sun = _unit(DEFAULT_SUN if source_sun is None else source_sun)
    surface, colors = _dedup(*_collect_surface(sample, stride), dedup_m)
    wire_xyz, wire_quat, tangent = _wire_points()
    actor = _actor_points(sample.timestamp)
    chunks_xyz = [surface, wire_xyz, actor]
    xyz = np.concatenate(chunks_xyz, axis=0).astype(np.float32)
    count = xyz.shape[0]
    scale = np.full((count, 3), 0.06, dtype=np.float32)
    quat = np.zeros((count, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    opacity = np.full((count, 1), 0.88, dtype=np.float32)
    velocity = np.zeros((count, 3), dtype=np.float32)
    wire_prob = np.zeros((count,), dtype=np.float32)
    color = np.zeros((count, 3), dtype=np.float32)
    n_surface = surface.shape[0]
    n_wire = wire_xyz.shape[0]
    color[:n_surface] = colors
    scale[:n_surface] = 0.055
    ground = xyz[:n_surface, 2] < 0.15
    scale[:n_surface][ground] = 0.09
    color[n_surface : n_surface + n_wire] = np.array([0.55, 0.62, 0.72], dtype=np.float32)
    scale[n_surface : n_surface + n_wire, 0] = 0.16
    scale[n_surface : n_surface + n_wire, 1] = 0.012
    scale[n_surface : n_surface + n_wire, 2] = 0.012
    quat[n_surface : n_surface + n_wire] = wire_quat
    wire_prob[n_surface : n_surface + n_wire] = 1.0
    color[n_surface + n_wire :] = np.array([0.95, 0.45, 0.12], dtype=np.float32)
    scale[n_surface + n_wire :] = 0.045
    velocity[n_surface + n_wire :] = ACTOR_VELOCITY.astype(np.float32)
    opacity[n_surface + n_wire :] = 0.95
    lobe = np.full((count,), 0.15, dtype=np.float32)
    lobe[n_surface + n_wire :] = 0.55
    sh = _sh_for_colors(color, sun, lobe=0.0)
    # Per-point lobe gain: add the unit-gain lobe scaled per row.
    base_lobe = linear_sun_lobe(sun, gain=1.0)
    sh += base_lobe[None, :, :] * lobe[:, None, None]

    packed = pack_gaussians(
        torch.from_numpy(xyz).unsqueeze(0),
        torch.from_numpy(scale).unsqueeze(0),
        torch.from_numpy(quat).unsqueeze(0),
        torch.from_numpy(opacity).unsqueeze(0),
        torch.from_numpy(sh).unsqueeze(0),
        torch.from_numpy(velocity).unsqueeze(0),
    )
    valid = np.ones((1, count), dtype=bool)
    volume = _voxelize(
        xyz,
        wire_prob,
        tangent,
        voxel_size=voxel_size,
        origin=np.asarray([-2.0, -6.0, -1.0] if origin is None else origin, dtype=np.float32),
        grid_size=grid_size,
    )
    glare = {name: _glare_image(frame, sun) for name, frame in sample.cameras.items()}
    return BakedFrame(
        packed=packed.detach().cpu().numpy().astype(np.float32),
        valid=valid,
        wire_prob=wire_prob.reshape(1, count),
        occupancy_logits=volume[0],
        wire_logits=volume[1],
        wire_direction=volume[2],
        wire_offset=volume[3],
        origin=volume[4],
        voxel_size=float(voxel_size),
        glare=glare,
    )


def _voxelize(
    xyz: np.ndarray,
    wire_prob: np.ndarray,
    tangent: np.ndarray,
    *,
    voxel_size: float,
    origin: np.ndarray,
    grid_size: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dims = tuple(int(v) for v in grid_size)
    occ = np.full(dims, _EMPTY_LOGIT, dtype=np.float32)
    wire = np.full(dims, _EMPTY_LOGIT, dtype=np.float32)
    direction = np.zeros(dims + (3,), dtype=np.float32)
    direction[..., 2] = 1.0
    offset = np.zeros(dims + (3,), dtype=np.float32)
    wire_hits: dict[tuple[int, int, int], list[np.ndarray]] = {}
    scaled = (xyz - origin.reshape(1, 3)) / float(voxel_size)
    ijk = np.floor(scaled).astype(np.int32)
    for index in range(xyz.shape[0]):
        ix, iy, iz = (int(ijk[index, 0]), int(ijk[index, 1]), int(ijk[index, 2]))
        if not (0 <= ix < dims[0] and 0 <= iy < dims[1] and 0 <= iz < dims[2]):
            continue
        if wire_prob[index] >= 0.5:
            wire[ix, iy, iz] = _OCC_LOGIT
            wire_hits.setdefault((ix, iy, iz), []).append(xyz[index])
        else:
            occ[ix, iy, iz] = _OCC_LOGIT
    tangent_u = tangent / max(float(np.linalg.norm(tangent)), 1e-8)
    for (ix, iy, iz), hits in wire_hits.items():
        center = origin + (np.array([ix, iy, iz], dtype=np.float32) + 0.5) * float(voxel_size)
        mean = np.mean(np.stack(hits, axis=0), axis=0)
        shift = np.clip((mean - center) / float(voxel_size), -0.5, 0.5)
        offset[ix, iy, iz] = shift.astype(np.float32)
        direction[ix, iy, iz] = tangent_u.astype(np.float32)
    return occ, wire, direction, offset, origin.astype(np.float32)


def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32)))
