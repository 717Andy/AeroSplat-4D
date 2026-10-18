"""Phase 4 losses. Andy's equations, with the decisions recorded in the docs.

.. code-block:: text

    L_total = λ1 L_photo + λ2 L_ssim + λ3 L_temp + λ4 L_occ + λ5 L_sparse

    L_photo  = (1 / N_cam) Σ_c  mean_{pixels} |Î_c - I_c|
    L_ssim   = 1 - (1 / N_cam) Σ_c SSIM(Î_c, I_c)
    L_temp   = mean_{surviving splats} || (x + v Δt) - x_{t+1, reprojected} ||_2^2
    L_occ    = BCE over known voxels, mean by default (Andy's sum is the other reduction)
    L_sparse = (1 / N) Σ_i  α_i * max(0, s_x s_y s_z - τ_max)

``L_photo`` is the mean absolute error of every scalar in the image, so a
fully wrong ``[0, 1]`` image scores 1 regardless of resolution. Cameras are
then averaged, not pooled by pixel count: a 48² downward camera weighs the
same as a 224² nose camera.

``L_sparse`` is Andy's ungated sum (``wire_prob`` does not multiply it) divided
by the static slot count ``N``. The division is the one change from the
written sum, so ``λ5`` does not grow when the rig gets more slots. The Phase 3
anisotropic wire cap is a separate hard constraint and stays on.

``L_occ`` defaults to a mean over known voxels (unknown masked). ``pos_weight``
and a focal gamma are optional. ``λ4`` defaults to 0 until the Phase 5 head
replaces the placeholder. ``L_velocity`` is an extra term, not one of Andy's
five: squared error against a logged actor velocity, weight 0 unless a sim
export provides it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from aerosplat.geometry.occupancy import FREE, OCCUPIED, UNKNOWN
from aerosplat.models.gaussians import GaussianCloud, scale_volume, unpack_gaussians
from aerosplat.models.temporal import reprojected_temporal_loss


@dataclass
class LossWeights:
    """Coefficients in front of Andy's five terms, plus optional velocity GT.

    ``photo=0.8`` and ``ssim=0.2`` are the decided photometric mix. The others
    are starting points: ``τ`` and ``λ3``..``λ5`` were unset in the equations.
    ``occ`` and ``velocity`` stay at 0 until their supervision is turned on.
    """

    photo: float = 0.8
    ssim: float = 0.2
    temp: float = 1.0
    occ: float = 0.0
    sparse: float = 0.01
    velocity: float = 0.0


@dataclass
class LossOutput:
    """Unweighted terms plus the weighted sum. Logging should record each field."""

    total: Tensor
    photo: Tensor
    ssim: Tensor
    temp: Tensor
    occ: Tensor
    sparse: Tensor
    velocity: Tensor

    def scalars(self) -> dict[str, float]:
        names = ("total", "photo", "ssim", "temp", "occ", "sparse", "velocity")
        return {name: float(getattr(self, name).detach().cpu()) for name in names}


def photometric_l1(rendered: dict[str, Tensor], targets: dict[str, Tensor]) -> Tensor:
    """Mean over cameras of the per-scalar L1. Empty renders contribute nothing."""
    terms = []
    for name, prediction in rendered.items():
        if name not in targets:
            raise KeyError(f"rendered camera {name!r} has no ground-truth image")
        target = targets[name]
        if prediction.shape != target.shape:
            raise ValueError(
                f"camera {name}: rendered {tuple(prediction.shape)} != target {tuple(target.shape)}"
            )
        terms.append((prediction - target).abs().mean())
    if not terms:
        raise ValueError("photometric loss needs at least one camera")
    return torch.stack(terms).mean()


def ssim(prediction: Tensor, target: Tensor, window_size: int = 11, sigma: float = 1.5) -> Tensor:
    """Mean SSIM over ``[B, C, H, W]`` (or ``[C, H, W]``). Images are in ``[0, 1]``.

    The window is a separable Gaussian. Reflect padding keeps a constant image
    constant, so two flat fields match the closed form
    ``(2 μx μy + C1) / (μx² + μy² + C1)`` with ``C1 = 0.01²`` and ``C2 = 0.03²``.
    A window that does not fit the image falls back to ``1 - L1``.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"SSIM shapes differ: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
    if prediction.ndim != 4:
        raise ValueError(f"SSIM expected [B, C, H, W], got {tuple(prediction.shape)}")
    channels = int(prediction.shape[1])
    height, width = int(prediction.shape[-2]), int(prediction.shape[-1])
    window = min(int(window_size), height, width)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return 1 - (prediction - target).abs().mean()
    coords = torch.arange(window, device=prediction.device, dtype=prediction.dtype)
    coords = coords - window // 2
    gauss = torch.exp(-(coords * coords) / (2 * sigma * sigma))
    gauss = gauss / gauss.sum()
    kernel = (gauss[:, None] * gauss[None, :]).expand(channels, 1, window, window).contiguous()
    pad = window // 2
    pred_p = F.pad(prediction, (pad, pad, pad, pad), mode="reflect")
    tgt_p = F.pad(target, (pad, pad, pad, pad), mode="reflect")
    mu_x = F.conv2d(pred_p, kernel, groups=channels)
    mu_y = F.conv2d(tgt_p, kernel, groups=channels)
    sigma_x = (F.conv2d(pred_p * pred_p, kernel, groups=channels) - mu_x * mu_x).clamp_min(0)
    sigma_y = (F.conv2d(tgt_p * tgt_p, kernel, groups=channels) - mu_y * mu_y).clamp_min(0)
    sigma_xy = F.conv2d(pred_p * tgt_p, kernel, groups=channels) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator).mean()


def ssim_loss(
    rendered: dict[str, Tensor],
    targets: dict[str, Tensor],
    window_size: int = 11,
) -> Tensor:
    """``1 - mean_c SSIM``. Identical images score 0."""
    scores = []
    for name, prediction in rendered.items():
        if name not in targets:
            raise KeyError(f"rendered camera {name!r} has no ground-truth image")
        scores.append(ssim(prediction, targets[name], window_size=window_size))
    if not scores:
        raise ValueError("SSIM loss needs at least one camera")
    return 1 - torch.stack(scores).mean()


def sparsity_loss(packed: Tensor, tau_max: float) -> Tensor:
    """Andy's volume hinge, divided by ``N``.

    ``(1 / N) Σ_i α_i max(0, s_x s_y s_z - τ_max)``, then averaged over the
    batch. ``wire_prob`` is not a gate. ``N`` is the static slot count, so an
    invalid slot with opacity 0 adds nothing to the sum and still sits in the
    denominator. That is what keeps ``λ5`` stable when the allocator turns
    extras on. The Phase 3 cap is not applied here; pass the packed scales.
    """
    if tau_max < 0:
        raise ValueError(f"tau_max must be >= 0, got {tau_max}")
    parts = unpack_gaussians(packed)
    volume = scale_volume(parts["scale"])
    penalty = parts["opacity"].squeeze(-1) * (volume - float(tau_max)).clamp_min(0)
    return penalty.mean()


def occupancy_bce(
    logits: Tensor,
    target: Tensor,
    known: Tensor,
    *,
    pos_weight: float = 1.0,
    focal_gamma: float = 0.0,
    reduction: str = "mean",
) -> Tensor:
    """BCE on voxels the cache has actually observed.

    ``target`` is 1 on occupied and 0 on free. ``known`` is false on unknown
    cells, which are dropped rather than treated as empty. ``reduction="mean"``
    divides by the number of known voxels so the term does not grow with the
    grid; ``"sum"`` is Andy's written sum. ``pos_weight`` multiplies only the
    positive term. ``focal_gamma > 0`` multiplies by ``(1 - p_t)^γ``.
    """
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"reduction must be mean or sum, got {reduction!r}")
    if pos_weight < 0 or focal_gamma < 0:
        raise ValueError("pos_weight and focal_gamma must be >= 0")
    target = target.to(dtype=logits.dtype)
    if target.shape != logits.shape or known.shape != logits.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)}, target {tuple(target.shape)}, "
            f"known {tuple(known.shape)} must match"
        )
    log_positive = F.logsigmoid(logits)
    log_negative = F.logsigmoid(-logits)
    loss = -(target * float(pos_weight) * log_positive + (1.0 - target) * log_negative)
    if focal_gamma > 0:
        probability = torch.sigmoid(logits)
        probability_t = torch.where(target > 0.5, probability, 1.0 - probability)
        loss = loss * (1.0 - probability_t).clamp(0, 1).pow(float(focal_gamma))
    selected = loss[known.to(dtype=torch.bool)]
    if selected.numel() == 0:
        return logits.sum() * 0.0
    if reduction == "sum":
        return selected.sum()
    return selected.mean()


def occupancy_target_from_grid(grid) -> tuple[Tensor, Tensor]:
    """``(target, known)`` from a Phase 1 :class:`OccupancyGrid`.

    Unknown (0) is masked. Free (1) is the negative class. Occupied (2) is
    the positive class. ``hit_count`` is not a weight; it is not a wire label.
    """
    state = torch.as_tensor(grid.state)
    if int(UNKNOWN) != 0:
        raise RuntimeError("occupancy UNKNOWN is no longer 0; update the loss mask")
    known = state != int(UNKNOWN)
    target = (state == int(OCCUPIED)).to(dtype=torch.float32)
    free_or_occupied = (state == int(FREE)) | (state == int(OCCUPIED)) | (state == int(UNKNOWN))
    if not bool(free_or_occupied.all()):
        raise ValueError("occupancy grid has a state other than unknown/free/occupied")
    return target, known


def velocity_supervision(
    cloud: GaussianCloud,
    gt_velocity: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Mean squared error of world-frame velocity against a logged target.

    ``gt_velocity`` is ``[B, N, 3]``, ``[B, 3]`` (one rigid twist per batch
    element, broadcast onto every slot), or ``[3]``. ``mask`` further restricts
    the mean; slots with ``valid`` false always drop out. This is the optional
    term for Cosys/Colosseum actor poses, not part of Andy's five.
    """
    predicted = cloud.unpack()["velocity"]
    target = gt_velocity.to(device=predicted.device, dtype=predicted.dtype)
    if target.shape == (3,):
        target = target.view(1, 1, 3).expand_as(predicted)
    elif target.shape == (predicted.shape[0], 3):
        target = target[:, None, :].expand_as(predicted)
    elif target.shape != predicted.shape:
        raise ValueError(
            f"gt_velocity {tuple(gt_velocity.shape)} cannot broadcast onto {tuple(predicted.shape)}"
        )
    error = (predicted - target).pow(2).sum(dim=-1)
    keep = cloud.valid if mask is None else cloud.valid & mask.to(dtype=torch.bool)
    if not bool(keep.any()):
        return predicted.sum() * 0.0
    return error[keep].mean()


def composite_loss(
    rendered: dict[str, Tensor] | None,
    targets: dict[str, Tensor] | None,
    *,
    cloud: GaussianCloud | None = None,
    cloud_tp1: GaussianCloud | None = None,
    cameras_tp1: dict[str, dict[str, Tensor]] | None = None,
    dt: float | Tensor | None = None,
    occ_logits: Tensor | None = None,
    occ_target: Tensor | None = None,
    occ_known: Tensor | None = None,
    gt_velocity: Tensor | None = None,
    gt_velocity_mask: Tensor | None = None,
    weights: LossWeights | None = None,
    tau_max: float = 0.008,
    ssim_window: int = 11,
    occ_pos_weight: float = 1.0,
    occ_focal_gamma: float = 0.0,
    occ_reduction: str = "mean",
    temp_depth_abs: float = 0.5,
    temp_depth_rel: float = 0.1,
    temp_max_search: float = 1.0,
) -> LossOutput:
    """Weighted sum. A missing input makes that term zero and drops it from the graph.

    ``tau_max`` defaults to ``0.2³ = 0.008`` m³, the volume of the Phase 3 base
    scale at initialization, so the hinge is closed on a fresh model and opens
    when an opaque Gaussian grows past that cube.
    """
    weights = weights if weights is not None else LossWeights()
    zero_source = _zero_source(rendered, targets, cloud, occ_logits)
    photo = photometric_l1(rendered, targets) if rendered is not None else zero_source
    structural = (
        ssim_loss(rendered, targets, window_size=ssim_window)
        if rendered is not None
        else zero_source
    )
    if cloud is not None and cloud_tp1 is not None and cameras_tp1 is not None and dt is not None:
        temporal = reprojected_temporal_loss(
            cloud,
            cloud_tp1,
            cameras_tp1,
            dt,
            depth_abs=temp_depth_abs,
            depth_rel=temp_depth_rel,
            max_search=temp_max_search,
        )["loss"]
    else:
        temporal = zero_source
    if occ_logits is not None and occ_target is not None and occ_known is not None:
        occupied = occupancy_bce(
            occ_logits,
            occ_target,
            occ_known,
            pos_weight=occ_pos_weight,
            focal_gamma=occ_focal_gamma,
            reduction=occ_reduction,
        )
    else:
        occupied = zero_source
    sparse = sparsity_loss(cloud.packed, tau_max) if cloud is not None else zero_source
    if cloud is not None and gt_velocity is not None:
        velocity = velocity_supervision(cloud, gt_velocity, gt_velocity_mask)
    else:
        velocity = zero_source
    total = (
        float(weights.photo) * photo
        + float(weights.ssim) * structural
        + float(weights.temp) * temporal
        + float(weights.occ) * occupied
        + float(weights.sparse) * sparse
        + float(weights.velocity) * velocity
    )
    return LossOutput(
        total=total,
        photo=photo,
        ssim=structural,
        temp=temporal,
        occ=occupied,
        sparse=sparse,
        velocity=velocity,
    )


def _zero_source(
    rendered: dict[str, Tensor] | None,
    targets: dict[str, Tensor] | None,
    cloud: GaussianCloud | None,
    occ_logits: Tensor | None,
) -> Tensor:
    if cloud is not None:
        return cloud.packed.sum() * 0.0
    if rendered:
        tensor = next(iter(rendered.values()))
        return tensor.sum() * 0.0
    if targets:
        tensor = next(iter(targets.values()))
        return tensor.sum() * 0.0
    if occ_logits is not None:
        return occ_logits.sum() * 0.0
    return torch.zeros(())
