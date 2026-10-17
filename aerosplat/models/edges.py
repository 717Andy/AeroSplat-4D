"""Thin-structure prior and the fixed-budget Gaussian allocator.

The ViT runs at 224 px on the laptop, and a crane cable is often one pixel
wide at the sensor. The edge branch therefore looks at the image itself
(the network resolution, or a same-FOV higher-resolution copy), not only at
the 16x16 patch grid. A Sobel / structure-tensor prior fires on a line before
any training. A small conv predicts a residual logit on top of that prior and
is zero-initialized, so at initialization the prior is the allocator.

Each lift cell receives the max response inside its pixel bin. A wire that
misses the cell center still allocates. The tensor length is
``anchors * slots_per_cell`` whether or not the wire is present. Thresholding
and top-k only clear the ``valid`` mask.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _luma(image: Tensor) -> Tensor:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"image must be [B, 3, H, W], got {tuple(image.shape)}")
    red, green, blue = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    return 0.2989 * red + 0.5870 * green + 0.1140 * blue


def _unit_channels(vector: Tensor) -> Tensor:
    return vector / vector.norm(dim=1, keepdim=True).clamp_min(1e-8)


def structure_prior(image: Tensor) -> tuple[Tensor, Tensor]:
    """Sobel magnitude times structure-tensor coherence, and the edge tangent.

    Returns ``prior`` ``[B, 1, H, W]`` in ``[0, 1]`` and ``tangent``
    ``[B, 2, H, W]`` in pixel ``(du, dv)``. A flat image yields a zero prior
    and a zero tangent. The prior is peak-normalized per image so a thin line
    on a blank background lands near 1 without a learned gain.
    """
    luma = _luma(image)
    # Replicate, not zero, padding. Zero padding paints a fake frame around a
    # constant image and the peak-normalized prior then treats the border as a wire.
    padded = F.pad(luma, (1, 1, 1, 1), mode="replicate")
    kernel_x = luma.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    kernel_y = luma.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    kernel_x = kernel_x.view(1, 1, 3, 3)
    kernel_y = kernel_y.view(1, 1, 3, 3)
    grad_x = F.conv2d(padded, kernel_x)
    grad_y = F.conv2d(padded, kernel_y)
    txx = F.avg_pool2d(grad_x * grad_x, kernel_size=3, stride=1, padding=1)
    tyy = F.avg_pool2d(grad_y * grad_y, kernel_size=3, stride=1, padding=1)
    txy = F.avg_pool2d(grad_x * grad_y, kernel_size=3, stride=1, padding=1)
    diff = txx - tyy
    root = torch.sqrt((0.5 * diff) ** 2 + txy * txy + 1e-12)
    trace = txx + tyy
    lambda_1 = 0.5 * trace + root
    lambda_2 = 0.5 * trace - root
    coherence = (lambda_1 - lambda_2) / (lambda_1 + lambda_2 + 1e-8)
    magnitude = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-12)
    response = magnitude * coherence.clamp(0, 1)
    peak = response.amax(dim=(-2, -1), keepdim=True)
    flat = peak < 1e-4
    prior = response / peak.clamp_min(1e-6)
    prior = torch.where(flat, torch.zeros_like(prior), prior).clamp(0, 1)
    theta = 0.5 * torch.atan2(2 * txy, diff)
    tangent = torch.cat((-torch.sin(theta), torch.cos(theta)), dim=1)
    tangent = torch.where(flat, torch.zeros_like(tangent), _unit_channels(tangent))
    return prior, tangent


def reduce_to_grid(
    prob: Tensor, tangent: Tensor, grid_h: int, grid_w: int
) -> tuple[Tensor, Tensor]:
    """Max-pool a full-resolution edge map onto the lift grid.

    The tangent is taken from the pixel that won the max, so the wire's
    direction is the direction at the wire rather than a bilinear smear
    toward the cell center. Non-divisible sizes fall back to adaptive pooling.
    """
    if grid_h < 1 or grid_w < 1:
        raise ValueError(f"grid must be at least 1x1, got {grid_h}x{grid_w}")
    _, _, height, width = prob.shape
    if height % grid_h == 0 and width % grid_w == 0:
        kernel_h = height // grid_h
        kernel_w = width // grid_w
        pooled, index = F.max_pool2d(
            prob,
            kernel_size=(kernel_h, kernel_w),
            stride=(kernel_h, kernel_w),
            return_indices=True,
        )
        flat = tangent.reshape(tangent.shape[0], 2, height * width)
        gather_index = index.reshape(index.shape[0], 1, grid_h * grid_w).expand(-1, 2, -1)
        gathered = flat.gather(2, gather_index).reshape(tangent.shape[0], 2, grid_h, grid_w)
        return pooled, _unit_channels(gathered)
    pooled = F.adaptive_max_pool2d(prob, (grid_h, grid_w))
    gathered = F.interpolate(tangent, size=(grid_h, grid_w), mode="bilinear", align_corners=True)
    return pooled, _unit_channels(gathered)


class EdgeBranch(nn.Module):
    """Learned residual on the structure-tensor prior.

    The last convolution is zero-initialized. At initialization
    ``sigmoid(learned + logit(prior))`` equals the prior, which is what the
    thin-line test relies on. Gradients reach the convolution through the
    soft wire probability (scale cap and tangent alignment), not through the
    hard top-k mask.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        prior, tangent = structure_prior(image)
        learned = self.net(image)
        clamped = prior.clamp(1e-4, 1.0 - 1e-4)
        prior_logit = torch.log(clamped) - torch.log(1.0 - clamped)
        probability = torch.sigmoid(learned + prior_logit.detach())
        return probability, tangent

    def on_grid(self, image: Tensor, grid_h: int, grid_w: int) -> tuple[Tensor, Tensor]:
        probability, tangent = self.forward(image)
        return reduce_to_grid(probability, tangent, grid_h, grid_w)


def allocate_valid(
    edge_prob: Tensor,
    slots_per_cell: int,
    threshold: float,
    max_extra: int | None,
) -> Tensor:
    """Bool mask ``[B, N_cells * slots_per_cell]``, cell-major, subslot last.

    Subslot 0 is the base Gaussian and is always valid. Extra subslots are
    valid when the cell's edge probability is at least ``threshold`` and the
    slot is inside the global top-k of extra candidates. Top-k never changes
    the tensor length. Earlier extra subslots get a small score bonus so a
    tie breaks toward the first extra rather than an arbitrary index.
    """
    if edge_prob.ndim != 2:
        raise ValueError(f"edge_prob must be [B, N], got {tuple(edge_prob.shape)}")
    if slots_per_cell < 1:
        raise ValueError("slots_per_cell must be >= 1")
    batch, cells = edge_prob.shape
    slots = slots_per_cell
    extras = slots - 1
    valid = torch.ones(batch, cells, slots, dtype=torch.bool, device=edge_prob.device)
    if extras == 0:
        return valid.reshape(batch, cells * slots)
    # A 0.1% taper so equal edge scores prefer the earlier extra subslot.
    # It is too small to reorder two cells whose edge probabilities differ.
    decay = torch.linspace(1.0, 0.999, extras, device=edge_prob.device, dtype=edge_prob.dtype)
    scores = edge_prob.unsqueeze(-1) * decay
    above = edge_prob >= threshold
    keep = above.unsqueeze(-1).expand(batch, cells, extras).contiguous()
    if max_extra is not None:
        if max_extra < 0:
            raise ValueError("max_extra_gaussians must be >= 0")
        flat_scores = scores.reshape(batch, cells * extras)
        flat_keep = keep.reshape(batch, cells * extras)
        budget = min(int(max_extra), flat_scores.shape[1])
        chosen = torch.zeros_like(flat_keep)
        if budget > 0:
            masked_scores = flat_scores.masked_fill(~flat_keep, -1.0)
            index = masked_scores.topk(budget, dim=1).indices
            chosen.scatter_(1, index, True)
            chosen = chosen & flat_keep & (masked_scores > -0.5)
        keep = chosen.view(batch, cells, extras)
    valid[:, :, 1:] = keep
    return valid.reshape(batch, cells * slots)
