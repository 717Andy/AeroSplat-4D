"""Depth-supervised pretraining of the lift on static multi-view samples.

Phase 4 owns the photometric Gaussian loss. This loop is the stage-1 geometry
objective from the dataset deep-dive: multi-view depth and point maps. The
target is planar Z in meters. Invalid depths (0, the project-wide "no return")
are masked out. A binned cross-entropy keeps the distribution peaked on the
plane nearest the ground-truth depth; L1 trains the expected depth.

The default entry point runs a couple of steps on the tiny fixture on CPU.
It is a smoke test of the graph, not a trained model.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch import Tensor

from aerosplat.data.transforms import resize_sample
from aerosplat.models.factory import build_lifter
from aerosplat.models.lift import LiftedGeometry
from aerosplat.schema import CameraFrame, MultiCameraSample


def prepare_rig(sample: MultiCameraSample, size: int, patch_size: int) -> MultiCameraSample:
    """Resize every camera to ``size`` and pad so the ViT patch grid divides it.

    ``K`` is scaled with the resize and left unchanged by the pad (the pad is
    on the bottom and right). Padded depth is 0, so the loss ignores it.
    """
    if size < 1:
        raise ValueError("size must be positive")
    resized = resize_sample(sample, {name: (size, size) for name in sample.cameras})
    pad_h = (patch_size - size % patch_size) % patch_size
    pad_w = (patch_size - size % patch_size) % patch_size
    if pad_h == 0 and pad_w == 0:
        return resized
    cameras = {}
    for name, frame in resized.cameras.items():
        image = F.pad(frame.image, (0, pad_w, 0, pad_h))
        depth = None if frame.depth is None else F.pad(frame.depth, (0, pad_w, 0, pad_h))
        cameras[name] = CameraFrame(
            name=name,
            image=image,
            K=frame.K,
            R=frame.R,
            t=frame.t,
            timestamp=frame.timestamp,
            depth=depth,
            R_query=frame.R_query,
            t_query=frame.t_query,
            query_timestamp=frame.query_timestamp,
            sync=frame.sync,
            frame_index=frame.frame_index,
        )
    return MultiCameraSample(
        scene_id=resized.scene_id,
        sample_index=resized.sample_index,
        timestamp=resized.timestamp,
        cameras=cameras,
    )


def batch_from_sample(
    sample: MultiCameraSample,
) -> tuple[dict[str, Tensor], dict[str, dict[str, Tensor]]]:
    """Add the batch dimension the lifter expects. One sample, batch size 1."""
    images: dict[str, Tensor] = {}
    cameras: dict[str, dict[str, Tensor]] = {}
    for name, frame in sample.cameras.items():
        images[name] = frame.image.unsqueeze(0)
        cameras[name] = {
            "K": frame.K.unsqueeze(0),
            "R": frame.R.unsqueeze(0),
            "t": frame.t.unsqueeze(0),
        }
    return images, cameras


def depth_supervision_loss(lifted: LiftedGeometry, sample: MultiCameraSample) -> Tensor:
    """L1 on expected depth plus cross-entropy on the nearest depth plane."""
    terms: list[Tensor] = []
    planes = lifted.planes
    for name, camera in lifted.cameras.items():
        frame = sample.cameras[name]
        if frame.depth is None:
            continue
        uv = camera.uv
        width = frame.depth.shape[-1]
        height = frame.depth.shape[-2]
        u = uv[..., 0].round().long().clamp(0, width - 1)
        v = uv[..., 1].round().long().clamp(0, height - 1)
        target = frame.depth[v, u].to(device=camera.depth.device, dtype=camera.depth.dtype)
        mask = target > 0
        if camera.valid.any():
            mask = mask & camera.valid[0]
        if int(mask.sum()) == 0:
            continue
        pred = camera.depth[0, 0]
        l1 = (pred - target).abs()[mask].mean()
        logits = camera.depth_logits[0]  # [D, h, w]
        bins = (target.unsqueeze(0) - planes.view(-1, 1, 1)).abs().argmin(dim=0)
        ce = F.cross_entropy(logits.unsqueeze(0), bins.unsqueeze(0), reduction="none")[0]
        ce = ce[mask].mean()
        terms.append(l1 + 0.1 * ce)
    if not terms:
        raise RuntimeError(
            "depth loss had no valid pixels. The sample needs planar depth, and at "
            "least one lift cell has to fall on a return."
        )
    return torch.stack(terms).mean()


def run_depth_pretrain(
    model: torch.nn.Module,
    samples: list[MultiCameraSample],
    *,
    steps: int = 2,
    lr: float = 1e-3,
    device: str = "cpu",
    match: str = "hybrid",
) -> list[float]:
    """Run ``steps`` AdamW updates. Returns the loss after each step."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not samples:
        raise ValueError("pretraining needs at least one sample")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    for step in range(steps):
        sample = samples[step % len(samples)]
        images, cameras = batch_from_sample(sample)
        images = {name: image.to(device) for name, image in images.items()}
        cameras = {
            name: {key: value.to(device) for key, value in pose.items()}
            for name, pose in cameras.items()
        }
        optimizer.zero_grad(set_to_none=True)
        lifted = model(images, cameras, match=match)
        loss = depth_supervision_loss(lifted, sample)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite depth loss at step {step}: {float(loss)}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Depth-supervised pretraining smoke loop.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides. The smoke defaults are model=vit_tiny and a CPU fixture run.",
    )
    args = parser.parse_args(argv)
    from aerosplat.config import load_config
    from aerosplat.data.dataset import build_dataset

    # vit_small is the training recipe. The script itself defaults to the tiny
    # CPU model so a checkout can execute the loop without a GPU. A later
    # override such as model=vit_small replaces this one.
    cfg = load_config(["model=vit_tiny", *args.overrides])
    dataset = build_dataset(cfg)
    model = build_lifter(cfg)
    size = int(cfg.model.img_size)
    patch = int(cfg.model.patch_size)
    sample = prepare_rig(dataset[0], size, patch)
    steps = int(cfg.pretrain.steps)
    losses = run_depth_pretrain(
        model,
        [sample],
        steps=steps,
        lr=float(cfg.pretrain.lr),
        device="cpu",
    )
    print("depth pretrain losses:", " ".join(f"{loss:.4f}" for loss in losses))


if __name__ == "__main__":
    main()
