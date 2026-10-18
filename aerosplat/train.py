"""Phase 4 training loop.

One step renders the packed ``[B, N, 62]`` cloud (velocity stripped inside the
rasterizer) and applies Andy's composite loss. The ViT runs under autocast
when CUDA AMP is on; the rasterizer stays fp32. Gradient checkpointing is the
flag on the shared encoder. Pairs of consecutive samples supply ``L_temp``.

Laptop and DDP recipes live in ``configs/experiment/``. CPU CI calls
:func:`run_training` with ``model=vit_tiny`` and the reference splatter.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from aerosplat.data.collate import collate_samples
from aerosplat.data.dataset import build_dataset
from aerosplat.models.factory import build_model
from aerosplat.models.heads import PlaceholderOccupancyHead
from aerosplat.models.losses import (
    LossOutput,
    LossWeights,
    composite_loss,
    occupancy_target_from_grid,
)
from aerosplat.models.pretrain import prepare_rig
from aerosplat.models.rasterizer import GaussianRasterizer
from aerosplat.schema import MultiCameraSample


class SplatTrainer(nn.Module):
    """Splatter plus the occupancy placeholder. Phase 5 swaps the placeholder."""

    def __init__(self, splatter: nn.Module, occ_head: PlaceholderOccupancyHead) -> None:
        super().__init__()
        self.splatter = splatter
        self.occ_head = occ_head

    def forward(self, images: dict[str, Tensor], cameras: dict[str, dict[str, Tensor]]):
        return self.splatter(images, cameras, match="hybrid")


class FramePairDataset(Dataset):
    """Consecutive samples from one scene, resized to the ViT grid.

    Index ``i`` yields ``(sample_t, sample_t+1 or None, raw_t)``. ``raw_t`` is
    the unresized sample so the occupancy cache can reuse a grid built by
    ``scripts/prepare_cache.py``. The network sees the resized copy.
    """

    def __init__(
        self,
        base,
        pairs: list[tuple[int, int | None]],
        img_size: int,
        patch_size: int,
        camera_names: list[str] | None,
    ) -> None:
        self.base = base
        self.pairs = pairs
        self.img_size = img_size
        self.patch_size = patch_size
        self.camera_names = camera_names

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        current, nxt = self.pairs[index]
        raw = _select_cameras(self.base[current], self.camera_names)
        prepared = prepare_rig(raw, self.img_size, self.patch_size)
        prepared_next = None
        if nxt is not None:
            raw_next = _select_cameras(self.base[nxt], self.camera_names)
            prepared_next = prepare_rig(raw_next, self.img_size, self.patch_size)
        return prepared, prepared_next, raw


def pair_indices(dataset) -> list[tuple[int, int | None]]:
    """``(i, i+1)`` when the next row is the same scene at a later timestamp."""
    count = len(dataset)
    meta = _timeline(dataset)
    pairs: list[tuple[int, int | None]] = []
    for index in range(count):
        nxt = None
        if index + 1 < count and meta is not None:
            scene, timestamp = meta[index]
            next_scene, next_time = meta[index + 1]
            if next_scene == scene and next_time > timestamp + 1e-9:
                nxt = index + 1
        elif index + 1 < count and meta is None:
            nxt = index + 1
        pairs.append((index, nxt))
    return pairs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    seed: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    module = model.module if isinstance(model, DistributedDataParallel) else model
    torch.save(
        {
            "step": step,
            "seed": seed,
            "model": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> int:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    module = model.module if isinstance(model, DistributedDataParallel) else model
    module.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler"):
        scaler.load_state_dict(payload["scaler"])
    return int(payload["step"])


class TrainLogger:
    """JSONL always. TensorBoard when the package imports. W&B when asked."""

    def __init__(self, directory: Path, enable_wandb: bool, project: str, config: dict, rank: int):
        self.rank = rank
        self.directory = Path(directory)
        self.writer = None
        self.wandb = None
        self._file = None
        if rank != 0:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        self._file = open(self.directory / "metrics.jsonl", "a", encoding="utf-8")
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(self.directory / "tb"))
        except ImportError:
            self.writer = None
        if enable_wandb:
            import wandb

            wandb.init(project=project, config=config, dir=str(self.directory))
            self.wandb = wandb

    def scalars(self, step: int, values: dict[str, float]) -> None:
        if self.rank != 0 or self._file is None:
            return
        row = {"step": step, **values}
        self._file.write(json.dumps(row) + "\n")
        self._file.flush()
        if self.writer is not None:
            for key, value in values.items():
                self.writer.add_scalar(key, value, step)
        if self.wandb is not None:
            self.wandb.log(values, step=step)

    def images(self, step: int, grid: Tensor) -> None:
        if self.rank != 0:
            return
        image = grid.detach().clamp(0, 1).cpu()
        if self.writer is not None:
            self.writer.add_image("rendered_vs_gt", image, step)
        array = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        from PIL import Image

        Image.fromarray(array).save(self.directory / f"render_{step:06d}.png")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        if self.writer is not None:
            self.writer.close()
        if self.wandb is not None:
            self.wandb.finish()


def comparison_grid(rendered: dict[str, Tensor], targets: dict[str, Tensor]) -> Tensor:
    """Ground truth on the left, render on the right, cameras stacked vertically."""
    rows: list[Tensor] = []
    for name, prediction in rendered.items():
        pred = prediction[0].detach().float().clamp(0, 1).cpu()
        target = targets[name][0].detach().float().clamp(0, 1).cpu()
        rows.append(torch.cat((target, pred), dim=-1))
    width = max(row.shape[-1] for row in rows)
    padded = []
    for row in rows:
        if row.shape[-1] == width:
            padded.append(row)
        else:
            padded.append(F.pad(row, (0, width - row.shape[-1])))
    return torch.cat(padded, dim=-2)


def run_training(cfg, dataset=None) -> list[dict[str, float]]:
    """Train from a composed config. Returns the rank-0 scalar rows."""
    world, rank, local_rank = _world()
    distributed = world > 1
    if bool(cfg.train.distributed) and not distributed:
        raise RuntimeError(
            "train.distributed is set but this process is not a torchrun worker. "
            "Launch with: torchrun --nproc_per_node=N scripts/train.py "
            "--config-name experiment/ddp"
        )
    seed = int(cfg.seed) if "seed" in cfg else int(cfg.train.seed)
    set_seed(seed)
    device = _device(str(cfg.train.device), local_rank)
    if distributed:
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
        if device.type == "cuda":
            torch.cuda.set_device(device)

    if dataset is None:
        dataset = build_dataset(cfg)
    camera_names = _camera_names(cfg.train.camera_names)
    pairs = pair_indices(dataset)
    pair_data = FramePairDataset(
        dataset,
        pairs,
        img_size=int(cfg.model.img_size),
        patch_size=int(cfg.model.patch_size),
        camera_names=camera_names,
    )
    batch_size = max(1, int(cfg.train.batch_size))
    if batch_size > len(pair_data):
        batch_size = len(pair_data)
    sampler: DistributedSampler | RandomSampler
    if distributed:
        sampler = DistributedSampler(
            pair_data,
            num_replicas=world,
            rank=rank,
            shuffle=True,
            seed=seed,
        )
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = RandomSampler(pair_data, generator=generator)
    num_workers = int(cfg.train.num_workers)
    loader = DataLoader(
        pair_data,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_collate_pairs,
        drop_last=False,
    )

    splatter = build_model(cfg)
    trainer = SplatTrainer(splatter, PlaceholderOccupancyHead())
    _set_checkpointing(trainer, bool(cfg.train.gradient_checkpointing))
    trainer.to(device)
    if distributed:
        trainer = DistributedDataParallel(
            trainer,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    weights = LossWeights(
        photo=float(cfg.train.lambda_photo),
        ssim=float(cfg.train.lambda_ssim),
        temp=float(cfg.train.lambda_temp),
        occ=float(cfg.train.lambda_occ),
        sparse=float(cfg.train.lambda_sparse),
        velocity=float(cfg.train.lambda_velocity),
    )
    amp = bool(cfg.train.amp) and device.type == "cuda"
    if bool(cfg.train.amp) and not amp and rank == 0:
        print("AMP is enabled in the config and CUDA is not available; training in fp32.")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    optimizer = torch.optim.AdamW(
        trainer.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    start = 0
    resume = cfg.train.resume
    if resume not in {None, "", "null"}:
        start = load_checkpoint(Path(str(resume)), trainer, optimizer, scaler)
    rasterizer = GaussianRasterizer(backend=_backend(str(cfg.train.backend), device))
    rasterizer.to(device)
    logger = TrainLogger(
        Path(str(cfg.train.log_dir)),
        enable_wandb=bool(cfg.train.wandb),
        project=str(cfg.project) if "project" in cfg else "aerosplat",
        config={"seed": seed, "steps": int(cfg.train.steps)},
        rank=rank,
    )
    cache = None
    if float(cfg.train.lambda_occ) != 0 or bool(cfg.train.occ_compute):
        from aerosplat.data.cache import CacheConfig, GroundTruthCache

        cache = GroundTruthCache(CacheConfig.from_hydra(cfg))

    history: list[dict[str, float]] = []
    steps = int(cfg.train.steps)
    accum = max(1, int(cfg.train.grad_accum_steps))
    log_every = int(cfg.train.log_every)
    ckpt_every = int(cfg.train.ckpt_every)
    trainer.train()
    stream = _cycle(loader, sampler)
    step = start
    try:
        while step < steps:
            optimizer.zero_grad(set_to_none=True)
            last: LossOutput | None = None
            last_rendered: dict[str, Tensor] | None = None
            last_targets: dict[str, Tensor] | None = None
            for micro in range(accum):
                batch = next(stream)
                keep_images = log_every > 0 and micro == accum - 1 and (step + 1) % log_every == 0
                sync_last = micro == accum - 1
                unsynced = distributed and not sync_last
                context = (
                    trainer.no_sync()
                    if unsynced and isinstance(trainer, DistributedDataParallel)
                    else nullcontext()
                )
                with context:
                    result, rendered, targets = _step(
                        trainer,
                        rasterizer,
                        batch,
                        weights,
                        cfg,
                        device,
                        amp,
                        cache,
                        keep_images,
                    )
                    if not torch.isfinite(result.total):
                        raise RuntimeError(f"non-finite loss at step {step}: {result.scalars()}")
                    scaler.scale(result.total / accum).backward()
                last = result
                last_rendered, last_targets = rendered, targets
            if amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainer.parameters(), float(cfg.train.clip_grad))
            scaler.step(optimizer)
            scaler.update()
            step += 1
            assert last is not None
            row = last.scalars()
            history.append(row)
            logger.scalars(step, row)
            if last_rendered is not None and last_targets is not None and rank == 0:
                logger.images(step, comparison_grid(last_rendered, last_targets))
            if rank == 0 and ckpt_every > 0 and step % ckpt_every == 0:
                save_checkpoint(
                    Path(cfg.train.log_dir) / f"step_{step:06d}.pt",
                    trainer,
                    optimizer,
                    scaler,
                    step,
                    seed,
                )
            if rank == 0 and log_every > 0 and step % log_every == 0:
                print(
                    f"step {step} total {row['total']:.4f} photo {row['photo']:.4f} "
                    f"ssim {row['ssim']:.4f} temp {row['temp']:.4f} "
                    f"occ {row['occ']:.4f} sparse {row['sparse']:.4f} "
                    f"velocity {row['velocity']:.4f}"
                )
        if rank == 0 and bool(cfg.train.save_last):
            save_checkpoint(
                Path(cfg.train.log_dir) / "last.pt",
                trainer,
                optimizer,
                scaler,
                step,
                seed,
            )
    finally:
        logger.close()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
    return history


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 4 photometric training.")
    parser.add_argument(
        "--config-name",
        default="config",
        help="Hydra config name. Laptop: experiment/laptop_8gb or experiment/laptop_16gb.",
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, for example train.steps=100")
    args = parser.parse_args(argv)
    from aerosplat.config import load_config

    cfg = load_config(args.overrides, config_name=args.config_name)
    history = run_training(cfg)
    if history and _world()[1] == 0:
        last = history[-1]
        print("final", " ".join(f"{key}={value:.4f}" for key, value in last.items()))


def _step(model, rasterizer, batch, weights: LossWeights, cfg, device, amp, cache, keep_images):
    images, cameras = _move_views(batch["images"], batch["cameras"], device)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        if amp
        else nullcontext()
    )
    with autocast:
        cloud = model(images, cameras)
    # The rasterizer and the losses stay fp32. AMP covers the ViT only.
    cloud.packed = cloud.packed.float()
    rendered = rasterizer(cloud.packed, cameras, valid=cloud.valid)
    cloud_next = None
    cameras_next = None
    dt = None
    if batch["images_next"] is not None and float(weights.temp) != 0.0:
        images_next, cameras_next = _move_views(batch["images_next"], batch["cameras_next"], device)
        with autocast:
            cloud_next = model(images_next, cameras_next)
        cloud_next.packed = cloud_next.packed.float()
        dt = batch["dt"].to(device)
    occ_logits = occ_target = occ_known = None
    if cache is not None:
        targets = []
        knowns = []
        for sample in batch["raw"]:
            grid = cache.get(sample)
            target, known = occupancy_target_from_grid(grid)
            targets.append(target)
            knowns.append(known)
        occ_target = torch.stack(targets, dim=0).to(device)
        occ_known = torch.stack(knowns, dim=0).to(device)
        module = model.module if isinstance(model, DistributedDataParallel) else model
        occ_logits = module.occ_head(occ_target.shape[0], tuple(occ_target.shape[-3:]))
    result = composite_loss(
        rendered,
        images,
        cloud=cloud,
        cloud_tp1=cloud_next,
        cameras_tp1=cameras_next,
        dt=dt,
        occ_logits=occ_logits,
        occ_target=occ_target,
        occ_known=occ_known,
        gt_velocity=None if batch["gt_velocity"] is None else batch["gt_velocity"].to(device),
        weights=weights,
        tau_max=float(cfg.train.tau_max),
        ssim_window=int(cfg.train.ssim_window),
        occ_pos_weight=float(cfg.train.occ_pos_weight),
        occ_focal_gamma=float(cfg.train.occ_focal_gamma),
        occ_reduction=str(cfg.train.occ_reduction),
        temp_depth_abs=float(cfg.train.temp_depth_abs),
        temp_depth_rel=float(cfg.train.temp_depth_rel),
        temp_max_search=float(cfg.train.temp_max_search),
    )
    logged_render = logged_target = None
    if keep_images:
        logged_render = {name: tensor.detach() for name, tensor in rendered.items()}
        logged_target = {name: tensor.detach() for name, tensor in images.items()}
    return result, logged_render, logged_target


def _collate_pairs(batch: list) -> dict:
    current = [item[0] for item in batch]
    collated = collate_samples(current)
    images, cameras = _views_from_collated(collated)
    nxt = [item[1] for item in batch]
    images_next = cameras_next = dt = None
    if all(sample is not None for sample in nxt):
        collated_next = collate_samples(nxt)
        images_next, cameras_next = _views_from_collated(collated_next)
        dt = torch.tensor(
            [
                float(after.timestamp - before.timestamp)
                for before, after in zip(current, nxt, strict=True)
            ],
            dtype=torch.float32,
        )
    velocities = [sample.gt_velocity for sample in current]
    gt_velocity = None
    if all(velocity is not None for velocity in velocities):
        gt_velocity = torch.stack(velocities, dim=0)
    return {
        "images": images,
        "cameras": cameras,
        "images_next": images_next,
        "cameras_next": cameras_next,
        "dt": dt,
        "gt_velocity": gt_velocity,
        "raw": [item[2] for item in batch],
    }


def _views_from_collated(collated: dict) -> tuple[dict[str, Tensor], dict[str, dict[str, Tensor]]]:
    images: dict[str, Tensor] = {}
    cameras: dict[str, dict[str, Tensor]] = {}
    for name, camera in collated["cameras"].items():
        images[name] = camera["image"]
        cameras[name] = {
            "K": camera["K"],
            "R": camera["R"],
            "t": camera["t"],
            "height": int(camera["image"].shape[-2]),
            "width": int(camera["image"].shape[-1]),
        }
    return images, cameras


def _move_views(images, cameras, device):
    moved_images = {name: tensor.to(device) for name, tensor in images.items()}
    moved_cameras = {
        name: {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in camera.items()
        }
        for name, camera in cameras.items()
    }
    return moved_images, moved_cameras


def _cycle(loader: DataLoader, sampler):
    epoch = 0
    while True:
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def _set_checkpointing(model: SplatTrainer, enabled: bool) -> None:
    model.splatter.lifter.backbone.encoder.gradient_checkpointing = enabled


def _backend(name: str, device: torch.device) -> str:
    if name == "auto" and device.type != "cuda":
        return "reference"
    return name


def _device(requested: str, local_rank: int) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("train.device=cuda but torch.cuda.is_available() is false")
        return torch.device("cuda", local_rank)
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _world() -> tuple[int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world, rank, local_rank


def _camera_names(value) -> list[str] | None:
    if value is None:
        return None
    names = list(value)
    return names or None


def _select_cameras(sample: MultiCameraSample, names: list[str] | None) -> MultiCameraSample:
    if not names:
        return sample
    missing = [name for name in names if name not in sample.cameras]
    if missing:
        raise KeyError(f"sample {sample.scene_id} has no cameras {missing}")
    cameras = {name: sample.cameras[name] for name in sample.camera_names if name in names}
    return MultiCameraSample(
        scene_id=sample.scene_id,
        sample_index=sample.sample_index,
        timestamp=sample.timestamp,
        cameras=cameras,
        gt_velocity=sample.gt_velocity,
    )


def _timeline(dataset) -> list[tuple[str, float]] | None:
    if hasattr(dataset, "_index"):
        return [(stamp.scene_id, float(stamp.timestamp)) for stamp in dataset._index]
    if hasattr(dataset, "groups"):
        return [(group.scene_id, float(group.timestamp)) for group in dataset.groups]
    return None


if __name__ == "__main__":
    main()
