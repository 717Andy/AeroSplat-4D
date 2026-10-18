"""Per-stage latency of the feedforward network and the collision query.

The numbers this script prints are measurements on the machine that runs it.
A CPU measurement of ``vit_tiny`` is not an Orin measurement of ``vit_small``.
What does transfer is the split:

* the collision query on a local occupancy grid is sub-millisecond
* the occupancy head (scatter plus a small 3D conv) is a few milliseconds
* the ViT over four cameras is tens of milliseconds on an embedded GPU, and
  that is before the rasterizer

The spec's sub-millisecond budget fits the query. It does not fit the ViT.
``docs/PHASE5.md`` records the estimates used for planning.
"""

from __future__ import annotations

import argparse
import time

import torch

from aerosplat.models.collision import query_collision
from aerosplat.models.factory import build_model
from aerosplat.models.occupancy_head import OccupancyHead
from aerosplat.models.relight import RelightHead
from aerosplat.train import SplatTrainer

# Planning numbers, not a measurement of this process. The script prints both.
ESTIMATES_MS = {
    "collision_query_32cubed": 0.2,
    "occupancy_head_32cubed_gpu": 2.0,
    "vit_small_224_four_cameras_orin": 50.0,
    "vit_small_224_four_cameras_rtx": 15.0,
    "rasterizer_separate": 5.0,
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time(fn, device: torch.device, repeats: int) -> float:
    fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync(device)
    return (time.perf_counter() - start) * 1000.0 / repeats


def benchmark(cfg, repeats: int = 10, device: str = "cpu") -> dict[str, float]:
    """Milliseconds per stage. The model is ``cfg``'s ViT, in eval mode."""
    torch_device = torch.device(device)
    splatter = build_model(cfg).to(torch_device).eval()
    splatter.lifter.backbone.encoder.gradient_checkpointing = False
    trainer = SplatTrainer(
        splatter,
        RelightHead(feature_dim=int(cfg.model.embed_dim)).to(torch_device),
        OccupancyHead().to(torch_device),
    ).eval()
    size = int(cfg.model.img_size)
    images = {"front": torch.rand(1, 3, size, size, device=torch_device)}
    cameras = {
        "front": {
            "K": torch.tensor(
                [[size, 0, size / 2], [0, size, size / 2], [0, 0, 1]],
                dtype=torch.float32,
                device=torch_device,
            ).view(1, 3, 3),
            "R": torch.eye(3, device=torch_device).view(1, 3, 3),
            "t": torch.zeros(1, 3, device=torch_device),
        }
    }
    sun = torch.tensor([[0.2, 0.1, 0.9]], device=torch_device)
    sun = sun / sun.norm()
    origin = torch.zeros(1, 3, device=torch_device)
    grid = (16, 16, 16)
    with torch.no_grad():
        lifter_ms = _time(
            lambda: splatter.lifter(images, cameras, match="hybrid"),
            torch_device,
            repeats,
        )
        lifted = splatter.lifter(images, cameras, match="hybrid")
        predictor_ms = _time(
            lambda: splatter.predictor(lifted, images, cameras),
            torch_device,
            repeats,
        )
        cloud, glare, volume = trainer(images, cameras, sun, origin, 0.5, grid)
        occupancy_ms = _time(
            lambda: trainer.occupancy.forward_cloud(cloud, origin, 0.5, grid),
            torch_device,
            repeats,
        )
        relight_ms = _time(
            lambda: trainer(
                images,
                cameras,
                sun,
                None,
                None,
                None,
            ),
            torch_device,
            repeats,
        )
        # The call above includes the ViT. The glare-only number is the difference
        # from a cloud-only forward, floored at zero when the timer noise wins.
        network_ms = _time(
            lambda: trainer(images, cameras, sun, origin, 0.5, grid),
            torch_device,
            repeats,
        )
        assert volume is not None
        query_ms = _time(
            lambda: query_collision(volume, torch.zeros(8, 3, device=torch_device), 0.5),
            torch_device,
            repeats,
        )
    del glare
    return {
        "lifter_ms": lifter_ms,
        "gaussian_heads_ms": predictor_ms,
        "occupancy_head_ms": occupancy_ms,
        "network_with_glare_ms": relight_ms,
        "network_total_ms": network_ms,
        "collision_query_ms": query_ms,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    from aerosplat.config import load_config

    cfg = load_config(args.overrides, config_name=args.config_name)
    measured = benchmark(cfg, repeats=args.repeats, device=args.device)
    print("measured_ms")
    for key, value in measured.items():
        print(f"  {key}: {value:.3f}")
    print("planning_estimates_ms")
    for key, value in ESTIMATES_MS.items():
        print(f"  {key}: {value:.1f}")
    print(
        "collision_query_ms is the sub-millisecond stage. network_total_ms is the ViT and is not."
    )


if __name__ == "__main__":
    main()
