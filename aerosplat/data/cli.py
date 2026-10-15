"""Precompute occupancy caches for a configured dataset."""

from __future__ import annotations

import argparse

from aerosplat.config import load_config
from aerosplat.data.cache import CacheConfig, GroundTruthCache
from aerosplat.data.dataset import build_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Cache occupancy grids for a multi-camera dataset."
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, for example data=tartanair data.root=/data/P000",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only cache the first N samples.")
    args = parser.parse_args(argv)
    cfg = load_config(args.overrides)
    dataset = build_dataset(cfg)
    cache = GroundTruthCache(CacheConfig.from_hydra(cfg))
    count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    for index in range(count):
        sample = dataset[index]
        grid = cache.get(sample)
        occupied = int(grid.occupied.sum())
        free = int(grid.free.sum())
        print(
            f"{sample.scene_id} #{sample.sample_index} t={sample.timestamp:.3f}s "
            f"occupied={occupied} free={free} -> {cache.path_for(sample)}"
        )


if __name__ == "__main__":
    main()
