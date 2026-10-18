"""Phase 4 training entry point.

Laptop (MSI Vector 17 HX, Windows, CUDA), from the repo root::

    python scripts/train.py --config-name experiment/laptop_8gb
    python scripts/train.py --config-name experiment/laptop_16gb

Multi-GPU later::

    torchrun --nproc_per_node=4 scripts/train.py --config-name experiment/ddp

CPU smoke (no CUDA rasterizer)::

    python scripts/train.py model=vit_tiny train.steps=2 train.device=cpu train.backend=reference
"""

from aerosplat.train import main

if __name__ == "__main__":
    main()
