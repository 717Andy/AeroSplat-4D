"""Run a few depth-supervised steps on the configured dataset.

The default is the tiny ViT on the synthetic fixture, on CPU. Pass Hydra
overrides to point it at a real pretraining root, for example:

    python scripts/pretrain_depth.py data=aerial_megadepth data.root=/data/amd model=vit_small
"""

from aerosplat.models.pretrain import main

if __name__ == "__main__":
    main()
