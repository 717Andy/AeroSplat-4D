# AeroSplat-4D

Feedforward 4D Gaussian splatting for urban air mobility. A multi-camera UAV rig
goes in; one forward pass predicts a Gaussian cloud with position, scale,
orientation, opacity, degree-3 spherical harmonics, and a velocity. Later phases
add a ViT backbone, a CUDA rasterizer, a glare head, and a local occupancy head.
This repository is Phase 1: the package layout, the sample schema, calibration,
and the ground-truth cache those heads will read.

The flight frame is ENU Z-up, cameras are OpenCV, and extrinsics are
world-to-camera. AirSim / Colosseum and TartanAir are converted into that frame
on load. The packed Gaussian tensor later phases must emit is `[B, N, 62]`.
`docs/ARCHITECTURE.md` defines the slices and the heads. `docs/DATASETS.md`
covers formats and why the Stanford Drone Dataset is not used.

## Layout

```
aerosplat/          package
  data/             sync, canonical / AirSim / TartanAir adapters, occupancy cache
  geometry/         poses, unprojection, voxelization, software raycast
  models/           [B, N, 62] packer and Phase 2–5 stubs
  capture/          Colosseum 4-camera settings and recorder
  vis/              multi-camera alignment check
configs/            Hydra configs (data + rig groups)
fixtures/tiny_rig/  synthetic 4-camera recording, no download
scripts/            fixture, capture, cache, sanity
tests/              CPU tests
```

Phase 2–5 modules raise `NotImplementedError`. `diff-gaussian-rasterization` is
not imported. CI is CPU-only lint and tests; it does not compile CUDA.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]" --no-deps
pip install numpy pyyaml omegaconf hydra-core pillow pytest ruff matplotlib
```

`--no-deps` matters if a CPU torch wheel is already installed: a normal
`pip install -e .` can replace it with a CUDA build. GPU training machines can
install a CUDA torch instead. The rasterizer extension is still a Phase 4
decision (`diff-gaussian-rasterization` or a maintained alternative such as
`gsplat`) and is not part of this install.

## Fixture and sanity check

The fixture is a rig flying east at 4 m/s: front and starboard at 10 Hz, port
at 10 Hz with a 20 ms offset, down at 5 Hz, each at its own resolution. Depth
is rendered from the same K and extrinsics the loader returns.

```bash
python scripts/generate_fixture.py
python scripts/sanity_alignment.py --index 2 --save outputs/alignment
python scripts/prepare_cache.py --limit 1
pytest
```

`sanity_alignment.py` unprojects depth, checks that each camera reprojects onto
itself, and reports where two cameras see the same surface. `--index 2` is the
0.2 s sample, where the slow downward camera has a frame.

## Real data

Colosseum (the maintained AirSim fork) is the primary four-camera source.
TartanAir is the public stereo set with depth and pose; it shares AirSim's
coordinate conventions, which is why it is the second adapter. Mid-Air is a
single forward camera and is not wired up. The Stanford Drone Dataset is a
single overhead view with tracks and no calibration; it is not a multi-view
training set.

```bash
# settings only, no simulator required
python scripts/capture_colosseum_rig.py --dry-run --output outputs/colosseum_settings

# with the Colosseum Python client installed and a sim running
python scripts/capture_colosseum_rig.py --output data/colosseum/flight --frames 100

# TartanAir trajectory (see docs/DATASETS.md for the directory layout)
python scripts/prepare_cache.py data=tartanair data.root=/path/to/P000 --limit 2
```

Hydra overrides use the same `data=` and `rig=` groups as `configs/`.

## Roadmap

| Phase | Contents |
| --- | --- |
| 1 | This tree. Ingestion, calibration, occupancy cache, tensor contract. |
| 2 | Shared ViT, pose tokens, epipolar lift. |
| 3 | Geometry, appearance, and dynamics heads → `[B, N, 62]`. |
| 4 | Differentiable rasterizer, photometric and temporal losses. |
| 5 | Glare residual, occupancy head, export. |

Disagreements with the original spec (scale bounds, SH versus flare, occupancy
BCE, the Stanford set, sub-millisecond inference) are listed in
`docs/ARCHITECTURE.md`. The loss equations in the spec PDF were images and are
not recoverable as text; do not implement a guessed formula as if it were exact.

## Tests

`pytest` generates its own tiny recordings under the temp directory and also
reads `fixtures/tiny_rig`. No dataset download and no GPU.
