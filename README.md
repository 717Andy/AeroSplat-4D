# AeroSplat-4D

Feedforward 4D Gaussian splatting for urban air mobility. A multi-camera UAV rig
goes in; one forward pass predicts a Gaussian cloud with position, scale,
orientation, opacity, degree-3 spherical harmonics, and a velocity. Phase 1 is
the sample schema, calibration, and the occupancy cache. Phase 2 is the shared
ViT, pose-conditioned tokens, and the plane-sweep lift that hands Phase 3 a
set of flight-frame anchors. The rasterizer, the glare head, and the occupancy
head are still stubs.

The flight frame is ENU Z-up, cameras are OpenCV, and extrinsics are
world-to-camera. AirSim-family simulators (Cosys-AirSim and archived Colosseum)
and TartanAir are converted into that frame on load. The packed Gaussian tensor
later phases must emit is `[B, N, 62]`.
`docs/ARCHITECTURE.md` defines the slices and the heads. `docs/DATASETS.md`
covers formats and why the Stanford Drone Dataset is not used.

## Layout

```
aerosplat/          package
  data/             sync, canonical / AirSim / TartanAir adapters, occupancy cache
  geometry/         poses, unprojection, voxelization, software raycast
  models/           ViT, plane-sweep lift, [B, N, 62] packer, Phase 3–5 stubs
  capture/          AirSim-family rig capture (Cosys-AirSim, Colosseum) and render benchmark
  vis/              multi-camera alignment check
configs/            Hydra configs (data + rig groups)
fixtures/tiny_rig/  synthetic 4-camera recording, no download
scripts/            fixture, capture, cache, sanity
tests/              CPU tests
```

Phase 3–5 heads and the rasterizer raise `NotImplementedError`.
`diff-gaussian-rasterization` is not imported. CI is CPU-only lint and tests;
it does not compile CUDA and it does not download pretrained weights.

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

Cosys-AirSim is the maintained four-camera source. Colosseum, the previous
AirSim fork, is archived (read-only); its adapter and
`scripts/capture_colosseum_rig.py` stay for existing recordings. Both simulators
share one backend (`--simulator cosys` or `--simulator colosseum`) because the
Python APIs match. Settings version, the Scene-RGB flip, segmentation id space,
and `External` pose frames differ and are handled in that backend.
TartanAir is the public stereo set with depth and pose; it shares the AirSim
NED convention, which is why it is the other adapter. Mid-Air is a single
forward camera and is not wired up. The Stanford Drone Dataset is a single
overhead view with tracks and no calibration; it is not a multi-view training set.

```bash
# settings only, no simulator required
python scripts/capture_cosys_rig.py --dry-run --output outputs/cosys_settings
python scripts/capture_colosseum_rig.py --dry-run --output outputs/colosseum_settings

# with the cosysairsim client installed and Cosys-AirSim running
python scripts/capture_cosys_rig.py --output data/cosys/flight --frames 100

# archived Colosseum client, same rig
python scripts/capture_colosseum_rig.py --output data/colosseum/flight --frames 100

# TartanAir trajectory (see docs/DATASETS.md for the directory layout)
python scripts/prepare_cache.py data=tartanair data.root=/path/to/P000 --limit 2
```

Hydra overrides use the same `data=` and `rig=` groups as `configs/`.

## Roadmap

| Phase | Contents |
| --- | --- |
| 1 | Ingestion, calibration, occupancy cache, tensor contract. |
| 2 | Shared ViT, pose tokens, plane-sweep lift, depth pretraining. |
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

## Phase 2

The lifter is `aerosplat.models.build_lifter`. It returns per-camera features,
a depth distribution, and `[B, N, 3]` anchors in the flight frame. `N` is the
number of lift cells; Phase 3 packs those into `[B, N, 62]` once the other
heads exist. `docs/ARCHITECTURE.md` records why the lift is a plane sweep.

Laptop training budgets are in `configs/model/`: ViT-S at 224px for 8 GB,
ViT-B at 224px for 16 GB, batch size 1, four cameras. `vit_large` matches an
aerial DUSt3R / MASt3R checkpoint and does not fit that budget unfrozen.

```bash
python scripts/pretrain_depth.py
```

That runs two depth-supervised steps of the tiny model on the fixture, on CPU.

Rendering speed of the 4-camera rig is a separate measurement.
`docs/BENCHMARK.md` is the Windows / Unreal walkthrough for an MSI Vector 17
HX with an NVIDIA RTX GPU. The script is `scripts/benchmark_sim_render.py`.
