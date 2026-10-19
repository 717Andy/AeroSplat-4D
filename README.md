# AeroSplat-4D

Feedforward 4D Gaussian splatting for urban air mobility. A multi-camera UAV rig
goes in; one forward pass predicts a Gaussian cloud with position, scale,
orientation, opacity, degree-3 spherical harmonics, and a velocity. Phase 1 is
the sample schema, calibration, and the occupancy cache. Phase 2 is the shared
ViT, pose-conditioned tokens, and the plane-sweep lift. Phase 3 reads that
lift and packs `[B, N, 62]`: anchor plus a bounded offset, post-exp scales
with an anisotropic cap on wires, a unit wxyz quaternion, sigmoid opacity,
degree-3 SH whose DC matches the pixel, and velocity in m/s. `N` is a fixed
slot budget. A wire allocates extra slots by clearing a mask, not by changing
the tensor length. Phase 5 adds an image-space glare layer, a coarse occupancy
volume with a sub-voxel wire line, and ONNX export of the network. A CPU
reference splatter renders tests. The CUDA rasterizer is imported only when
it is installed.

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
  models/           ViT, plane-sweep lift, splat heads, rasterizer, losses
  capture/          AirSim-family rig capture (Cosys-AirSim, Colosseum) and render benchmark
  vis/              multi-camera alignment check
configs/            Hydra configs (data + rig groups)
fixtures/tiny_rig/  synthetic 4-camera recording, no download
scripts/            fixture, capture, cache, sanity
tests/              CPU tests
```

Phase 4 trains through `diff-gaussian-rasterization` when that extension
imports and CUDA is available. The import is `importlib` inside the call, so
CPU CI never compiles it. The reference splatter renders the CPU tests.
`L_occ` uses `OccupancyHead` and the Phase 1 cache. The default weight is
`λ4 = 0.1`. Glare, collision queries, and export are described in
`docs/PHASE5.md`.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]" --no-deps
pip install numpy pyyaml omegaconf hydra-core pillow pytest ruff matplotlib onnx onnxruntime
```

`--no-deps` matters if a CPU torch wheel is already installed: a normal
`pip install -e .` can replace it with a CUDA build. GPU training machines
install a CUDA torch first, then this package with `--no-deps`. The CUDA
rasterizer is a separate source build. Windows steps for an MSI Vector 17 HX
are in `docs/RASTERIZER.md`. CPU runs use the reference splatter.

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
| 3 | Geometry, appearance, and dynamics heads, wire slots, `[B, N, 62]`. |
| 4 | Differentiable CUDA rasterizer, photometric, reprojected temporal, and sparsity losses. |
| 5 | Glare layer, occupancy and collision, ONNX and TensorRT export. |

Disagreements with the original spec (scale bounds, SH versus flare, occupancy
BCE, the Stanford set, sub-millisecond inference) are listed in
`docs/ARCHITECTURE.md`. Andy's loss equations are implemented there.
`λ1 = 0.8`, `λ2 = 0.2`. `L_sparse` is his ungated volume hinge divided by `N`,
with `τ_max = 0.008` m³. `L_temp` reprojects `x + v Δt` into the next frame
instead of comparing pixel slots, which are wrong under ego-motion. `λ4 = 0.1`
now that the occupancy head exists. `λ_glare = 0.1` supervises a known flare
layer when one is present.

## Training

Configs are Hydra. The loop is `scripts/train.py`: AMP on CUDA, gradient
checkpointing on the ViT, gradient accumulation, seed control, checkpoints,
and a JSONL of every loss term. TensorBoard is used when it is installed
(`pip install tensorboard`). W&B is off unless `train.wandb=true` and `wandb`
imports. Rendered-versus-ground-truth grids are written as PNG next to the log.

8 GB laptop (ViT-S/14, 224 px, batch 1, four cameras). Install the rasterizer
first (`docs/RASTERIZER.md`):

```bash
python scripts/train.py --config-name experiment/laptop_8gb
```

16 GB laptop (ViT-B/14, 224 px, same splat budget):

```bash
python scripts/train.py --config-name experiment/laptop_16gb
```

ViT-S at 336 px on that 16 GB card, instead of ViT-B:

```bash
python scripts/train.py --config-name experiment/laptop_16gb model=vit_small model.img_size=336 train.log_dir=outputs/train_16gb_336
```

If the 8 GB forward runs out of memory, lower the plane count or the image
size. 112 is divisible by the patch size of 14. Accumulation does not shrink
one forward.

```bash
python scripts/train.py --config-name experiment/laptop_8gb model.depth_planes=16
python scripts/train.py --config-name experiment/laptop_8gb model.img_size=112
```

Multi-GPU, later, with torchrun. One process per GPU. Rank 0 writes logs and
checkpoints. The default backbone is ViT-S; override `model=vit_large` only
when each GPU can hold it.

```bash
torchrun --nproc_per_node=4 scripts/train.py --config-name experiment/ddp
```

A short CPU smoke, no CUDA extension, reference splatter, tiny ViT:

```bash
python scripts/train.py model=vit_tiny train.steps=2 train.device=cpu train.backend=reference train.lambda_temp=0
```

`L_temp` needs a later frame of the same scene. The fixture has that. Static
pretraining sets (`aerial_megadepth`, `matrixcity`) do not, and the term is
skipped. Actor-velocity supervision stays at weight 0 until a sample carries
`gt_velocity` (m/s, flight frame) from logged Cosys/Colosseum actor poses.
`train.lambda_velocity` turns it on.

Useful overrides: `train.steps`, `train.lr`, `train.tau_max`,
`train.lambda_sparse`, `train.grad_accum_steps`, `train.occ_compute=true`
(runs the placeholder occupancy term even while `lambda_occ` is 0),
`data.root`. Checkpoints land in `train.log_dir` (`last.pt`, and
`step_NNNNNN.pt` every `ckpt_every` steps). Resume with
`train.resume=outputs/train_8gb/last.pt`.

## Workbench

The debugging UI lives outside the training package, in `server/` and `web/`.
It serves recorded episodes, runs the feedforward model, and draws sorted
EWA splats, occupancy, wires, and camera frustums. No simulator and no GPU
are required: `episodes/demo` is baked from the tiny fixture.

```bash
python -m server.dev
```

That starts the API on `127.0.0.1:8000` and the Vite UI on `127.0.0.1:5173`.
Windows and macOS steps, the AGS1 / AOC1 binary layouts, and the sun-slider
contract are in `docs/WORKBENCH.md`.

## Tests

`pytest` generates its own tiny recordings under the temp directory and also
reads `fixtures/tiny_rig`. No dataset download and no GPU. Workbench tests
are a separate path because they need the API extras:

```bash
pip install -r server/requirements.txt
pytest server/tests
cd web && npm ci && npm run typecheck && npm test
```

## Phase 2

The lifter is `aerosplat.models.build_lifter`. It returns per-camera features,
a depth distribution, and `[B, N, 3]` anchors in the flight frame. `N` there is
the number of lift cells. `aerosplat.models.build_model` runs the lifter and
the Phase 3 predictor and returns a `GaussianCloud` whose packed tensor is
`[B, N_slots, 62]` with `N_slots = N_cells * slots_per_cell`.
`docs/ARCHITECTURE.md` records why the lift is a plane sweep, why wire scales
are capped anisotropically, and why Gaussian `i` is a pixel slot.

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
