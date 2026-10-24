# Phase 5: glare, occupancy, export

Phase 5 replaces the occupancy placeholder and adds the camera-side glare
layer. The rasterizer is still a separate call. The ONNX graph stops at the
Gaussian cloud, the glare images, and the occupancy volume.

## Glare

Lens flare is light the lens added. It is not radiance on a Gaussian. Degree-3
spherical harmonics stay on the cloud and are trained as clean scene color.

`RelightHead` reads, per camera:

* the lift feature map
* the sun direction in that camera (`d_cam = R @ d_world`)
* the degree-3 SH field, pooled over the Gaussians that belong to the camera
  (weight = valid × opacity) and evaluated toward the sun

The last layer is biased to `sigmoid(-4) ≈ 0.018`, so a fresh head is a faint
veil with gradient. The composite is

```
rendered = clamp(clean_render + glare, 0, 1)
```

`L_photo` and `L_ssim` compare `rendered` to the observed frame. `clean_render`
is the rasterizer output before the addition. A downstream view that should
not show the sun in the lens uses that tensor.

The sun comes from a compact ENU ephemeris when the sample has a Unix
timestamp (greater than 1e9) and latitude and longitude. Otherwise the
brightest pixel is unprojected. That estimate is detached. A white roof can
win it.

### Supervision

Two sources, same additive composite:

* A simulator pair. `CameraFrame.clean_image` is the flare-off frame.
  `paired_glare_target` takes the positive difference as the layer and the
  mask.
* A pasted flare. `FlareLibrary` reads a local copy of the Google Research
  flare-removal set (Wu et al., ICCV 2021, CC BY 4.0). The set is not
  downloaded. `train.flare_root` points at `flare/` plus an optional `mask/`,
  or at `Flare/` plus optional `Light_source/`. `train.flare_prob` is the
  paste rate. Augmentation is off when `flare_root` is null, so a seeded
  overfit does not draw an extra random number.

`λ_glare` (default 0.1) is the mean absolute error of the predicted layer
inside the known mask. `λ_glare_l1` (default 0.02) is the mean of the layer
outside that mask, so glare cannot explain a building.

### Sun slider

`evaluate_relit_sh` and `rotate_sh_coeffs` are the utility a later frontend
slider can call. They rotate the view direction so the lobe that sat on the
old sun sits on the new one, then evaluate the same coefficients (plus the
rasterizer's `+0.5`). `rotate_sh_coeffs` fits new coefficients by least
squares on a Fibonacci sphere, for a shader that only knows how to evaluate
SH.

This is an environment rotation. It does not recompute shadows,
interreflections, or the glare layer. Degree 3 cannot make a hard solar disk.
A new sun direction needs a new glare prediction if the lens artifact should
move with it.

## Occupancy and collision

`OccupancyHead` splats the packed cloud into the local flight-frame grid
(nearest voxel) and runs a small 3D convolution. Inputs are opacity,
`wire_prob`, scale, the SH DC color, and the fractional voxel offset. A
`wire_skip` parameter starts at 1, so a wire-flagged Gaussian lights its cell
before the convolution trains. The scatter index is not differentiable.
Photometry moves the means; the occupancy loss moves opacity, `wire_prob`,
and the convolution.

Outputs, all on the coarse grid (the cache default is 0.5 m, not changed):

| Field | Meaning |
| --- | --- |
| `occupancy_logits` | solid surface, occupied versus free |
| `wire_logits` | a cable crosses this cell |
| `wire_direction` | unit tangent |
| `wire_offset` | where the cable sits in the cell, in `[-0.5, 0.5]` voxel units |

A 1–3 cm cable is the line, not a filled 0.5 m cube. `query_collision` is the
function `GET /api/v1/occupancy` will call. It does not run the ViT. Given a
polyline and a corridor radius it returns `risk` (highest occupancy or wire
probability the tube touches), `nearest_distance_m`, and the solid and wire
risks separately. Solid cells use the distance to the voxel box. Wire cells
use the distance to the line segment, including when the cell is also marked
solid, so a cable sharing a cell with a facade is measured as the cable. A
logit of 0 is sigmoid 0.5 and is not a hit: the threshold is strict `> 0.5`.

`L_occ` is Andy's mean BCE on known solid cells, plus the same BCE on the wire
channel. Unknown cells stay masked. Wire voxels use `occ_wire_pos_weight`
(default 8) and `occ_wire_focal_gamma` (default 2) so a missed cable costs
more than a missed facade. `λ4` defaults to 0.1. `LossWeights.occ` in the
dataclass stays 0 so a caller that constructs weights by hand does not turn
the term on by accident.

Two extra pieces fold into the occupancy term when their weights are positive:

* `occ_smooth_weight` (default 0.05): neighboring wire cells should agree on
  tangent. That is a local line prior. It is not one catenary for the whole
  corridor. A crane cable can be vertical.
* `occ_line_weight` (default 0.1): distance from known cable points to the
  predicted line, in voxels. This term uses the first sample when the batch
  is larger than 1. Laptop training uses batch 1.

### Wire ground truth

`cache.wire_ids` (label maps) and `cache.wire_colors` (RGB palettes, 0–255)
say which segmentation pixels are cables. Empty means no wire class. The
capture config has to name them; the palette is not inferred. Fusion marks
those coarse cells and stores the 3D hit points. A fine grid at
`cache.wire_voxel_size` (default 0.02 m) over `cache.wire_extent_m` (default
4 m) is written only when points exist and the volume is at most 8 million
cells. Old `.npz` files without wire keys load as an empty wire channel.
Padding id 0 on a resized segmentation is not a wire label: the cache is built
from the unresized sample.

## Export and latency

`aerosplat.export` writes static-shape ONNX (opset 17). Three targets:

| `--what` | Graph |
| --- | --- |
| `relight` | glare head |
| `occupancy` | occupancy head |
| `network` | ViT through Gaussians, glare, and occupancy |

The rasterizer is not in the graph. Render clean, then add the exported glare.
`scripts/build_tensorrt.py` builds an engine on the GPU that will run it. An
Orin engine does not load on a desktop RTX. The default precision is FP16.
`--int8` attaches `Int8Calibrator`, which does not import TensorRT until the
build. CI does not install TensorRT.

`scripts/benchmark_latency.py` prints milliseconds per stage on the machine
that runs it, next to planning estimates:

| Stage | Planning estimate |
| --- | --- |
| collision query, 32³ | 0.2 ms |
| occupancy head, 32³, GPU | 2 ms |
| ViT-S, 224 px, four cameras, Orin | 50 ms |
| ViT-S, 224 px, four cameras, RTX | 15 ms |
| rasterizer, separate | 5 ms |

Sub-millisecond is a realistic budget for the collision query. It is not a
budget for the ViT. Replace the estimates with a measurement from the script
before quoting a number for a specific board. A CPU timing of `vit_tiny` is
not an Orin timing of `vit_small`.

One measurement on the CPU used for these tests, `vit_tiny` at 32 px, one
camera, a 16³ grid, five repeats:

| Stage | Milliseconds |
| --- | --- |
| lifter | 1.2 |
| Gaussian heads | 2.5 |
| occupancy head | 3.2 |
| lifter, Gaussians, and glare | 4.6 |
| full network, including occupancy | 7.6 |
| collision query, untrained volume | 0.27 |

A separate 32³ volume with a diagonal of occupied cells and a line of wire
cells, queried with 40 waypoints, took 0.39 ms on the same CPU. That is the
stage that fits a sub-millisecond budget. The ViT row does not.

## Commands

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pyyaml omegaconf hydra-core pillow pytest ruff matplotlib onnx onnxruntime
pip install -e . --no-deps
pytest
ruff check .
ruff format --check .

# Train. Flare paste stays off until flare_root is a local CC BY 4.0 directory.
python scripts/train.py model=vit_tiny train.steps=20 train.device=cpu train.backend=reference

# Wire class, once the recording's segmentation ids are known.
python scripts/prepare_cache.py cache.wire_ids=[3] --limit 1

python scripts/export_onnx.py --what occupancy --output outputs/occupancy.onnx
python scripts/export_onnx.py --what relight --output outputs/relight.onnx
python scripts/export_onnx.py --what network model=vit_tiny model.img_size=32 --output outputs/network.onnx

# On the Jetson or the RTX, after TensorRT is installed:
python scripts/build_tensorrt.py --onnx outputs/occupancy.onnx --engine outputs/occupancy.plan
python scripts/build_tensorrt.py --onnx outputs/occupancy.onnx --engine outputs/occupancy_int8.plan --int8

python scripts/benchmark_latency.py model=vit_tiny model.img_size=32 train.device=cpu
```

## Open questions

1. Additive clipping versus a screen blend. This tree adds and clips at 1,
   which matches a saturating sensor. A screen blend would keep more of the
   scene under a bright flare and would change both the photometric target
   and the pasted-flare composite.
2. The brightest-pixel sun fails on white roofs and cloud decks. A synthetic
   shard copies latitude, longitude, and a Unix timestamp onto the sample, and
   the ephemeris wins for those recordings. Other recordings still omit the
   location, and those still take the image estimate.
3. The SH slider is an environment rotation, not a lighting solve. Glare has
   to be predicted again for a new sun. Degree 3 cannot represent a solar disk.
4. A cell that contains both a facade and a cable is measured as the cable
   line only. A trajectory that hits the facade and misses the cable can look
   clear. A second solid distance, reported beside the line distance, would
   fix that and is not implemented.
5. Which segmentation ids or colors are wires is capture configuration
   (`cache.wire_ids`, `cache.wire_colors`). The AirSim mesh palette is not
   loaded automatically.
6. `L_occ` is mean solid BCE plus mean wire BCE. Andy wrote one unweighted
   sum. `λ4 = 0.1` is a starting point, not a number he set. The same is true
   of `λ_glare` and `λ_glare_l1`.
7. With no paired flare and no paste, only `λ_glare_l1` acts, and it pushes
   the layer toward zero. That is intentional until a flare source exists. It
   also means the head does not learn a flare from photometric error alone
   while that weight is on: the photometric term wants the layer, and the L1
   term does not.
8. The latency table above is a planning estimate. The benchmark script is
   the measurement. Do not treat the ViT row as sub-millisecond.
9. TensorRT engines are not portable across GPU architectures. The INT8
   calibrator is a stub that yields batches; it is not a tuned calibration set.
10. The Google flare set is not in CI and is not vendored. Tests synthesize
    a blob.
11. Phase 4 items stay open: high-resolution crops, `anchor_valid` slots that
    still render, which Gaussian owns which logged actor, and ego-motion.
12. There is no global catenary parameter. The prior is local tangent
    smoothness plus point-to-line.
13. `wire_point_loss` runs on the first sample when the batch is larger than 1.
14. A Phase 4 `last.pt` will not load. The trainer state dict grew the relight
    and occupancy heads, and the occupancy head is no longer one shared logit.
