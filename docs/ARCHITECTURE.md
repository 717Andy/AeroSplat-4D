# Architecture

AeroSplat-4D is a feedforward model: one pass over a synchronized multi-camera
rig predicts a 4D Gaussian cloud. Phase 1 is the data contract. Phase 2 is the
shared ViT and the plane-sweep lift that turns it into per-pixel features and
flight-frame anchors. The splat heads, the CUDA rasterizer, and the losses are
still stubs in `aerosplat.models`.

```
N camera frames (image, K, R|t, time, optional depth)
        |
        v
shared ViT  +  pose-conditioned tokens          (Phase 2)
        |
        v
epipolar / frustum lift                         (Phase 2)
        |
        +---------------------------+----------------------+
        v                           v                      v
 geometry head              appearance head         dynamics head     (Phase 3)
 xyz, scale, quat           opacity, SH degree 3    velocity m/s
        |                           |                      |
        +---------------------------+----------------------+
                                |
                                v
                    packed Gaussians [B, N, 62]
                                |
                +---------------+----------------+
                v                                v
        CUDA rasterizer                   occupancy head          (Phase 4 / 5)
        photometric + temporal            local voxel logits
                                          supervised by the cache
                |
                v
        relight head (consumes SH, predicts a glare residual)     (Phase 5)
```

## Flight frame

Every sample uses one frame, regardless of the source dataset.

| | |
| --- | --- |
| World | Right-handed ENU, Z up. X east, Y north, Z up. |
| Camera | OpenCV. X right, Y down, Z forward. |
| Extrinsics | World-to-camera. `X_cam = R @ X_world + t`. |
| Depth | Planar Z in meters (the camera-frame Z, not the ray length). 0 means no return. |
| Quaternion | wxyz, unit length, `w >= 0` when we convert from a matrix. |
| Velocity | Meters per second in the world frame. The spec does not state a unit; this is the one the temporal loss can multiply by `Δt` in seconds. |

AirSim, Colosseum, and TartanAir use NED (X north, Y east, Z down) and a camera
with X forward, Y right, Z down. `aerosplat.geometry.coordinates` converts both.
Skipping that conversion makes unprojection disagree with any OpenCV rasterizer.

`R` and `t` on a `CameraFrame` are the pose at the image timestamp. `R_query` and
`t_query` are the pose interpolated to the sample timestamp. Depth is unprojected
with the capture pose. Using the query pose on an older depth map moves the points
off the surface by `velocity * dt`.

## Sample schema

`MultiCameraSample` is one query time.

* `scene_id`, `sample_index`, `timestamp`
* `cameras`: ordered dict of `CameraFrame`

`CameraFrame` holds `image` `[3, H, W]` in `[0, 1]`, `K` `[3, 3]`, `R` `[3, 3]`,
`t` `[3]`, `timestamp`, optional `depth` `[H, W]`, plus the query pose and a
`sync` tag (`exact`, `nearest`, `interpolated`, `clamped`).

Cameras in one sample may have different `H` and `W`. They are not stacked into
`[N, 3, H, W]`. `collate_samples` stacks a batch per camera name, and only when
that name has one resolution across the batch. A resize transform scales `K`
with the image. It does not letterbox.

## Gaussian tensor `[B, N, 62]`

`B` is the number of timesteps in the batch. It is not a flattened video. The
temporal loss compares the packed tensor at `t` with the packed tensor at `t+1`.
The spec's wording ("B is the batch size (video frame sequence)") is ambiguous;
this is the reading that matches a feedforward step plus a velocity loss.

`N` is the number of Gaussians at that timestep. The spec does not fix `N`. A
batch will pad to one `N`. A validity mask is not part of the 62 and does not
exist yet.

| Slice | Count | Field |
| --- | --- | --- |
| 0:3 | 3 | xyz, world meters |
| 3:6 | 3 | scale, positive meters (activated) |
| 6:10 | 4 | quaternion wxyz, normalized |
| 10:11 | 1 | opacity in `[0, 1]` (activated) |
| 11:59 | 48 | SH degree 3, RGB. `(degree+1)^2 = 16` coeffs, times 3 |
| 59:62 | 3 | velocity, world m/s |

SH is coefficient-major, then RGB. `packed[..., 11:59].reshape(B, N, 16, 3)` is
the layout `diff-gaussian-rasterization` takes as `(N, 16, 3)`.

The packed values are physical parameters, not logits. Heads that predict
log-scale and logit-opacity go through `physical_from_raw` (exp, sigmoid,
normalize) before packing. The rasterizer reads `packed[..., :59]`. Velocity
stays in the tensor for the temporal loss and is not a rasterizer input.

`GaussianLayout` in `aerosplat.models.gaussians` is the source of truth for the
slices. Tests check that the names and the slices add up to 62.

## Heads

| Head | Phase | Outputs | Packed into the 62? |
| --- | --- | --- | --- |
| Geometry | 3 | xyz, scale, quat | yes |
| Appearance | 3 | opacity, SH | yes |
| Dynamics | 3 | velocity | yes, not read by the rasterizer |
| Relight | 5 | glare residual RGB | no; consumes SH |
| Occupancy | 5 | local voxel logits | no; supervised by the cache |

Phase 3 and Phase 5 heads still raise `NotImplementedError`. `GaussianRasterizer`
checks the packed shape and then raises. It does not import a CUDA extension.
The Phase 2 modules below are implemented.

## Phase 2

`MultiViewLifter` is the call Phase 3 should make. It returns a `LiftedGeometry`:

* per camera, a feature map, a depth distribution over planes, the expected
  planar depth, and the flight-frame points those depths unproject to
* `anchors` `[B, N, 3]` and `anchor_features` `[B, N, C]`, the same `N` a
  geometry head would use if it emitted one Gaussian per anchor
* `gaussian_xyz()` is the slice that packs into `GaussianLayout.XYZ`. The other
  59 channels are not filled here

`N` is still not fixed by the spec. It is the number of lift-grid cells across
the rig. Phase 3 can subsample it. The lifter does not pad and does not emit
opacity, scale, quaternion, SH, or velocity.

### Backbone

One ViT, shared across every camera. The block names match DINOv2
(`patch_embed.proj`, `blocks.*.attn.qkv`, `blocks.*.mlp.fc1`, LayerScale
`ls1` / `ls2`) and the CroCo encoder used by DUSt3R and MASt3R (`enc_blocks`
is renamed on load). `aerosplat.models.pretrained` copies any key whose shape
matches and skips the rest, including the DUSt3R decoder. A different
`pos_embed` grid is interpolated. Nothing is downloaded unless
`pretrained_allow_download` is set. CI does not download weights.

Configs:

| config | width | laptop note |
| --- | --- | --- |
| `vit_small` | ViT-S/14, dim 384 | 8 GB target at 224px, batch 1, 4 cameras, gradient checkpointing. 16 GB has room for 336px. |
| `vit_base` | ViT-B/14, dim 768 | 16 GB target at 224px. Does not fit 8 GB. |
| `vit_large` | ViT-L/16, dim 1024, depth 24 | Matches aerial DUSt3R / MASt3R. Above an 8–16 GB laptop for unfrozen 4-camera training. |
| `vit_tiny` | dim 32, 2 blocks | CPU fixture and tests only. |

Do not train at 1080p on the laptop. The render benchmark measures the
simulator at 1080p and 720p. That is a different number from training VRAM.

Cross-view attention is a short stack over the concatenated patch tokens.
There is no learned camera index. Permuting the cameras permutes the tokens.
Images must be a multiple of the patch size. `prepare_rig` resizes and pads.

### Pose

Each patch token gets a Plücker ray in the flight frame, `(direction, origin ×
direction)`, from `K` and `R|t`. Optionally a sinusoidal embedding of the
flattened intrinsics and extrinsics is added as a per-camera bias. The last
linear layers are zero-initialized on `vit_small` and `vit_base` so a loaded
encoder does not move until those layers train.

### Why a plane sweep

The spec allows a cost volume or a frustum projection. The lift does both
jobs, in that order. A plane sweep warps the other cameras onto hypothesized
depths with the known calibration and scores the match. That is the part that
uses overlap. A frustum head that only regresses a depth along a ray never
looks at the other view, so it cannot be checked on the fixture without
training, and it ignores the extrinsics Phase 1 exists to get right.

The sweep assumes fronto-parallel planes in the reference camera. That is the
MVSNet hypothesis. It is approximate on slanted walls and on this rig, whose
side cameras sit about 90° off the nose, so their overlap with the nose is a
grazing band. The photometric cost is the initialization. A 1×1 convolution
from the ViT features adds a residual on the depth logits (`match="hybrid"`),
which is what the depth-pretraining loop updates. `match="photometric"` leaves
the residual off.

The mode of the cost is the geometric readout. The expected depth is the mean
of the softmax and is only sharp when the surface is textured. The fixture's
own albedo is flat on each face, so the correctness test re-renders the
fixture boxes from the front pose and a 0.5 m stereo partner and paints a
world-locked texture. On the mutually visible pixels the mode is within a
fraction of a meter of the fixture depth.

Unprojection of that depth with the capture `R|t` is the frustum step. The
points are in the flight frame. A rigid motion of the whole flight frame
leaves the camera-frame depth unchanged and moves the points with the frame,
because the cost depends only on relative pose.

### Pretraining data

Stage 1 is static multi-view depth, not the 4D rig. Loaders, all of them
returning OpenCV world-to-camera in meters (the flight-frame `R|t`; the OpenCV
cam2world 4×4 is the inverse):

| source | config | what the loader fixes |
| --- | --- | --- |
| tiny fixture | `data=fixture` | already in the flight frame |
| TartanAir V2 | `data=tartanair_v2` | NED + AirSim camera via the V1 conversion. Depth is float32-in-RGBA PNG, meters. |
| AerialMegaDepth | `data=aerial_megadepth` | COLMAP OpenCV w2c, or a `cam2world` npz / `views.json`. Depth treated as meters. |
| MatrixCity aerial | `data=matrixcity` | NeRF axes to OpenCV. Per-block rotation ×100. EXR Z-depth centimeters → meters (`/ 100`, not the official `/ 10000` which lands in units of 100 m). |

`scripts/pretrain_depth.py` runs the depth loop. Its default is `vit_tiny` on
the fixture, on CPU: L1 on the expected depth plus a cross-entropy onto the
nearest plane. Invalid depth (0) is masked. That script is the smoke test, not
a trained model.

### Thin structure

The spec asks for high-density microscopic Gaussians on wires and an upper bound
`τ_max` on their scale, and also for a "lower bound on scale" so Gaussians cannot
become oversized. Those two sentences disagree. A lower bound prevents scales from
getting small, which is the opposite of a microscopic splat and the opposite of
`τ_max`. Phase 3 should implement the upper bound. The lower bound looks like a
wording error.

Feedforward "adaptive density" is also unspecified. Classic 3DGS densifies by
cloning and splitting during optimization. A single forward pass needs a defined
token budget (more tokens where the feature map is high frequency, or a variable-N
decoder). That mechanism is not in the spec. `N` stays a free dimension until
Phase 3 picks one.

### Relighting

Degree-3 SH is 16 coefficients per color. That models low-frequency view-dependent
color. A lens flare from the sun is a high-frequency, view-specific artifact. SH
alone will not isolate it. The relight head has to predict an explicit residual.
Raising the SH degree would also change the 48-channel block and the rasterizer
binding, and it still would not be a flare model.

## Occupancy cache

`aerosplat.geometry.occupancy` fuses depth and capture-time poses into a local
grid around the vehicle (`extent_m`, `voxel_size`). Three states:

| Value | Meaning |
| --- | --- |
| 0 unknown | no ray entered the voxel |
| 1 free | a ray crossed it before the surface |
| 2 occupied | a depth sample landed in it |

Occupied overwrites free. `hit_count` stores how many samples landed in the cell.
The grid is written as a compressed `.npz` under `cache.root`. The filename
includes voxel size, extent, stride, max depth, and the query timestamp.

Depth does not see inside a building. Interior voxels stay unknown. A binary
"occupied vs empty" target that treats unknown as empty will teach the head that
building interiors are free space. The BCE loss has to mask unknown cells.

The spec says occupancy BCE "explicitly penalizes false negatives on thin wires."
Unweighted BCE does not. That needs a positive weight or a focal term. `hit_count`
is stored so a later loss can up-weight cells that were seen by very few pixels.
It is not a wire label. There are no semantic wire annotations in these datasets.

A uniform grid fine enough for a crane cable (centimeters) over a flight corridor
does not fit in memory. The cache is a local crop on purpose. Wire-scale
supervision is an open Phase 5 problem: smaller local crop, a 2D thin-structure
channel, or a different primitive. The default 0.5 m voxels will not represent a
cable. They exist so the cache API and the tests are real.

## Losses (not implemented)

The spec's equations were embedded as images and did not survive text extraction.
What follows is the prose, not a recovered formula. Coefficients are unspecified.
Do not treat this as the "exact formulation" the spec asked to implement until
the equation images are available.

* Photometric: pixel L1 between the rasterized view and the camera image.
* SSIM: combined with L1, as `1 - SSIM`, to keep edges from going soft.
* Temporal: `|| (x_t + v_t * Δt) - x_{t+1} ||` with `v` in m/s and `Δt` in
  seconds. The norm and any per-Gaussian mask are unspecified.
* Occupancy: BCE of voxel logits against the occupied mask, unknown cells
  masked out. A false-negative weight is a separate, unspecified term.
* Sparsity: a penalty on opacity, plus a penalty when a wire-like Gaussian's
  scale exceeds `τ_max`. Which Gaussians count as wire-like is unspecified.

## Rasterizer

Phase 4 binds a differentiable Gaussian rasterizer. The spec names
`diff-gaussian-rasterization`. That package is a source build against CUDA and
is not on PyPI in a usable CPU form. `gsplat` is a maintained alternative with
the same job. The choice is open. `aerosplat.models.rasterizer.GaussianRasterizer`
is the binding site either way. CI installs a CPU PyTorch wheel and does not
compile either extension.

## Phase roadmap

| Phase | Status | What it adds |
| --- | --- | --- |
| 1 Foundation, ingestion, calibration | this tree | schema, sync, Colosseum + TartanAir + fixture, occupancy cache, tensor contract |
| 2 ViT and 2D-to-3D lift | this tree | shared ViT, pose tokens, plane-sweep lift, depth pretrain |
| 3 Splat parameter heads | stubbed | geometry, appearance, dynamics, pack into `[B, N, 62]` |
| 4 Rasterizer and losses | stubbed | CUDA rasterizer, photometric, temporal, sparsity |
| 5 Relight, occupancy, export | occupancy cache only | glare head, occupancy head, ONNX/TensorRT |

Sub-millisecond inference for a ViT over four high-resolution images plus a
rasterizer is not a credible onboard target. A control-rate budget (on the order
of 10–30 Hz, and only after the model exists) is the number worth measuring.
`GET /api/v1/health` in the spec can report that measurement later. The FastAPI
and React workbench are out of scope for Phase 1.

## Spec notes

These are disagreements with the attached spec, not silent edits.

1. Stanford Drone Dataset cannot train this system. Single top-down camera, no
   rig, no depth. TartanAir is the public depth+pose set. Colosseum is how a
   four-camera rig gets recorded. Details in `docs/DATASETS.md`.
2. "Lower bound on scale" contradicts `τ_max` and contradicts microscopic
   Gaussians. Implement the upper bound.
3. Degree-3 SH does not isolate lens flare. The relight head has to be a
   separate residual.
4. Unweighted occupancy BCE does not preferentially punish missed wires.
5. Depth supervision is a surface plus free space along the ray, not a solid
   building. Unknown voxels must be masked.
6. A flight-corridor voxel grid cannot be both cable-thin and memory-bounded.
   The cache is local. Wire resolution is unsolved.
7. `B` is a batch of timesteps, not "the video frame sequence" as one axis.
8. Velocity units were unspecified. They are m/s in the world frame.
9. The loss equations were images in the PDF and are not recoverable from the
   text. The prose is recorded above. Coefficients stay unset.
10. Adaptive density in a feedforward net is not 3DGS densification. The
    mechanism is unspecified.
11. AirSim's camera frame is not OpenCV. Loaders convert. A rasterizer that
    assumes the raw NED camera will be silently wrong.
