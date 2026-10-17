# Architecture

AeroSplat-4D is a feedforward model: one pass over a synchronized multi-camera
rig predicts a 4D Gaussian cloud. Phase 1 is the data contract. Phase 2 is the
shared ViT and the plane-sweep lift that turns it into per-pixel features and
flight-frame anchors. Phase 3 is the geometry, appearance, and dynamics heads,
the fixed-budget wire allocator, and the slot layout the temporal loss indexes.
The CUDA rasterizer binding is guarded. A pure-PyTorch reference splatter
covers CPU tests. The losses, the glare head, and the occupancy head are still
stubs.

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

AirSim, Colosseum, Cosys-AirSim, and TartanAir use NED (X north, Y east, Z down)
and a camera with X forward, Y right, Z down. `aerosplat.geometry.coordinates`
converts that body-mounted rig. Cosys-AirSim `External` cameras are Unreal-world
poses and are rejected. Skipping the conversion makes unprojection disagree
with any OpenCV rasterizer.

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

`N` is the number of Gaussian slots at that timestep. Phase 3 fixes it for a
rig resolution: `N = sum over cameras of (lift_grid_h * lift_grid_w * slots_per_cell)`.
It does not change when a wire appears or disappears. A batch is only valid
when every sample shares that layout. `GaussianCloud.valid` (`[B, N]` bool) and
`GaussianCloud.wire_prob` (`[B, N]` in `[0, 1]`) sit beside the 62. They are
not extra channels. `wire_prob` is what Phase 4 can multiply into `L_sparse`
and what Phase 5 can read for thin structure.

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

The packed values are the tensors `diff-gaussian-rasterization` multiplies.
The CUDA kernel does not apply `exp` or `sigmoid`.

| Field | Stored in the 62 | What the kernel does with it |
| --- | --- | --- |
| xyz | meters, flight frame | `means3D` |
| scale | **post-exp** meters, positive | `R diag(s)^2 R^T`. A log-scale here is wrong. |
| quat | unit wxyz, Phase 3 forces `w >= 0` | real part is `q.x`. The kernel normalizes again. |
| opacity | **post-sigmoid**, in `[0, 1]` | alpha, not a logit |
| SH | raw coefficients, 16×3, not RGB | `rgb = eval(sh, view) + 0.5`. DC uses `SH_C0 = 0.28209…` so `rgb = SH_C0 * sh0 + 0.5` |
| velocity | m/s, world | not read. Slice `packed[..., :59]` before binding. |

`physical_from_raw` is the exp / sigmoid / normalize helper. Phase 3 does not
call it. Wire Gaussians go through `activate_scales` first, which adds the
anisotropic upper cap and the numerical floor, and then `pack_gaussians`.
`GaussianCloud` is the typed pack: activated channels plus `valid`,
`wire_prob`, and the slot index.

`GaussianLayout` in `aerosplat.models.gaussians` is the source of truth for the
slices. Tests check that the names and the slices add up to 62.

## Heads

| Head | Phase | Outputs | Packed into the 62? |
| --- | --- | --- | --- |
| Geometry | 3, implemented | xyz = anchor + tanh offset, log-scale, quat | yes, after activation |
| Appearance | 3, implemented | opacity, SH residual | yes. DC is initialized from the pixel |
| Dynamics | 3, implemented | velocity, zero at init | yes, not read by the rasterizer |
| Relight | 5, stub | glare residual RGB | no; consumes SH |
| Occupancy | 5, stub | local voxel logits | no; supervised by the cache |

`GaussianRasterizer(backend="auto")` uses the CUDA binding when
`diff_gaussian_rasterization` imports and CUDA is available, and the reference
splatter otherwise. The import is `importlib` inside the call. CPU CI never
compiles the extension.

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

## Phase 3

`SplatPredictor` reads a `LiftedGeometry` and returns a `GaussianCloud`.
`FeedforwardSplatter` is the lifter plus that predictor. `build_model` wires
both from the Hydra config.

### Heads

The geometry head's last layer is zero-initialized except for quaternion `w = 1`,
so the residual rotation starts at the identity. XYZ is the lift anchor plus
`tanh(offset) * offset_radius_m`. Extras are additionally spread along the
image-plane tangent of the edge, inside `along_radius_m`. Log-scale is the head
output plus a per-subslot bias: the base slot is isotropic (`base_scale_m`),
each extra starts long on local X and thin on Y and Z.

The appearance head's SH residual is zero-initialized. The predictor adds
`sh_dc = (rgb - 0.5) / SH_C0` from the lift-cell color, which is the degree-0
coefficient the rasterizer turns back into that RGB. Higher bands stay zero
until they train. Opacity starts at `opacity_init` (0.1). Slots with `valid`
false are packed with opacity 0 so a rasterizer that ignores the mask does not
draw them.

The dynamics head is zero-initialized. Velocity is meters per second in the
flight frame, the unit `Δt` in seconds can multiply.

### Slots, and why `L_temp` can index `i`

Andy's loss is

```
L_temp = (1 / N_splats) sum_i || (x_i^(t) + v_i^(t) Δt) - x_i^(t+1) ||_2^2
```

Index `i` has to be the same Gaussian at both times. The predictor does not
re-rank. `i` is the slot `(camera name, lift-cell row, lift-cell col, subslot)`,
packed in that order. Subslot 0 is the base Gaussian on the anchor. Subslots
`1..K` are the extra candidates. `temporal_residual(cloud_t, cloud_tp1, dt)`
returns `x_t + v_t Δt - x_{t+1}` and a weight `valid_t * valid_{t+1}`. It
raises if the camera order or the grid changed. The `x + v Δt` term is the
loss target, not a second matching stage.

This is pixel correspondence, not a tracked world point. A moving camera looks
at a different surface point through the same pixel next frame, so a static
world still has a nonzero residual when velocity is zero. A tracker that
gathered Gaussians into a new order would change `N` and break export. Phase 4
should reduce `L_temp` with the weight this helper returns. Whether ego-motion
should be subtracted before that residual is an open question, below.

### Wires: density and the scale cap

The spec asks for high-density microscopic Gaussians on wires, an upper bound
`τ_max` on their scale, and also a "lower bound on scale" so Gaussians cannot
become oversized. Those sentences disagree with each other and with Andy's
loss. A lower bound stops scales from getting small, which is the opposite of
a microscopic splat. `L_sparse` is

```
L_sparse = sum_i α_i * max(0, s_x,i * s_y,i * s_z,i - τ_max)
```

`τ_max` upper-bounds the **volume** `s_x s_y s_z`, not a single axis, and the
sum is over every Gaussian. The prose around it says the threshold is for
wire-like structures.

Phase 3 implements the upper bound, anisotropically, in the parameterization
rather than as a loss (the loss is Phase 4):

* Every scale is floored at `min_scale_m` (default `1e-4` m, a tenth of a
  millimeter). That is only so the covariance stays invertible. It is far
  below the wire thickness. It is not the spec's lower bound, and it does not
  stop a microscopic splat.
* `wire_prob = 0` caps every axis at `max_scale_m`.
* `wire_prob = 1` caps the largest predicted axis at `wire_long_max_m` and the
  other two at `wire_transverse_max_m`. The volume is then at most
  `wire_transverse_max_m^2 * wire_long_max_m`. A straight-through one-hot
  picks the long axis, so the forward cap is hard and the log-scale logits
  still get gradient. The long axis is local X at initialization, and the
  quaternion slerps toward the rotation that maps local X onto the wire
  tangent in the flight frame. The tangent is the structure-tensor direction,
  fronto-parallel: `(du/fx, dv/fy, 0)` in the camera, then `R^T`.

`wire_prob` is a side output. Andy's formula does not multiply by it. The spec
prose does. Phase 4 has to choose; the probability is there either way.

Adaptive density in one forward pass cannot be 3DGS clone-and-split, and a
variable `N` cannot export. The edge branch is a Sobel / structure-tensor
prior plus a zero-initialized conv residual, run on the image the lifter saw
or on a same-FOV higher-resolution copy (`highres_images`). The ViT is at
224 px; a cable is often one pixel at the sensor, so the prior is computed at
that image's resolution and max-pooled onto the lift cell. The max, not the
cell center, is what fires, and the tangent is taken from the winning pixel.
Padding is replicate, so a constant image does not grow a fake border wire.

Each cell owns `slots_per_cell` slots (default 4: one base, three extras). The
base slot is always valid. An extra is valid when the cell's edge probability
clears `edge_threshold` and the slot is inside the global top-`max_extra_gaussians`.
Top-k writes `valid`. It does not gather. `N` is the same for a blank wall and
for a wire. At 224 px, stride 14, 4 cameras, 4 slots, `N = 4096`. That tensor
is about a megabyte. The ViT is the 8–16 GB limit, not the splat count. Configs
say so.

The hard mask is not differentiable. The edge conv trains through the soft
`wire_prob` (the scale cap and the tangent slerp), which is why the last conv
is zero-initialized but still receives gradient. `anchor_valid` from the lift
is copied onto the cloud and does not, by itself, turn a base slot off.

### Rasterizer

`ReferenceSplatRenderer` is an EWA splatter: project with the full OpenCV `K`,
form the 2D covariance from the Jacobian of the perspective divide, add the
same 0.3 px low-pass the CUDA kernel adds on the diagonal, evaluate SH, and
alpha-composite front to back. It is `O(N * H * W)`. Use it for a handful of
Gaussians on CPU. Do not train with it, and do not point it at 1080p.

`DiffGaussianRasterizationAdapter` builds the transposed world-to-camera and
projection matrices the Inria bindings expect (`viewmatrix = w2c.T`, camera
center `-R^T t`) and calls `GaussianRasterizer` with post-exp scales,
post-sigmoid opacity, wxyz quaternions, and `[N, 16, 3]` SH. That package's
focal length comes from `tan(fov/2)` and the image size, so the principal
point is the image center. The reference path uses `cx, cy`. They match on a
centered camera. Phase 4 should check the first CUDA frame against the
reference before trusting an off-center `K`.

### Open questions for Phase 4 and Phase 5

1. Should `L_sparse` follow Andy's ungated sum, or the spec prose and multiply
   by `wire_prob`? Applying `τ_max` to every Gaussian also penalizes a
   building-sized splat.
2. `τ_max` is unset. The cap's volume at `wire_prob = 1` is
   `wire_transverse_max_m^2 * wire_long_max_m` (defaults `0.03^2 * 1 = 9e-4` m³).
   A loss threshold equal to that cap is zero wherever the cap already binds.
   A slightly smaller `τ_max` still has gradient inside the feasible set.
3. Pixel slots make `i` stable and `N` static. They are not object tracks.
   Under ego-motion, `x_i^(t+1)` is a different surface point. Subtracting the
   camera motion, or supervising velocity as zero on static scenes another
   way, is not decided.
4. `L_temp` on a slot that is invalid at one of the two times: the helper's
   weight is the product of the two masks. A wire that appears at `t+1` then
   contributes nothing. That may or may not be what the sum over `N_splats`
   should do.
5. `highres_images` must be the same frustum as the network image, only finer.
   A sliding crop with its own `K` is not implemented.
6. The base slot stays valid when the plane sweep had no overlap
   (`anchor_valid` false). Phase 4 may want to drop those from the photometric
   term.

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

## Losses (not implemented; Phase 4)

Andy's equations (2026-10-16) are the formulation Phase 4 implements. The PDF
embedded them as images; the text form is authoritative. Coefficients and
`τ_max` are still unset. Phase 3 does not evaluate these. It produces the
packed cloud, the slot index, and `wire_prob`.

```
L_total = λ1 L_photo + λ2 L_ssim + λ3 L_temp + λ4 L_occ + λ5 L_sparse
L_photo = (1 / N_cam) sum_c || Ihat_c - I_c ||_1
L_ssim  = 1 - (1 / N_cam) sum_c SSIM(Ihat_c, I_c)
L_temp  = (1 / N_splats) sum_i || (x_i^(t) + v_i^(t) Δt) - x_i^(t+1) ||_2^2
L_occ   = - sum_{v in V} [ y_v log yhat_v + (1 - y_v) log(1 - yhat_v) ]
L_sparse = sum_i α_i * max(0, s_x,i * s_y,i * s_z,i - τ_max)
```

`L_photo` should be normalized per pixel. `L_temp` needs the slot
correspondence above; feedforward predictions are not an unordered set.
`L_occ` is a sum, so it scales with `|V|`, and it has no wire weight. A
positive weight or a focal term is still recommended, and unknown voxels stay
masked. `L_sparse` penalizes volume. It does not add density on edges. The
edge branch is what adds density. Whether the sum is gated by `wire_prob` is
open question 1 above.

## Rasterizer

Phase 4 trains through `diff-gaussian-rasterization`. That package is a source
build against CUDA and is not on PyPI in a usable CPU form. `gsplat` remains a
possible substitute with the same job; the adapter does not import it. The
binding site is `DiffGaussianRasterizationAdapter`. `GaussianRasterizer` picks
it only when the import succeeds and CUDA is available. CI installs a CPU
PyTorch wheel, does not compile the extension, and renders tests with
`ReferenceSplatRenderer`.

## Phase roadmap

| Phase | Status | What it adds |
| --- | --- | --- |
| 1 Foundation, ingestion, calibration | this tree | schema, sync, Colosseum + Cosys-AirSim + TartanAir + fixture, occupancy cache, tensor contract |
| 2 ViT and 2D-to-3D lift | this tree | shared ViT, pose tokens, plane-sweep lift, depth pretrain |
| 3 Splat parameter heads | this tree | geometry, appearance, dynamics, wire slots, pack into `[B, N, 62]` |
| 4 Rasterizer and losses | adapter stubbed, reference renderer in | CUDA rasterizer training, photometric, temporal, sparsity |
| 5 Relight, occupancy, export | occupancy cache only | glare head, occupancy head, ONNX/TensorRT |

Sub-millisecond inference for a ViT over four high-resolution images plus a
rasterizer is not a credible onboard target. A control-rate budget (on the order
of 10–30 Hz, and only after the model exists) is the number worth measuring.
`GET /api/v1/health` in the spec can report that measurement later. The FastAPI
and React workbench are out of scope for Phase 1.

## Spec notes

These are disagreements with the attached spec, not silent edits.

1. Stanford Drone Dataset cannot train this system. Single top-down camera, no
   rig, no depth. TartanAir is the public depth+pose set. Cosys-AirSim is the
   maintained four-camera recorder. Colosseum is archived and still supported.
   Details in `docs/DATASETS.md`.
2. "Lower bound on scale" contradicts `τ_max` and contradicts microscopic
   Gaussians. Phase 3 caps wire volume anisotropically (two thin axes, one
   long axis) and keeps only a numerical floor. See Phase 3 above.
3. Degree-3 SH does not isolate lens flare. The relight head has to be a
   separate residual.
4. Unweighted occupancy BCE does not preferentially punish missed wires.
5. Depth supervision is a surface plus free space along the ray, not a solid
   building. Unknown voxels must be masked.
6. A flight-corridor voxel grid cannot be both cable-thin and memory-bounded.
   The cache is local. Wire resolution is unsolved.
7. `B` is a batch of timesteps, not "the video frame sequence" as one axis.
8. Velocity units were unspecified. They are m/s in the world frame.
9. The loss equations were images in the PDF. Andy's text form is now the
   formulation Phase 4 implements. It is recorded under Losses. Coefficients
   and `τ_max` stay unset.
10. Adaptive density in a feedforward net is not 3DGS densification. Phase 3
    uses a fixed slot budget, an edge prior at image resolution, and a
    `valid` mask. `N` does not change when a wire appears.
11. AirSim's camera frame is not OpenCV. Loaders convert. A rasterizer that
    assumes the raw NED camera will be silently wrong.
