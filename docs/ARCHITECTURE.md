# Architecture

AeroSplat-4D is a feedforward model: one pass over a synchronized multi-camera
rig predicts a 4D Gaussian cloud. Phase 1 is the data contract that later phases
read. The network, the CUDA rasterizer, and the losses are specified here and
stubbed in `aerosplat.models`. They are not implemented.

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

Stubs live in `aerosplat.models.heads` and `aerosplat.models.backbone`. Calling
them raises `NotImplementedError`. `GaussianRasterizer` checks the packed shape
and then raises. It does not import a CUDA extension.

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
| 1 Foundation, ingestion, calibration | this tree | schema, sync, Colosseum + Cosys-AirSim + TartanAir + fixture, occupancy cache, tensor contract |
| 2 ViT and 2D-to-3D lift | stubbed | shared backbone, pose tokens, cost volume |
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
   rig, no depth. TartanAir is the public depth+pose set. Cosys-AirSim is the
   maintained four-camera recorder. Colosseum is archived and still supported.
   Details in `docs/DATASETS.md`.
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
