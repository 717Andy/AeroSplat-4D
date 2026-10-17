# Datasets

AeroSplat needs, at every timestep, N calibrated cameras (images, K, world-to-camera
R|t, timestamps, and depth when occupancy is supervised). The cameras may run at
different rates and different resolutions. A single top-down video does not satisfy
that, no matter how much traffic it contains.

## What we use

| Source | Role | Cameras | Depth + pose |
| --- | --- | --- | --- |
| Synthetic fixture (`fixtures/tiny_rig`) | Tests and the sanity script. No download. | 4 (front, down, port, starboard), unequal rates and sizes | Yes, rendered to match K and R\|t |
| Colosseum / AirSim | Primary capture path for a real 4-camera UAM rig | Whatever `settings.json` mounts; the checked-in rig is 4 | Yes, if DepthPlanar is recorded |
| TartanAir | Public multi-view set with depth and pose, same simulator family | Stereo left/right, forward-facing | Yes |

## Why TartanAir, not Mid-Air

Mid-Air is a reasonable low-altitude synthetic set (weather, IMU, depth), and it is
closer to outdoor UAM lighting than many indoor AirSim maps. The public release is a
single forward camera. It cannot supervise epipolar matching between front and side
views, which is the geometry Phase 2 has to learn.

TartanAir was rendered in AirSim. Poses are NED, the camera frame is X-forward /
Y-right / Z-down, depth is planar Z in meters, and the pose line is
`tx ty tz qx qy qz qw` (xyzw). That is the same convention Colosseum uses, so one
conversion (`aerosplat.geometry.coordinates`) serves both adapters. The stereo baseline
is 0.25 m to the right of the left camera. Stereo is not a four-camera rig; it is the
public data that actually has multi-view geometry. The four-camera case is the fixture
and a Colosseum capture.

TartanAir has no capture clock. Frame `i` is assigned `t = i / fps` (default 10 Hz).
Both eyes share that timestamp, so the sync policy is `nearest` with a tiny tolerance.

### Layout the adapter reads

```
<root>/<env>/<Easy|Hard>/<Pxxx>/
  image_left/*.png
  image_right/*.png
  depth_left/*.npy
  depth_right/*.npy
  pose_left.txt
  pose_right.txt          # optional; otherwise the 0.25 m baseline is applied
```

`root` may be a single trajectory directory (the folder that contains `pose_left.txt`)
or any parent of one. Frame ids are the integer in the filename. Image and depth are
paired by that id, then aligned with pose line order.

Default intrinsics, for the official 640×480 frames, are `fx = fy = 320`, `cx = 320`,
`cy = 240`. Other resolutions scale that K. Pixel coordinates are column and row
indices, not `u + 0.5`.

### Fetch

The files are large. A single Easy trajectory with RGB, depth, and pose is enough to
exercise the adapter. The downloader lives in
[tartanair_tools](https://github.com/castacks/tartanair_tools); Azure URLs move, so
follow that repo rather than a copied link.

```bash
# after the official downloader has produced the tree above
python scripts/sanity_alignment.py data=tartanair data.root=/path/to/P000 --index 0 --save outputs/tartanair
python scripts/prepare_cache.py data=tartanair data.root=/path/to/P000 --limit 2
```

`depth_kind` defaults to `planar_z`. If a future dump is Euclidean range, set
`data.depth_kind=euclidean` in the override. The dataset converts it to planar Z on load.

## Colosseum / AirSim

Upstream AirSim is archived. [Colosseum](https://github.com/CodexLabsLLC/Colosseum)
is the maintained fork and speaks the same RPC and `settings.json` schema. The Python
client is imported as `airsim` or `colosseum`, whichever is installed. It is not a
dependency of this package and CI does not install it.

### Rig

`configs/rig/uam_quad.yaml` is the mount description in the vehicle body frame
(X forward, Y right, Z down, degrees):

| Camera | Yaw | Pitch | Role |
| --- | --- | --- | --- |
| front | 0 | 0 | along the nose |
| down | 0 | -90 | nadir |
| port | -90 | 0 | left |
| starboard | +90 | 0 | right |

Positive pitch looks up. Positive yaw looks to the vehicle's right. Resolutions differ
on purpose (front 1280×720, down 640×480, sides 960×720). Horizontal FOV is 90° and
pixels are square, which is how AirSim builds K:

```
fx = fy = (W / 2) / tan(fov / 2)
cx, cy = W / 2, H / 2
```

Checked-in simulator config: `configs/rig/colosseum_settings.json`. Regenerate it with:

```bash
python scripts/capture_colosseum_rig.py --dry-run --output outputs/colosseum_settings
```

Copy that `settings.json` to the simulator's settings path before launching.

### Record

```bash
python scripts/capture_colosseum_rig.py --output data/colosseum/flight --frames 100
```

Each frame stores RGB and DepthPlanar (ImageType 1, camera-frame Z in meters). The
writer converts the vehicle pose plus the mount into ENU/OpenCV world-to-camera
extrinsics and writes a canonical recording (`dataset.json`). Load it with
`data=airsim` (that config's `source` is `canonical`).

### Legacy `airsim_rec.txt`

`data.source=airsim_raw` parses the tab- or comma-separated log:

```
VehicleName  TimeStamp  POS_X POS_Y POS_Z  Q_W Q_X Q_Y Q_Z  ImageFile
```

`ImageFile` is `;`-separated paths in `settings.json` camera order. `TimeStamp` is
nanoseconds unless `data.airsim_raw.timestamp_unit` is set. Depth is picked up beside
each RGB frame (`<stem>.pfm`, `<stem>_depth.npy`, or `depth/<stem>.npy`). Pose in the
log is the vehicle in NED, quaternion wxyz. The live Python API stores the quaternion
as xyzw fields (`w_val` is the scalar); the capture client swaps that before the
conversion. Do not mix the two orders.

## Canonical recording

Everything the training loop sees is a `MultiCameraSample`, regardless of source.
The on-disk form of a recording we write ourselves:

```
root/dataset.json
root/<scene>/<camera>/frames.jsonl
root/<scene>/<camera>/images/000000.png
root/<scene>/<camera>/depth/000000.npy
```

`frames.jsonl` holds `timestamp` (seconds), `K`, and world-to-camera `R`, `t` in the
flight frame. See `docs/ARCHITECTURE.md` for that frame.

## Geometry pretraining (Phase 2)

These three are not the 4-camera training rig. They supervise the ViT and the
plane sweep: static views, depth, OpenCV poses in meters. Each loader returns
the same `MultiCameraSample` the fixture does. `R` and `t` are world-to-camera.
`opencv_cam2world_matrix` is the 4×4 inverse, with the camera center in the
translation column.

```bash
python scripts/pretrain_depth.py
python scripts/pretrain_depth.py data=tartanair_v2 data.root=/path/to/P000 model=vit_tiny pretrain.steps=2
```

The first command is the CPU smoke loop on `fixtures/tiny_rig`. A real run
points `data.root` at a downloaded subset and selects `model=vit_small` (8 GB)
or `model=vit_base` (16 GB).

### AerialMegaDepth

Non-commercial. Access is an approval form on Hugging Face; the loader does not
download anything. A scene is either `views.json` (image, depth, `K`, OpenCV
`cam2world`), a COLMAP `cameras.txt` + `images.txt` beside `images/` and
`depths/`, or per-image `.npz` files with `intrinsics` and `cam2world`. Depth
`.h5` (key `depth`) needs `h5py` and is treated as meters. Views are one
camera per frame, so `views_per_sample` chunks filename order into a virtual
rig. That is not a synchronized mount.

### TartanAir V2

CC BY 4.0. Twelve cameras per tick: `lcam_*` and `rcam_*` for
front/back/left/right/top/bottom. 640×640, 90° FOV, `fx = fy = cx = cy = 320`,
10 Hz. Depth is float32 meters packed into an RGBA PNG. Poses are the V1 line
`tx ty tz qx qy qz qw` in NED. The loader applies the V1 AirSim-camera
conversion. The V2 axis convention has not been checked with a reprojection on
a real trajectory; do that before trusting a trained model.

`customize()` in the official toolkit cannot move the cameras to a wide
baseline. Use the recorded cameras as they are.

### MatrixCity aerial

CC BY-NC 4.0. `transforms.json` is a NeRF cam2world. In the per-block files the
translation is meters and the 3×3 rotation must be multiplied by 100
(`rotation_fix: auto` does that when the determinant is tiny, and leaves a real
rotation alone). The paper `pose/` splits are already normalized and store
translation in units of 100 m: set `rotation_fix=none` and
`translation_scale=100`.

Depth is float32 EXR, Z-depth in centimeters. The loader multiplies by
`depth_scale: 0.01`. The official script divides by 10,000 because it works in
units of 100 m; do not copy that if you want meters. Use the float32 EXR, not
float16, which clips at 655.04 m. The city frame stays Z-up. It is not
georeferenced into ENU. Consecutive frames are grouped with `views_per_sample`
(the aerial capture is four yaw directions).

## Stanford Drone Dataset

Not supported. The Stanford Drone Dataset (also called the Stanford aerial / campus
drone set) is a single overhead camera per scene, with 2D tracks of people and
vehicles. It has no second view, no calibrated extrinsics, and no depth. It cannot
supervise multi-view lifting or an occupancy grid. Ground-traffic dynamics from those
tracks could be a later, separate signal. They are not a substitute for the rig, and
there is no adapter for them.

## Synchronization

Reference timestamps come from one camera (`front` on the UAM rig, `left` on
TartanAir). For every other camera:

* `nearest` uses the closest frame whose `|dt|` is within `max_image_dt`. Image,
  depth, and pose all come from that frame.
* `interpolate_pose` uses that same image and depth, keeps the capture-time pose on
  `R` and `t` (so unprojection stays consistent), and stores a pose at the query
  time on `R_query` and `t_query`. Interpolation is on the camera center and on the
  quaternion. Outside the pose stream the pose is clamped, not extrapolated.

Samples that are missing a required camera are dropped when `require_all_cameras` is
true. The fixture's downward camera is 5 Hz against a 10 Hz nose camera, and the port
camera is offset by 20 ms, so both policies are covered by `fixtures/tiny_rig`.
