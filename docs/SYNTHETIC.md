# Synthetic episodes

This generates a training set the existing loaders already understand: RGB,
planar depth, segmentation, `K`, world-to-camera `R|t`, and Unix timestamps,
plus sidecars for actor velocity, wire polylines, the sun vector, and a
flare-off frame. Colosseum is the default simulator. Cosys-AirSim uses the
same script with `--simulator cosys`.

The machine this was written for is an MSI Vector 17 HX: Windows, an NVIDIA
RTX laptop GPU, Unreal Engine installed. You do not need to know Unreal beyond
the clicks below. `docs/BENCHMARK.md` is the install order (which engine
version, which Python client, where `settings.json` goes). Read that once,
then come back here.

Two ways to run:

1. `--mock` writes a real shard with a software renderer. No Unreal. Use this
   first, including on a machine that has no GPU, to prove the files train.
2. Without `--mock`, the same script drives Colosseum or Cosys-AirSim. Play
   has to be running before you start it.

## What one episode contains

`configs/scenario/urban_corridor.yaml` is the checked-in example.

* Map `Blocks`, the empty cube level that ships with the simulator.
* A 4-camera rig from `configs/rig/uam_quad.yaml`, flying 70–90 m above the
  origin (the allowed band is 30–300 m).
* A clock from 14:00Z to 18:00Z on 21 June 2026 at Chicago, so the sun moves.
  `yaw: toward_sun` points the nose, and the front camera, at that sun.
* One other drone, one bird (a sphere), and one ground vehicle, each with its
  own segmentation id.
* A 2 cm feeder and a 3 cm crane cable. Sag is the drop of the midpoint below
  the straight line between the poles, in meters.
* Domain randomization is off. `--randomize` turns it on. Episode `k` uses
  seed `base_seed + k`, so a resumed run repeats the same poses.

Each finished shard is a canonical dataset:

```
data/synthetic/shard_00000_00/
  dataset.json
  dynamics.json          # actor pose and velocity, latitude, longitude
  sun.json               # camera-frame sun vector per frame
  segmentation_ids.json  # the colors actually painted
  settings.json          # copy this into Documents\AirSim before a live run
  episode.json
  DONE
  ep000000/
    wires.json           # exact cable polylines, flight frame
    front/images/000000.png
    front/clean/000000.png          # flare off
    front/depth/000000.npy
    front/segmentation/000000.png
    front/velocity/000000.npy       # [3,H,W] m/s, 0 is background
    front/instance/000000.npy
    front/wire_coverage/000000.npy  # analytic pixel fraction, including sub-pixel cables
```

`image` is the photometric frame (flare on, or a synthetic sun blob when the
simulator cannot toggle lens flare). `clean` is the same view with flare off.
Depth and segmentation come from the flare-off grab.

## 1. Check the pipeline with no simulator

From the repo, in PowerShell or any terminal. This does not start Unreal.

```powershell
python scripts\generate_synthetic.py --scenario configs\scenario\urban_corridor.yaml --output data\synthetic --episodes 1 --frames 4 --mock
```

You should see a disk estimate, then a line like
`shard_00000_00: ... camera-frames/s, glare=synthetic`. The mock client cannot
toggle Unreal's lens flare, so it composites a sun-aligned blob. That blob is
not a measured lens PSF. It exists so the glare loss has a paired target.

`--frames 4` overrides the 20 frames in the YAML. The full rig is 1280×720,
640×480, and two 960×720 cameras. Four frames is enough to train two steps.
A long mock run at that resolution is slow because the stand-in renderer is
pure Python. The live simulator is the path for a real set.

The command is safe to run twice. A shard with a `DONE` file is skipped.
Delete that file, or pass `--no-resume`, to record it again.

`data\synthetic\disk_estimate.json` is an uncompressed upper bound for the
episodes you asked for. PNG compression makes the RGB smaller.
`data\synthetic\throughput.jsonl` is camera-frames per second and rig-frames
per second. `data\synthetic\manifest.json` lists every shard and its seed.

## 2. Train on that shard

Point `data.root` at the shard directory (the folder that contains
`dataset.json`), not at `data\synthetic`.

```powershell
python scripts\train.py data=synthetic data.root=data/synthetic/shard_00000_00 model=vit_tiny model.img_size=32 train.steps=2 train.device=cpu train.backend=reference cache.wire_polylines=data/synthetic/shard_00000_00
```

`model.img_size=32` resizes the rig before the ViT. CPU and `backend=reference`
do not need the CUDA rasterizer. `cache.wire_polylines` stamps the cable
centerlines into the occupancy cache, including the 2 cm wire grid, because a
2 cm cable is usually thinner than a pixel and never hits the depth buffer.

`train.lambda_velocity` defaults to 0. The per-pixel velocity maps are loaded
either way. Turn the term on once you want it in the loss:

```powershell
python scripts\train.py data=synthetic data.root=data/synthetic/shard_00000_00 model=vit_tiny model.img_size=32 train.steps=2 train.device=cpu train.backend=reference train.lambda_velocity=0.1 cache.wire_polylines=data/synthetic/shard_00000_00
```

The shipped corridor has three moving actors. The sample-level `gt_velocity`
vector stays unset in that case, because the loss would broadcast one velocity
onto the buildings as well. The maps are the supervision: background is 0,
each actor is its own velocity. A scenario with exactly one moving actor also
fills `gt_velocity`.

On the RTX, drop `train.device=cpu` and `train.backend=reference` after the
rasterizer is installed (`docs/RASTERIZER.md`), and use a laptop config:

```powershell
python scripts\train.py --config-name experiment/laptop_8gb data=synthetic data.root=data/synthetic/shard_00000_00 cache.wire_polylines=data/synthetic/shard_00000_00
```

## 3. Install the simulator (once)

Follow `docs/BENCHMARK.md` through "Launch the simulator". Short version:

1. Install the NVIDIA driver. In PowerShell, `nvidia-smi` should print the RTX
   and a VRAM size.
2. Install Unreal Engine **5.6** for Colosseum, or **5.8** for Cosys-AirSim,
   from the Epic Games Launcher. Use the version the simulator README names if
   it has moved.
3. Build or install the simulator's Windows Blocks environment. Colosseum is
   the default for this generator. Cosys-AirSim is the maintained simulator
   and is selected with `--simulator cosys`. Do not mix their Python packages:
   Colosseum installs as `airsim`, Cosys as `cosysairsim`.
4. Confirm the client imports:

```powershell
python -c "import airsim; print(airsim.__file__)"
```

For Cosys, that line is `import cosysairsim` instead.

## 4. Generate a small live set

Do these in order. The editor has to be in Play before Python connects.

1. Generate settings without connecting. This writes `settings.json` inside a
   throwaway shard and does not need Play:

```powershell
python scripts\generate_synthetic.py --scenario configs\scenario\urban_corridor.yaml --output data\synthetic_settings --episodes 1 --frames 1 --mock
```

2. Copy `data\synthetic_settings\shard_00000_00\settings.json` to:

```
C:\Users\<your name>\Documents\AirSim\settings.json
```

Create `Documents\AirSim` if it is not there. The file must be in place
**before** you press Play. The RPC cannot resize cameras that were created
from an older file. The generator's `capture.supersample` is baked into this
file as a larger resolution. A live run and this settings file have to use
the same supersample, or the grab size will not match.

3. Open the Blocks project (the `.uproject` from the simulator build) in the
   Unreal Editor. Wait until shaders finish. The first launch on a laptop can
   take several minutes.
4. You should see a floor and some cubes. Press the green **Play** button
   (Alt+P). The view switches to the game. Leave it running. Python talks to
   `127.0.0.1` port **41451**. If Windows asks, allow it on private networks.
5. In a second PowerShell, from the repo, with Play still running:

```powershell
python scripts\generate_synthetic.py --scenario configs\scenario\urban_corridor.yaml --output data\synthetic_live --episodes 1 --frames 4
```

Colosseum is the default, so there is no `--simulator` flag. For Cosys-AirSim,
add `--simulator cosys` and use a settings file produced with that same flag
(step 1 with `--simulator cosys`). A Colosseum settings file will not start a
Cosys session.

The script pauses the simulator, teleports the rig and every actor, steps
`1 / frame_rate_hz` seconds, teleports again so physics cannot drift the
script, then grabs every camera. Stop Play when the script prints the
throughput line. Train the same way as section 2, with
`data.root=data/synthetic_live/shard_00000_00`.

If Python cannot connect: Play is not running (you are still looking at the
editor viewport), `settings.json` is in the wrong folder, or another program
already has port 41451.

`autoload_map` in the YAML is false. Leave it false. Open Blocks yourself and
press Play. `simLoadLevel` restarts the map and is easy to get stuck on.

Spawned actors use the asset names `Cube` and `Sphere`, which Blocks provides.
If a live run raises `DataError` naming an asset, that mesh is not in the
level. Keep `Cube` and `Sphere` until you have imported something else.

## 5. A longer set, and randomization

Twenty frames, one episode, the altitudes and clock in the YAML:

```powershell
python scripts\generate_synthetic.py --scenario configs\scenario\urban_corridor.yaml --output data\synthetic_live --episodes 1
```

Ten episodes, each with its own seed, fog, hour, altitude, speed, and cable
thickness. Thickness stays inside 1–5 cm. Altitude stays inside 30–300 m.

```powershell
python scripts\generate_synthetic.py --scenario configs\scenario\urban_corridor.yaml --output data\synthetic_live --episodes 10 --randomize
```

Shards are `shard_00000_00`, `shard_00001_00`, and so on. `output.shard_frames`
splits a long episode into several canonical datasets
(`shard_00000_00`, `shard_00000_01`, ...). Train on one shard at a time, or
point a later merge at the manifest.

`--episode-start 10 --episodes 10` writes episodes 10 through 19. That is how
you split a batch across machines: each machine gets a different start index
and its own output directory. Seeds do not overlap.

## Thin cables and glare

At 100 m a 2 cm cable is well under one pixel at 720p and 1080p. The hard
rasterizer drops fragments thinner than half a pixel, so DepthPlanar often
misses the cable. Two mitigations are always on:

* `scene/wires.json` is the exact centerline. `cache.wire_polylines` voxelizes
  it. Cells the line crosses are occupied and wire-class, so the occupancy
  loss does not mask them out. The 2 cm grid is the line, not a filled 0.5 m
  cube. Diameter is the mesh thickness and the coverage fraction, not a voxel
  dilation.
* `wire_coverage` stores the analytic fraction of each pixel the cylinder
  covers, including fractions below 1.

`capture.supersample: 2` (or more) asks the simulator for twice the rig
resolution and keeps a wire-colored subpixel when the image is reduced back.
Change it in the YAML, regenerate `settings.json`, and copy that file again
before Play. It does not change the stored training resolution.

Glare tries, in order, a lens-flare RPC, a bloom RPC, an Unreal post-process
settings call, then the console commands `r.DefaultFeature.LensFlare` and
`r.BloomQuality`. If none of those exist, the script composites the synthetic
blob and records `glare_source: synthetic` in `episode.json`. `glare.mode:
simulator` refuses the blob and writes identical pairs when the toggle is
missing. `glare.mode: synthetic` always composites.

The sun vector in `sun.json` is the ephemeris in the OpenCV camera frame
(`X` right, `Y` down, `Z` forward). Positive `Z` means the sun is in front of
the lens. The rendered solar disk can disagree with that vector if AirSim
treats the clock string as local time. The loss uses the ephemeris.

## Cloud GPU machines later

Do this after the laptop path works. The split is by episode, not by sharing
one simulator across shards.

* One simulator process per machine. Do not point two writers at one Play
  session.
* Linux headless rendering uses the simulator's own Docker image and the
  NVIDIA container toolkit. Colosseum and Cosys-AirSim each document that
  image. Use theirs. Do not invent a tag, and do not pass `-nullrhi`: that
  flag produces empty images. `ViewMode=NoDisplay` (or the image's equivalent)
  keeps the GPU rendering without a monitor.
* The RPC port stays **41451**. Publish it to the writer, or run the writer
  in the same network namespace. `--ip` is the host when it is not localhost.
* Copy the shard `settings.json` into the place that image reads
  (`Documents/AirSim/settings.json` inside the container, unless the image
  says otherwise).
* Give each machine `--episode-start` and `--episodes` so seeds and filenames
  do not collide. Collect the shard directories. Each one is already a
  canonical dataset.
* Throughput on the laptop (`throughput.jsonl`) is the number to multiply by
  episode count when you ask whether a cloud batch fits the disk estimate.

City maps (City Sample and most marketplace cities) have their own license.
Blocks is the map to prove the script. Swap `map:` only after Blocks records
a shard you can train.
