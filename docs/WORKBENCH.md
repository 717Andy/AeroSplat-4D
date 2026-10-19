# Debugging workbench

The workbench is a separate FastAPI process and a React viewer. It imports
AeroSplat (the Gaussian layout, the trainer, glare, occupancy, and
`query_collision`) and does not change the training loop. A parallel data
generator can keep owning `aerosplat/`, `scripts/`, and `configs/`.

## Run

Python 3.10+, the AeroSplat package, the API extras, and Node 20+.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]" --no-deps
pip install numpy pyyaml omegaconf hydra-core pillow pytest ruff matplotlib onnx onnxruntime
pip install -r server/requirements.txt
python -m server.dev
```

`python -m server.dev` is the same command on macOS, Linux, and Windows
(from the repo root, in the environment where `aerosplat` imports). It writes
`episodes/demo` the first time that directory is missing, runs `npm install`
in `web/` if `node_modules` is absent, then starts:

| Process | URL |
| --- | --- |
| API | http://127.0.0.1:8000 |
| UI | http://127.0.0.1:5173 |

Vite proxies `/api` to the API. Open the UI. The demo episode is already in
the repo, so a fresh clone does not need a simulator or a GPU to scrub the
scene. On Windows, Node must be on `PATH` (`npm.cmd` is selected automatically).

API only:

```bash
python -m server.demo    # rewrite episodes/demo
python -m server         # uvicorn, host and port from server/workbench.yaml
```

Checkpoint and device overrides, after the YAML file:

Windows Command Prompt:

```bat
set AEROSPLAT_CHECKPOINT=outputs\train\last.pt
set AEROSPLAT_MODEL=vit_tiny
set AEROSPLAT_DEVICE=cpu
python -m server
```

Windows PowerShell:

```powershell
$env:AEROSPLAT_CHECKPOINT = "outputs\train\last.pt"
$env:AEROSPLAT_MODEL = "vit_tiny"
$env:AEROSPLAT_DEVICE = "cpu"
python -m server
```

macOS and Linux use `export` for the same three names. `AEROSPLAT_EPISODES`
is an `os.pathsep`-separated list of canonical dataset roots. A checkpoint
whose state dict does not match the model is ignored and the API reports
`init: random` plus `checkpoint_error`.

## What the demo is

`episodes/demo` is the Phase 1 tiny rig (`scripts/generate_fixture.py` /
`aerosplat.data.fixture.generate_fixture`): four cameras, the rig flying east
at 4 m/s, depth and RGB in the canonical layout. The baker then adds three
things the RGB does not contain, so the overlays have something to show
before a trained checkpoint exists:

- A cable from `(0.2, -2, 3.3)` to `(2, 2, 3.0)` with `wire_prob = 1`.
- An orange actor at `(1.0, 1.6, 2.4)` moving at `(1.5, 0, 0)` m/s.
- An occupancy volume and a synthetic glare PNG for the default sun
  `[0.25, 0.12, 0.96]`.

The HUD tag stays `recording` while those baked clouds are on screen. They
are not a network prediction. **Predict glare** posts the current frames to
`POST /api/v1/infer` and replaces the cloud, the volume, and the glare
layers. With no checkpoint that pass is a random-init `vit_tiny`, and the
HUD tag changes to `random init`.

## API

| Method | Path | Role |
| --- | --- | --- |
| `GET` | `/api/v1/health` | GPU name, VRAM used / total, per-stage milliseconds, inference throughput |
| `GET` | `/api/v1/schema` | 62-channel names and the binary magics |
| `GET` | `/api/v1/episodes` | Canonical recordings under `episode_dirs` |
| `GET` | `/api/v1/episodes/{id}/frames/{t}` | Synced cameras, K, world-to-camera `R\|t`, timestamps, cloud URL |
| `GET` | `/api/v1/episodes/{id}/cameras/{cam}/image` | PNG |
| `GET` | `/api/v1/episodes/{id}/cameras/{cam}/glare?index=` | Baked glare PNG |
| `GET` | `/api/v1/episodes/{id}/clouds/{t}` | `AGS1` bytes |
| `GET` | `/api/v1/occupancy` | Sparse `AOC1` plus `query_collision` for a corridor |
| `POST` | `/api/v1/infer` | Synchronized multi-camera batch. JSON or multipart |

`GET /api/v1/occupancy` query parameters: `episode`, `index`, `radius`,
`corridor` (`x,y,z;x,y,z;...`), `min_prob` (default 0.5), `format=json|binary`.
Omit `episode` to read the volume from the latest inference. The demo corridor
is `[[0,0,3.2],[1.2,0,3.2],[2.6,0,3.2]]` with radius 0.6 m, which intersects
the cable.

JSON infer body:

```json
{
  "format": "json",
  "sun_world": [0.25, 0.12, 0.96],
  "cameras": [
    {
      "name": "front",
      "timestamp": 0.0,
      "K": [[80, 0, 40], [0, 80, 30], [0, 0, 1]],
      "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      "t": [0, 0, 0],
      "image_png_base64": "<png or jpeg bytes, optional data-URL prefix>"
    }
  ]
}
```

Multipart uses a `manifest` JSON field (same camera records, without the
image) plus one file field per camera name. `format` may be a query parameter.
JSON responses embed `gaussian_b64`, `occupancy_b64`, and per-camera glare
PNGs. `format=binary` returns `application/vnd.aerosplat.infer`: a little-endian
`uint32` meta length, UTF-8 JSON (no Gaussian payload), then the `AGS1` bytes.

Health throughput is `1000 / last total_ms` for the model, not the viewer frame
rate. A CPU `vit_tiny` pass is not the sub-millisecond figure in the spec.

## Binary layouts

Documented in `server/codec.py` and returned by `GET /api/v1/schema`.

`AGS1` is a 64-byte little-endian header (`magic`, version 1, flags, batch,
count, 62 channels, 16 SH coefficients, timestamp, `dt`, header size, source
sun xyz) followed by `float16[B,N,62]`, `uint8[B,N]` valid, and
`float16[B,N]` wire probability. Channel order is
`GaussianLayout.channel_names()`: xyz, post-exp scale, wxyz quaternion,
opacity, degree-3 SH coefficient-major RGB (`11:59`), velocity m/s (`59:62`).
`valid` and `wire_prob` are side channels.

`AOC1` is a 64-byte header (origin, voxel size, dims, cell count, risk,
nearest distance, wire risk, solid risk, `min_prob`) plus 24-byte cells:
`ijk` uint16, occupancy and wire probability float16, direction and offset
float16, pad. Cells at or below `min_prob` are dropped. A hit is probability
strictly above 0.5. The drawn wire is the cell center plus `offset * voxel`,
extended by `0.5 * voxel * sqrt(3)` along the unit tangent — the same segment
`query_collision` measures. An infinite nearest distance is JSON `null`.

## Viewer

The scene is the flight frame: ENU, Z up. Three.js cameras are Y-up, so the
canvas sets `camera.up = (0, 0, 1)` and the ground grid is rotated onto `z = 0`.
Extrinsics stay OpenCV world-to-camera. Frustums use each camera's K and the
image is drawn on a plane in that camera.

Splats are a custom WebGL2 pass (`RawShaderMaterial`, GLSL 300 es), not
Three.js points:

- One instanced quad per Gaussian. Attributes stay in an `RGBA32F` data
  texture (16 texels per Gaussian) so N can grow toward a million without a
  huge attribute buffer.
- Depth order is a 4096-bin counting sort, far to near, each frame.
- The vertex shader builds the 2D EWA covariance from scale and quaternion,
  adds a 0.3 px low-pass, and draws a quad out to about 3 sigma.
- Degree-3 SH is evaluated in the shader, then `+ 0.5`, matching
  `aerosplat.models.rasterizer.eval_sh`.
- The timeline stores a fractional index. Between frame `t` and `t+1` the
  shader moves `xyz + velocity * alpha * dt`. **Velocity** draws arrows for
  the same motion.
- The sun slider builds a target direction
  `(cos el · cos az, cos el · sin az, sin el)` with azimuth from east toward
  north. The shader multiplies the view direction by `R^T`, where `R` is the
  rotation that takes the baked sun onto the slider sun. That is
  `evaluate_relit_sh`: an environment rotation, not a new lighting solve and
  not a new glare disk. Degree 3 cannot draw a hard solar disk. **Reset sun**
  restores the cloud's source sun. **Predict glare** is the path that asks
  the network for a new glare layer.
- **clean / glare / composite** composite the image-space glare PNG per camera
  (`clamp(clean + glare, 0, 1)` via additive blend). Each camera can follow
  that choice or override it.
- **Occupancy** draws cells above 0.5. **Wires** draws the tangent segments.
  The corridor turns red when `query_collision` reports risk above 0.5.

WebGPU is not implemented. The sort and the data texture are the pieces that
would move.

## Tests

```bash
pytest server/tests
cd web && npm test && npm run typecheck
```

`pytest` alone still runs only `tests/`, so core CI does not need FastAPI.
The workflow installs `server/requirements.txt`, runs `pytest server/tests`,
then `npm ci`, `npm run typecheck`, and `npm test` in `web/`.
