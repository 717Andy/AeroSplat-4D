# CUDA rasterizer on a Windows laptop

Phase 4 trains through the stock
[diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
package. It is a CUDA extension. It is not installed by `pip install -e .`,
and CI does not compile it. CPU runs, including `pytest`, use
`ReferenceSplatRenderer`.

The machine this was written for is an MSI Vector 17 HX: Windows, an NVIDIA
RTX laptop GPU, 8 GB or 16 GB. The same steps apply to any Windows + CUDA box.

## Principal point, and why there is no fork

The Inria kernel has no `cx, cy` argument. It turns NDC into a pixel with

```
((ndc + 1) * size - 1) / 2
```

and it recomputes the covariance focal length as `size / (2 * tanfov)`.

`aerosplat.models.rasterizer.diff_gaussian_camera_tensors` builds the
projection so that formula equals the OpenCV pixel `fx * x/z + cx` (and the
same for `y`). `tanfov` is `(size / 2) / f`, so the kernel's focal length
stays `fx` / `fy`. The 2D covariance Jacobian does not depend on the principal
point, so a fork that threads `cx, cy` into the CUDA settings is not required.

Two limits remain:

* The kernel clamps the point used for that Jacobian to `1.3 * tanfov` around
  the optical axis. A principal point far from the image center can still
  clip Gaussians near the far edge. This rig's `K` stays near the center.
  `prepare_rig` pads the bottom and the right without changing `K`, which is
  a half-pad shift, not a crop.
* `means2D` is an auxiliary tensor the backward writes screen-space gradients
  into. World gradients come back through `means3D`, then through the packed
  `[B, N, 62]` (velocity is not passed) into the ViT.

`kernel_pixels` applies the same matrix and `ndc2Pix` on CPU. The test
`test_cuda_raster_matches_the_reference_on_a_tiny_scene` compares a CUDA
render and its gradients to the reference splatter. It skips when CUDA or the
extension is missing, which is every CI run.

## Install

Do this in the environment you will train in. A CPU torch wheel and this
extension cannot share a process.

1. Install a current NVIDIA driver. In PowerShell, `nvidia-smi` should print
   the GPU name and a VRAM number (8 GB or 16 GB on this laptop).
2. Install **Visual Studio 2022 Build Tools** with the **Desktop development
   with C++** workload. The compiler `cl.exe` has to be on `PATH`. The
   "x64 Native Tools Command Prompt for VS 2022" already has it. A normal
   PowerShell often does not, and the extension build then fails with
   `cl.exe not found`.
3. Install a **CUDA Toolkit** whose major version matches the PyTorch wheel
   you are about to install. Check with `nvcc --version`. Set `CUDA_HOME` to
   that toolkit (for example `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`)
   if the installer did not.
4. Install a CUDA build of PyTorch, not the CPU index used in CI. Match the
   wheel to the toolkit. For a CUDA 12.4 toolkit:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`torch.cuda.is_available()` must be `True` before the next step. If it is
false, stop. The extension will compile against whatever torch is installed
and then fail at import.

5. Install AeroSplat without letting pip replace that torch wheel:

```powershell
pip install -e . --no-deps
pip install numpy pyyaml omegaconf hydra-core pillow matplotlib tensorboard
```

6. Build the rasterizer from source. `--recursive` fetches the `glm`
   submodule. Without it the compile stops in `third_party/glm`.

```powershell
git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization
cd diff-gaussian-rasterization
pip install -e . --no-build-isolation
python -c "import diff_gaussian_rasterization; print('rasterizer ok')"
```

`--no-build-isolation` matters. An isolated build can pick up a different
torch than the one you just installed and then the `.pyd` will not import.

If `git clone` was not recursive:

```powershell
git submodule update --init --recursive
```

## What usually breaks

| Symptom | Cause |
| --- | --- |
| `cl.exe` is not recognized | The shell is not the VS x64 native prompt. |
| `CUDA_HOME` / `nvcc` not found | Toolkit not installed, or the variable points at an older CUDA. |
| Import error about a missing DLL or a torch mismatch | The extension was compiled against a different torch. Reinstall torch, delete `build/`, run `pip install -e . --no-build-isolation` again. |
| `glm` header missing | Clone was not `--recursive`. |
| Training falls back to the CPU splatter and then runs out of memory | The import failed quietly and `backend=auto` chose the reference renderer. `backend=diff_gaussian` raises instead of falling back. The 8 GB and 16 GB recipes use `auto` and assume step 6 succeeded. Do not train 224 px with the reference renderer. It is `O(N * H * W)`. |

The adapter accepts a fork that adds an `antialiasing` flag and forces it off.
It does not look for a `cx` / `cy` field. A fork is optional, not the path
this repo trains.

## Check the binding

From the repo, after the extension imports:

```powershell
pytest tests/test_phase4_train.py::test_cuda_raster_matches_the_reference_on_a_tiny_scene -q
```

On a machine without the extension that test is skipped. That is the CI behavior.
