"""Rasterizer boundary for the packed Gaussian tensor.

Two backends share one call::

    images = rasterizer(packed, cameras, valid=mask)

* ``reference`` is a pure-PyTorch EWA splatter. It runs on CPU. It is
  O(N * H * W) and exists so tests and Phase 4's loss wiring can render a
  handful of Gaussians without a CUDA toolchain. It is not the training
  rasterizer.
* ``diff_gaussian`` binds ``diff_gaussian_rasterization`` when that extension
  imports and CUDA is available. The import is deferred through
  :func:`importlib.import_module`. Importing this module never loads it, and
  CPU CI does not compile it.

``auto`` uses the CUDA binding when it can and the reference renderer
otherwise. Both consume activated parameters: post-exp scales, post-sigmoid
opacity, unit wxyz quaternions, and raw degree-3 SH. See
``aerosplat.models.gaussians``.

The stock CUDA kernel has no ``cx, cy`` argument. It turns NDC into pixels
with ``((ndc + 1) * size - 1) / 2`` and derives the covariance focal length
from ``tan(fov/2)`` and the image size. This module does **not** fork the
extension. :func:`diff_gaussian_camera_tensors` builds a projection whose NDC,
run through that exact ``ndc2Pix``, lands on the OpenCV pixel
``f * x/z + c``. ``tanfov`` is ``(size / 2) / f``, so the kernel's focal
length stays ``fx`` / ``fy``. The covariance Jacobian does not depend on the
principal point. The symmetric FOV clamp inside the kernel
(``1.3 * tanfov`` about the optical axis) is unchanged; a principal point far
from the image center can still clip that Jacobian. See ``docs/RASTERIZER.md``.
"""

from __future__ import annotations

import importlib
import inspect

import torch
import torch.nn as nn
from torch import Tensor

from aerosplat.models.gaussians import SH_C0, quat_to_matrix, rasterizer_view, unpack_gaussians

# Real SH coefficients through degree 3, in the order the Inria kernel evaluates
# (DC, then the three linear terms, then the five quadratic, then the seven cubic).
_C1 = 0.4886025119029199
_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def load_diff_gaussian_rasterization():
    """Return the CUDA extension, or ``None`` when it is not installed.

    The lookup is deferred so importing :mod:`aerosplat.models.rasterizer` does
    not require a compiler or a GPU.
    """
    try:
        return importlib.import_module("diff_gaussian_rasterization")
    except ImportError:
        return None


def eval_sh(sh: Tensor, dirs: Tensor, degree: int = 3) -> Tensor:
    """Evaluate degree-``degree`` SH. ``sh`` is ``[..., 16, 3]``, ``dirs`` ``[..., 3]``.

    The return value does **not** include the ``+0.5`` the rasterizer adds.
    ``rgb = eval_sh(...) + 0.5``.
    """
    if degree < 0 or degree > 3:
        raise ValueError(f"degree must be in 0..3, got {degree}")
    coeff = sh.transpose(-1, -2)
    result = SH_C0 * coeff[..., 0]
    if degree == 0:
        return result
    x = dirs[..., 0:1]
    y = dirs[..., 1:2]
    z = dirs[..., 2:3]
    result = result - _C1 * y * coeff[..., 1] + _C1 * z * coeff[..., 2] - _C1 * x * coeff[..., 3]
    if degree == 1:
        return result
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    result = (
        result
        + _C2[0] * xy * coeff[..., 4]
        + _C2[1] * yz * coeff[..., 5]
        + _C2[2] * (2.0 * zz - xx - yy) * coeff[..., 6]
        + _C2[3] * xz * coeff[..., 7]
        + _C2[4] * (xx - yy) * coeff[..., 8]
    )
    if degree == 2:
        return result
    result = (
        result
        + _C3[0] * y * (3.0 * xx - yy) * coeff[..., 9]
        + _C3[1] * xy * z * coeff[..., 10]
        + _C3[2] * y * (4.0 * zz - xx - yy) * coeff[..., 11]
        + _C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * coeff[..., 12]
        + _C3[4] * x * (4.0 * zz - xx - yy) * coeff[..., 13]
        + _C3[5] * z * (xx - yy) * coeff[..., 14]
        + _C3[6] * x * (xx - 3.0 * yy) * coeff[..., 15]
    )
    return result


def projection_matrix(
    znear: float,
    zfar: float,
    fx: Tensor | float,
    fy: Tensor | float,
    cx: Tensor | float,
    cy: Tensor | float,
    height: int,
    width: int,
) -> Tensor:
    """Row-major projection aligned with the stock kernel's ``ndc2Pix``.

    ``diff-gaussian-rasterization`` maps NDC to a pixel with
    ``((ndc + 1) * size - 1) / 2`` and has no principal-point argument. The
    matrix below is the unique scale-and-shift that makes that formula equal
    the OpenCV pixel ``f * (x/z) + c``:

        ndc_x = (2 fx / W) * (x/z) + (2 cx - W + 1) / W

    A centered kernel (principal point ``(W - 1) / 2``) is the special case
    ``cx = (W - 1) / 2``, which zeros the off-diagonal. ``cx = W / 2`` is half
    a pixel to the right of that, and the shift term is ``1 / W``.

    ``fx``, ``fy``, ``cx``, ``cy`` are scalars. The matrix is built on
    ``fx``'s device when ``fx`` is a tensor, so an off-center ``K`` stays on
    the same CUDA device as the cloud.
    """
    if not 0 < znear < zfar:
        raise ValueError(f"need 0 < znear < zfar, got {znear}, {zfar}")
    if height < 1 or width < 1:
        raise ValueError(f"image size must be positive, got {height}x{width}")
    fx_t = torch.as_tensor(fx)
    matrix = fx_t.new_zeros(4, 4)
    width_t = fx_t.new_tensor(float(width))
    height_t = fx_t.new_tensor(float(height))
    matrix[0, 0] = 2.0 * fx_t / width_t
    matrix[1, 1] = 2.0 * torch.as_tensor(fy, device=fx_t.device, dtype=fx_t.dtype) / height_t
    matrix[0, 2] = (
        2.0 * torch.as_tensor(cx, device=fx_t.device, dtype=fx_t.dtype) - width_t + 1.0
    ) / width_t
    matrix[1, 2] = (
        2.0 * torch.as_tensor(cy, device=fx_t.device, dtype=fx_t.dtype) - height_t + 1.0
    ) / height_t
    matrix[3, 2] = 1.0
    matrix[2, 2] = zfar / (zfar - znear)
    matrix[2, 3] = -(zfar * znear) / (zfar - znear)
    return matrix


def ndc_to_pixel(ndc: Tensor, size: int) -> Tensor:
    """Stock CUDA ``ndc2Pix``: ``((ndc + 1) * size - 1) / 2``."""
    return ((ndc + 1.0) * float(size) - 1.0) * 0.5


def diff_gaussian_camera_tensors(
    K: Tensor,
    R: Tensor,
    t: Tensor,
    height: int,
    width: int,
    *,
    znear: float = 0.01,
    zfar: float = 100.0,
) -> dict[str, Tensor | float]:
    """Build the matrices one CUDA rasterizer call expects for a single camera.

    ``R``, ``t`` are world-to-camera, ``X_cam = R @ X_world + t``. The kernel
    reads column-major GLM matrices, so the torch tensors are the transposes of
    the row-major w2c and projection. ``campos`` is the camera center in the
    flight frame, ``-R^T @ t``.
    """
    if K.ndim != 2 or R.ndim != 2 or t.ndim != 1:
        raise ValueError("pass one camera, not a batch")
    if height < 1 or width < 1:
        raise ValueError(f"image size must be positive, got {height}x{width}")
    w2c = K.new_zeros(4, 4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    w2c[3, 3] = 1
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    if float(fx) <= 0 or float(fy) <= 0:
        raise ValueError(f"focal length must be positive, got fx={float(fx)} fy={float(fy)}")
    # The kernel recomputes focal = size / (2 * tanfov). This choice makes that
    # focal equal fx / fy. It is intentionally not the half-angle to the image
    # corner when the principal point is off center.
    tanfovx = float((0.5 * width) / fx)
    tanfovy = float((0.5 * height) / fy)
    proj = projection_matrix(znear, zfar, fx, fy, cx, cy, height, width)
    view = w2c.transpose(0, 1).contiguous()
    full = (view @ proj.transpose(0, 1)).contiguous()
    campos = torch.linalg.inv(view)[3, :3].contiguous()
    return {
        "viewmatrix": view,
        "projmatrix": full,
        "projection": proj,
        "w2c": w2c,
        "campos": campos,
        "tanfovx": tanfovx,
        "tanfovy": tanfovy,
    }


def kernel_pixels(
    xyz: Tensor,
    K: Tensor,
    R: Tensor,
    t: Tensor,
    height: int,
    width: int,
    *,
    znear: float = 0.01,
    zfar: float = 100.0,
) -> Tensor:
    """Pixels the stock CUDA kernel assigns to world points ``xyz`` ``[N, 3]``.

    Applies the same projection matrix and ``ndc2Pix`` as
    ``diff-gaussian-rasterization``. CPU tests compare this with the OpenCV
    ``u = fx x/z + cx`` formula; a CUDA render is only trustworthy where the
    two agree.
    """
    if xyz.ndim == 1:
        pixels = kernel_pixels(xyz.unsqueeze(0), K, R, t, height, width, znear=znear, zfar=zfar)
        return pixels[0]
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must be [N, 3], got {tuple(xyz.shape)}")
    mats = diff_gaussian_camera_tensors(K, R, t, height, width, znear=znear, zfar=zfar)
    full = mats["projection"] @ mats["w2c"]
    ones = xyz.new_ones(xyz.shape[0], 1)
    clip = torch.cat((xyz, ones), dim=-1) @ full.transpose(0, 1)
    ndc = clip[:, :2] / clip[:, 3:4]
    return torch.stack(
        (ndc_to_pixel(ndc[:, 0], width), ndc_to_pixel(ndc[:, 1], height)),
        dim=-1,
    )


class ReferenceSplatRenderer(nn.Module):
    """Depth-sorted EWA splatter in PyTorch.

    Projects each Gaussian with the OpenCV ``K`` and world-to-camera ``R|t``,
    builds the 2D covariance the same way the CUDA kernel does (Jacobian of
    the perspective divide, plus a 0.3 pixel low-pass on the diagonal), and
    alpha-composites front to back. SH uses :func:`eval_sh` plus ``0.5``.

    Images come back as ``[B, 3, H, W]``. Gaussians behind the camera, or
    marked invalid, contribute nothing. The sort is not differentiable; the
    colors and the per-pixel falloff are.
    """

    def __init__(self, low_pass: float = 0.3, sh_degree: int = 3) -> None:
        super().__init__()
        self.low_pass = low_pass
        self.sh_degree = sh_degree

    def forward(
        self,
        packed: Tensor,
        K: Tensor,
        R: Tensor,
        t: Tensor,
        height: int,
        width: int,
        valid: Tensor | None = None,
        background: Tensor | None = None,
    ) -> Tensor:
        rasterizer_view(packed)
        batch, count, _ = packed.shape
        if background is None:
            background = packed.new_zeros(3)
        else:
            background = background.to(device=packed.device, dtype=packed.dtype)
        image = background.view(1, 3, 1, 1).expand(batch, 3, height, width).clone()
        if count == 0 or height == 0 or width == 0:
            return image
        parts = unpack_gaussians(packed)
        xyz_cam = torch.einsum("bij,bnj->bni", R, parts["xyz"]) + t[:, None, :]
        depth = xyz_cam[..., 2]
        fx = K[:, 0, 0][:, None]
        fy = K[:, 1, 1][:, None]
        cx = K[:, 0, 2][:, None]
        cy = K[:, 1, 2][:, None]
        safe_z = torch.where(depth.abs() < 1e-6, torch.ones_like(depth), depth)
        pixel_u = fx * xyz_cam[..., 0] / safe_z + cx
        pixel_v = fy * xyz_cam[..., 1] / safe_z + cy
        sigma_2d = _screen_covariance(parts["quat"], parts["scale"], R, xyz_cam, K, self.low_pass)
        alpha = _alpha_maps(
            parts["opacity"],
            sigma_2d,
            pixel_u,
            pixel_v,
            depth,
            height,
            width,
            valid,
        )
        rgb = _view_colors(parts["xyz"], parts["sh"], R, t, self.sh_degree)
        order = depth.masked_fill(~(depth > 0.05), 1e6).argsort(dim=1, stable=True)
        acc = packed.new_zeros(batch, height, width, 3)
        transmittance = packed.new_ones(batch, height, width)
        batch_index = torch.arange(batch, device=packed.device)
        for slot in range(count):
            which = order[:, slot]
            alpha_i = alpha[batch_index, which]
            color_i = rgb[batch_index, which]
            acc = acc + transmittance[..., None] * alpha_i[..., None] * color_i[:, None, None, :]
            transmittance = transmittance * (1.0 - alpha_i)
        rendered = acc.permute(0, 3, 1, 2).contiguous()
        rendered = rendered + transmittance[:, None] * background.view(1, 3, 1, 1)
        return rendered


class DiffGaussianRasterizationAdapter(nn.Module):
    """Map a packed cloud onto ``diff_gaussian_rasterization``.

    Calling this without the extension, or without CUDA, raises
    :class:`RuntimeError`. The reference renderer is the CPU path. Training
    goes through this adapter. Velocity is not passed: the kernel arguments
    are the Inria names ``means3D``, ``means2D``, ``opacities``, ``shs``,
    ``scales``, ``rotations``, taken from the first 59 channels.

    The forward runs in fp32 with autocast off. The extension's backward is
    not reliable in fp16, and AMP is applied to the ViT around this call.
    ``means2D`` is an auxiliary leaf the kernel writes screen-space gradients
    into; those are not part of the 62-vector. World-space gradients return
    through ``means3D`` and therefore through the packed cloud into the ViT.
    """

    def __init__(self, sh_degree: int = 3, znear: float = 0.01, zfar: float = 100.0) -> None:
        super().__init__()
        self.sh_degree = sh_degree
        self.znear = znear
        self.zfar = zfar

    def available(self) -> bool:
        return load_diff_gaussian_rasterization() is not None and torch.cuda.is_available()

    def forward(
        self,
        packed: Tensor,
        cameras: dict[str, dict[str, Tensor]],
        valid: Tensor | None = None,
        images: dict[str, Tensor] | None = None,
        background: Tensor | None = None,
    ) -> dict[str, Tensor]:
        module = load_diff_gaussian_rasterization()
        if module is None or not torch.cuda.is_available():
            raise RuntimeError(
                "diff_gaussian_rasterization is not importable or CUDA is unavailable. "
                "Use GaussianRasterizer(backend='reference') on CPU. "
                "Phase 4 training binds this adapter; CPU CI does not compile it."
            )
        if packed.device.type != "cuda":
            raise RuntimeError("diff-gaussian-rasterization requires CUDA tensors")
        # The CUDA backward is fp32. A ViT autocast must not cast the cloud.
        with torch.autocast(device_type="cuda", enabled=False):
            packed = packed.float()
            rasterizer_view(packed)
            if background is None:
                background = packed.new_zeros(3)
            else:
                background = background.to(device=packed.device, dtype=torch.float32)
            parts = unpack_gaussians(packed)
            outputs: dict[str, Tensor] = {}
            batch, count, _ = packed.shape
            for name, camera in cameras.items():
                height, width = _camera_hw(camera, None if images is None else images.get(name))
                frames = []
                for index in range(batch):
                    if valid is None:
                        mask = torch.ones(count, dtype=torch.bool, device=packed.device)
                    else:
                        mask = valid[index]
                    if int(mask.sum()) == 0:
                        frames.append(background.view(3, 1, 1).expand(3, height, width).clone())
                        continue
                    matrices = diff_gaussian_camera_tensors(
                        camera["K"][index].float(),
                        camera["R"][index].float(),
                        camera["t"][index].float(),
                        height,
                        width,
                        znear=self.znear,
                        zfar=self.zfar,
                    )
                    settings = _raster_settings(
                        module,
                        image_height=height,
                        image_width=width,
                        tanfovx=matrices["tanfovx"],
                        tanfovy=matrices["tanfovy"],
                        bg=background,
                        scale_modifier=1.0,
                        viewmatrix=matrices["viewmatrix"],
                        projmatrix=matrices["projmatrix"],
                        sh_degree=self.sh_degree,
                        campos=matrices["campos"],
                        prefiltered=False,
                        debug=False,
                        antialiasing=False,
                    )
                    means = parts["xyz"][index, mask].contiguous()
                    # Screen-space auxiliary. The kernel's backward writes it;
                    # world gradients travel through means3D.
                    means_2d = torch.zeros_like(means, requires_grad=True)
                    try:
                        means_2d.retain_grad()
                    except RuntimeError:
                        pass
                    rasterizer = module.GaussianRasterizer(raster_settings=settings)
                    rendered, _radii = rasterizer(
                        means3D=means,
                        means2D=means_2d,
                        opacities=parts["opacity"][index, mask].contiguous(),
                        shs=parts["sh"][index, mask].contiguous(),
                        scales=parts["scale"][index, mask].contiguous(),
                        rotations=parts["quat"][index, mask].contiguous(),
                    )
                    frames.append(rendered)
                outputs[name] = torch.stack(frames, dim=0)
            return outputs


class GaussianRasterizer(nn.Module):
    """Render every camera in a rig from one packed cloud.

    ``cameras`` maps a name to ``K`` ``[B, 3, 3]``, ``R`` ``[B, 3, 3]``,
    ``t`` ``[B, 3]``, plus ``height`` and ``width`` when ``images`` is not
    passed. The return value maps that name to ``[B, 3, H, W]``.
    """

    def __init__(
        self,
        backend: str = "auto",
        sh_degree: int = 3,
        background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        super().__init__()
        if backend not in {"auto", "reference", "diff_gaussian"}:
            raise ValueError(f"backend must be auto, reference, or diff_gaussian, got {backend!r}")
        self.backend = backend
        self.sh_degree = sh_degree
        self.reference = ReferenceSplatRenderer(sh_degree=sh_degree)
        self.diff_gaussian = DiffGaussianRasterizationAdapter(sh_degree=sh_degree)
        self.register_buffer(
            "background",
            torch.tensor(background, dtype=torch.float32),
            persistent=False,
        )

    def resolved_backend(self) -> str:
        if self.backend == "reference":
            return "reference"
        if self.backend == "diff_gaussian":
            return "diff_gaussian"
        if self.diff_gaussian.available():
            return "diff_gaussian"
        return "reference"

    def forward(
        self,
        packed: Tensor,
        cameras: dict[str, dict[str, Tensor]] | None,
        valid: Tensor | None = None,
        images: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        if cameras is None:
            raise ValueError("cameras is required: each entry needs K, R, t, and an image size")
        rasterizer_view(packed)
        backend = self.resolved_backend()
        if backend == "diff_gaussian":
            return self.diff_gaussian(
                packed,
                cameras,
                valid=valid,
                images=images,
                background=self.background,
            )
        outputs: dict[str, Tensor] = {}
        for name, camera in cameras.items():
            height, width = _camera_hw(camera, None if images is None else images.get(name))
            outputs[name] = self.reference(
                packed,
                camera["K"],
                camera["R"],
                camera["t"],
                height,
                width,
                valid=valid,
                background=self.background,
            )
        return outputs


def _raster_settings(module, **kwargs):
    """Construct settings, dropping keys this build of the extension does not take.

    The stock Inria signature has no ``antialiasing`` field. Forks that added
    one usually default it, and we pass ``False`` when the name exists. The
    principal point is not a settings field; it lives in the projection matrix.
    """
    signature = inspect.signature(module.GaussianRasterizationSettings)
    accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return module.GaussianRasterizationSettings(**accepted)


def _camera_hw(camera: dict, image: Tensor | None) -> tuple[int, int]:
    if image is not None:
        return int(image.shape[-2]), int(image.shape[-1])
    if "height" in camera and "width" in camera:
        return int(camera["height"]), int(camera["width"])
    raise ValueError("rasterizer needs images or cameras[name]['height'/'width']")


def _screen_covariance(
    quat: Tensor,
    scale: Tensor,
    R: Tensor,
    xyz_cam: Tensor,
    K: Tensor,
    low_pass: float,
) -> Tensor:
    rotation = quat_to_matrix(quat)
    # Columns of R_q scaled by s, then Sigma = M M^T. Scales are already meters.
    scaled = rotation * scale.unsqueeze(-2)
    sigma_world = scaled @ scaled.transpose(-1, -2)
    sigma_cam = torch.einsum("bij,bnjk,blk->bnil", R, sigma_world, R)
    depth = xyz_cam[..., 2]
    safe_z = torch.where(depth.abs() < 1e-6, torch.ones_like(depth), depth)
    z2 = safe_z * safe_z
    fx = K[:, 0, 0][:, None]
    fy = K[:, 1, 1][:, None]
    jacobian = quat.new_zeros(quat.shape[0], quat.shape[1], 2, 3)
    jacobian[..., 0, 0] = fx / safe_z
    jacobian[..., 0, 2] = -fx * xyz_cam[..., 0] / z2
    jacobian[..., 1, 1] = fy / safe_z
    jacobian[..., 1, 2] = -fy * xyz_cam[..., 1] / z2
    sigma_2d = torch.einsum("bnij,bnjk,bnlk->bnil", jacobian, sigma_cam, jacobian)
    eye = torch.eye(2, device=quat.device, dtype=quat.dtype).view(1, 1, 2, 2)
    return sigma_2d + low_pass * eye


def _alpha_maps(
    opacity: Tensor,
    sigma_2d: Tensor,
    pixel_u: Tensor,
    pixel_v: Tensor,
    depth: Tensor,
    height: int,
    width: int,
    valid: Tensor | None,
) -> Tensor:
    ys = torch.arange(height, device=pixel_u.device, dtype=pixel_u.dtype)
    xs = torch.arange(width, device=pixel_u.device, dtype=pixel_u.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    du = grid_x.view(1, 1, height, width) - pixel_u[:, :, None, None]
    dv = grid_y.view(1, 1, height, width) - pixel_v[:, :, None, None]
    a = sigma_2d[..., 0, 0]
    b = sigma_2d[..., 0, 1]
    c = sigma_2d[..., 1, 0]
    d = sigma_2d[..., 1, 1]
    det = (a * d - b * c).clamp_min(1e-12)
    inv_00 = d / det
    inv_01 = -b / det
    inv_10 = -c / det
    inv_11 = a / det
    solved_u = inv_00[:, :, None, None] * du + inv_01[:, :, None, None] * dv
    solved_v = inv_10[:, :, None, None] * du + inv_11[:, :, None, None] * dv
    mahalanobis = (du * solved_u + dv * solved_v).clamp_min(0)
    alpha = opacity[..., 0][:, :, None, None] * torch.exp(-0.5 * mahalanobis)
    alpha = alpha.clamp(max=0.99)
    in_front = depth > 0.05
    if valid is not None:
        in_front = in_front & valid
    return alpha * in_front[:, :, None, None].to(dtype=alpha.dtype)


def _view_colors(xyz: Tensor, sh: Tensor, R: Tensor, t: Tensor, degree: int) -> Tensor:
    """RGB from SH evaluated along the world-frame ray from the camera center."""
    center = -torch.einsum("bji,bj->bi", R, t)
    direction = xyz - center[:, None, :]
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    if degree < 3:
        kept = sh.clone()
        kept[..., (degree + 1) ** 2 :, :] = 0
        sh = kept
    return (eval_sh(sh, direction, degree) + 0.5).clamp(0, 1)
