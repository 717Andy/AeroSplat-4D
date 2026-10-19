"""Composite captured lens flares onto clean frames.

The Google Research flare-removal set (Wu et al., ICCV 2021) is CC BY 4.0.
It is not downloaded here. Point ``root`` at a local copy. Two layouts are
accepted:

* ``root/flare/*.png`` (or ``.jpg``) with an optional ``root/mask/`` of the
  same stem. This is the captured-flare layout: the flare sits on black, and
  training adds it onto a clean photograph.
* ``root/Flare/`` and optional ``root/Light_source/``, the names used by some
  mirrors of that release.

A mask marks the pixels whose flare value is trusted. When the directory has
no mask, the mask is the flare support after gain and placement (luminance
above a small threshold). Synthetic paired renders from the simulator do not
go through this loader. Those are a clean image and a flared image on the
sample; :func:`paired_glare_target` turns the pair into a layer and a mask.

The composite is the same addition the relight head uses::

    augmented = clamp(clean + glare, 0, 1)

``glare`` is the known pre-clip layer. Clipping is the sensor. The separation
loss compares the head's layer to ``glare`` inside the mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from aerosplat.data.io import load_rgb

FLARE_LICENSE = "CC BY 4.0"
FLARE_ATTRIBUTION = (
    "Google Research flare removal dataset, Wu et al., ICCV 2021, CC BY 4.0. "
    "https://github.com/google-research/google-research/tree/master/flare_removal"
)
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class FlareComposite:
    """One augmented camera image and the layer that was added."""

    image: Tensor
    glare: Tensor
    mask: Tensor
    clean: Tensor
    source: str


class FlareLibrary:
    """Flare images on disk. Empty until :meth:`load`. No network access."""

    def __init__(self, root: Path | None) -> None:
        self.root = None if root is None else Path(root)
        self.flares: list[Path] = []
        self.masks: dict[str, Path] = {}
        if self.root is not None:
            self._discover()

    def _discover(self) -> None:
        assert self.root is not None
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"flare root {self.root} does not exist. "
                "Download the Google flare-removal set locally (CC BY 4.0) "
                "or point train.flare_root at a directory of flare PNGs."
            )
        flare_dir = self.root / "flare"
        mask_dir = self.root / "mask"
        if not flare_dir.is_dir() and (self.root / "Flare").is_dir():
            flare_dir = self.root / "Flare"
            mask_dir = self.root / "Light_source"
        if not flare_dir.is_dir():
            flare_dir = self.root
            mask_dir = self.root / "mask"
        self.flares = sorted(
            path
            for path in flare_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not self.flares:
            raise FileNotFoundError(f"no flare images under {flare_dir}")
        if mask_dir.is_dir():
            for path in mask_dir.iterdir():
                if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                    self.masks[path.stem] = path

    def __len__(self) -> int:
        return len(self.flares)

    def load(self, index: int) -> tuple[Tensor, Tensor | None, str]:
        """``(flare [3,H,W], mask [1,H,W] or None, filename)``."""
        if not self.flares:
            raise RuntimeError("flare library is empty")
        path = self.flares[int(index) % len(self.flares)]
        flare = torch.from_numpy(load_rgb(path))
        mask_path = self.masks.get(path.stem)
        mask = None
        if mask_path is not None:
            loaded = torch.from_numpy(load_rgb(mask_path))
            mask = loaded.mean(dim=0, keepdim=True).clamp(0, 1)
        return flare, mask, path.name


class FlareAugmenter:
    """Paste a random flare onto a clean ``[3, H, W]`` or ``[B, 3, H, W]`` image."""

    def __init__(
        self,
        library: FlareLibrary,
        *,
        probability: float = 0.5,
        gain: tuple[float, float] = (0.6, 1.4),
        threshold: float = 0.02,
        generator: torch.Generator | None = None,
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must lie in [0, 1]")
        if gain[0] < 0 or gain[1] < gain[0]:
            raise ValueError(f"gain range must be 0 <= low <= high, got {gain}")
        self.library = library
        self.probability = float(probability)
        self.gain = (float(gain[0]), float(gain[1]))
        self.threshold = float(threshold)
        self.generator = generator

    def maybe(self, image: Tensor) -> FlareComposite | None:
        """Return a composite, or ``None`` when this draw skips augmentation."""
        if len(self.library) == 0:
            return None
        draw = torch.rand((), generator=self.generator).item()
        if draw > self.probability:
            return None
        index = int(torch.randint(len(self.library), (), generator=self.generator).item())
        return self.composite(image, index=index)

    def composite(self, image: Tensor, index: int, gain: float | None = None) -> FlareComposite:
        """Add flare ``index`` onto ``image``. ``gain`` skips the random draw."""
        batched = image.ndim == 4
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"image must be [3, H, W] or [B, 3, H, W], got {tuple(image.shape)}")
        flare, mask, name = self.library.load(index)
        if gain is None:
            low, high = self.gain
            unit = torch.rand((), generator=self.generator)
            gain = float(low + (high - low) * unit)
        placed = _place(flare, image.shape[-2:], image.shape[0], self.generator)
        if mask is None:
            soft = (placed.mean(dim=1, keepdim=True) > self.threshold).to(dtype=image.dtype)
        else:
            soft = _place(mask.expand(3, -1, -1), image.shape[-2:], image.shape[0], self.generator)
            soft = soft.mean(dim=1, keepdim=True).clamp(0, 1)
        layer = placed * float(gain)
        augmented = (image + layer).clamp(0, 1)
        result = FlareComposite(
            image=augmented,
            glare=layer,
            mask=soft,
            clean=image,
            source=name,
        )
        if not batched:
            return FlareComposite(
                image=result.image[0],
                glare=result.glare[0],
                mask=result.mask[0],
                clean=result.clean[0],
                source=result.source,
            )
        return result


def paired_glare_target(
    observed: Tensor,
    clean: Tensor,
    *,
    threshold: float = 1.0 / 255.0,
) -> tuple[Tensor, Tensor]:
    """``(glare, mask)`` from a simulator pair. Glare is the positive difference.

    ``observed`` is the flare-on frame. ``clean`` is the flare-off frame.
    Negative differences (a flare that darkens a pixel) are clipped to zero:
    the head only adds light. The mask is 1 where that positive difference
    exceeds ``threshold``.
    """
    if observed.shape != clean.shape:
        raise ValueError(f"observed {tuple(observed.shape)} != clean {tuple(clean.shape)}")
    glare = (observed - clean).clamp(min=0)
    mask = (glare.sum(dim=-3, keepdim=True) > threshold).to(dtype=glare.dtype)
    return glare, mask


def _place(
    flare: Tensor,
    size: tuple[int, int],
    batch: int,
    generator: torch.Generator | None,
) -> Tensor:
    """Resize the flare to the frame and shift it by a random translation.

    The shift is a circular roll. A flare that hangs off one edge re-enters
    on the other, which is acceptable for a training paste and keeps the
    tensor shape static.
    """
    height, width = size
    resized = F.interpolate(
        flare.unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0]
    max_y = max(height // 4, 1)
    max_x = max(width // 4, 1)
    shift_y = int(torch.randint(-max_y, max_y + 1, (), generator=generator).item())
    shift_x = int(torch.randint(-max_x, max_x + 1, (), generator=generator).item())
    moved = torch.roll(resized, shifts=(shift_y, shift_x), dims=(-2, -1))
    return moved.unsqueeze(0).expand(batch, -1, -1, -1).contiguous()


def write_synthetic_flare(directory: Path, name: str = "blob") -> Path:
    """A black PNG with a bright blob, for tests. Not a Google image."""
    directory = Path(directory)
    flare_dir = directory / "flare"
    flare_dir.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[-1:1:64j, -1:1:64j]
    blob = np.exp(-((xx - 0.2) ** 2 + (yy + 0.1) ** 2) / 0.02)
    rgb = np.stack((blob, blob * 0.85, blob * 0.4), axis=-1)
    image = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    path = flare_dir / f"{name}.png"
    from PIL import Image

    Image.fromarray(image, mode="RGB").save(path)
    return path
