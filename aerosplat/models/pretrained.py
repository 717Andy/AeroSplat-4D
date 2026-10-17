"""Load DINOv2 or DUSt3R / MASt3R encoder weights into :class:`DinoEncoder`.

AerialMegaDepth publishes aerial fine-tunes of DUSt3R and MASt3R. Those
checkpoints use CroCo names (``enc_blocks``, ``enc_norm``). DINOv2 uses
``blocks`` and ``norm`` directly. Both land on ``ViTBackbone.encoder`` when
the tensor shapes match. Keys that do not match (the DUSt3R decoder, a
different ViT width, a class token from the wrong depth) are skipped and
listed in the report. Nothing is downloaded unless ``allow_download`` is set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_PREFIXES = (
    "module.",
    "model.",
    "state_dict.",
    "backbone.",
    "encoder.",
)
# CroCo / DUSt3R / MASt3R encoder names -> our DinoEncoder names.
_REMAP = (
    ("enc_blocks.", "blocks."),
    ("enc_norm.", "norm."),
    ("decoder_embed.", "__skip__"),
    ("dec_blocks.", "__skip__"),
    ("dec_norm.", "__skip__"),
    ("downstream_head", "__skip__"),
    ("mask_token", "__skip__"),
    ("head.", "__skip__"),
)

DINOV2_NAMES = {
    "dinov2_vits14": "dinov2_vits14",
    "dinov2_vitb14": "dinov2_vitb14",
    "dinov2_vitl14": "dinov2_vitl14",
}


@dataclass
class LoadReport:
    loaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def loaded_count(self) -> int:
        return len(self.loaded)


def load_pretrained_weights(
    module: nn.Module,
    source: str | Path,
    *,
    allow_download: bool = False,
) -> LoadReport:
    """Copy a checkpoint into ``module``.

    ``source`` is a filesystem path, or a DINOv2 hub name (``dinov2_vits14``,
    ``dinov2_vitb14``, ``dinov2_vitl14``) when ``allow_download`` is true.
    The module is the backbone's ``encoder`` (or the backbone itself; a leading
    ``encoder.`` in the module state is stripped when matching).
    """
    state = _read_source(source, allow_download=allow_download)
    remapped, dropped = remap_encoder_keys(state)
    report = apply_encoder_state(module, remapped)
    report.skipped = dropped + report.skipped
    return report


def remap_encoder_keys(
    state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Drop decoder keys and rename CroCo encoder keys to DINOv2 names.

    The second return is a list of keys that were dropped (DUSt3R's decoder,
    heads, mask token). Callers should surface those in the load report.
    """
    remapped: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    for key, value in state.items():
        name = _strip_prefix(key)
        if not torch.is_tensor(value):
            continue
        skip = False
        for src, dst in _REMAP:
            if name.startswith(src) or name == src:
                if dst == "__skip__":
                    skip = True
                    break
                name = dst + name[len(src) :]
                break
        if skip or name.startswith("__skip__"):
            dropped.append(f"{key}: not an encoder parameter")
            continue
        remapped[name] = value
    return remapped, dropped


def apply_encoder_state(module: nn.Module, state: dict[str, torch.Tensor]) -> LoadReport:
    params = dict(module.named_parameters())
    report = LoadReport()
    # A backbone module stores the tower under ``encoder.``.
    prefix = "encoder." if any(key.startswith("encoder.") for key in params) else ""
    for key, value in state.items():
        dest = prefix + key if prefix and not key.startswith(prefix) else key
        if dest not in params:
            report.skipped.append(f"{key}: no parameter {dest}")
            continue
        current = params[dest]
        if current.shape == value.shape:
            current.data.copy_(value.to(dtype=current.dtype, device=current.device))
            report.loaded.append(dest)
            continue
        if dest.endswith("pos_embed") and _is_pos_embed(current, value):
            current.data.copy_(_interpolate_pos_embed(value, current).to(dtype=current.dtype))
            report.loaded.append(dest + " (interpolated)")
            continue
        report.skipped.append(f"{key}: shape {tuple(value.shape)} != {tuple(current.shape)}")
    loaded_names = {item.split(" ")[0] for item in report.loaded}
    report.missing = [
        key for key in params if key not in loaded_names and _is_encoder_key(key, prefix)
    ]
    return report


def _read_source(source: str | Path, *, allow_download: bool) -> dict[str, torch.Tensor]:
    text = str(source)
    path = Path(text)
    if path.is_file():
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return _unwrap_state(blob)
    if text in DINOV2_NAMES:
        if not allow_download:
            raise FileNotFoundError(
                f"{text} is a DINOv2 hub name, not a local file. Download the checkpoint "
                "and pass the path, or set pretrained_allow_download=true. "
                "CI does not download weights."
            )
        hub = torch.hub.load("facebookresearch/dinov2", DINOV2_NAMES[text])
        return hub.state_dict()
    raise FileNotFoundError(
        f"pretrained weights {text!r} were not found. Pass a .pt/.pth file, "
        f"or one of {sorted(DINOV2_NAMES)} with downloads enabled."
    )


def _unwrap_state(blob) -> dict[str, torch.Tensor]:
    if isinstance(blob, dict):
        for key in ("state_dict", "model", "model_state_dict", "encoder"):
            inner = blob.get(key)
            if isinstance(inner, dict) and any(torch.is_tensor(v) for v in inner.values()):
                return {str(k): v for k, v in inner.items() if torch.is_tensor(v)}
        if any(torch.is_tensor(v) for v in blob.values()):
            return {str(k): v for k, v in blob.items() if torch.is_tensor(v)}
    raise ValueError("checkpoint does not contain a state dict")


def _strip_prefix(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in _PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def _is_pos_embed(current: torch.Tensor, value: torch.Tensor) -> bool:
    return (
        current.ndim == 3
        and value.ndim == 3
        and current.shape[0] == 1
        and value.shape[0] == 1
        and current.shape[-1] == value.shape[-1]
    )


def _interpolate_pos_embed(value: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    cls, patch = value[:, :1], value[:, 1:]
    count = patch.shape[1]
    side = int(round(count**0.5))
    if side * side != count:
        raise ValueError(f"cannot interpolate pos_embed with {count} patches")
    out_count = current.shape[1] - 1
    out_side = int(round(out_count**0.5))
    if out_side * out_side != out_count:
        raise ValueError(f"model pos_embed is not square ({out_count} patches)")
    patch = patch.reshape(1, side, side, -1).permute(0, 3, 1, 2).float()
    patch = F.interpolate(patch, size=(out_side, out_side), mode="bicubic", align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, out_count, -1)
    return torch.cat((cls.to(dtype=patch.dtype), patch), dim=1).to(dtype=current.dtype)


def _is_encoder_key(key: str, prefix: str) -> bool:
    name = key[len(prefix) :] if prefix and key.startswith(prefix) else key
    return name.split(".")[0] in {"patch_embed", "blocks", "norm", "cls_token", "pos_embed"}
