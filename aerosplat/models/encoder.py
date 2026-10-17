"""DINOv2-shaped Vision Transformer encoder.

The block names match a DINOv2 (and CroCo / DUSt3R) checkpoint closely enough
that :mod:`aerosplat.models.pretrained` can copy weights in:

    patch_embed.proj
    cls_token, pos_embed
    blocks.{i}.norm1, attn.qkv, attn.proj, ls1.gamma
    blocks.{i}.norm2, mlp.fc1, mlp.fc2, ls2.gamma
    norm

LayerScale gamma is initialized to 1. DINOv2 checkpoints overwrite it. A
DUSt3R / MASt3R encoder has no LayerScale; leaving gamma at 1 keeps those
blocks as ordinary residuals instead of silently multiplying them by 1e-5.

The class token exists so a pretrained ``pos_embed`` lines up. Dense outputs
drop it. Position embeddings are interpolated when the image grid differs from
the checkpoint grid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class Attention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, qkv_bias: bool = True, proj_bias: bool = True
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch, length, channels)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        init_values: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim, init_values)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)
        self.ls2 = LayerScale(dim, init_values)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int, in_chans: int = 3) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class DinoEncoder(nn.Module):
    """Shared per-image tower. It does not see other cameras."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        out_indices: tuple[int, ...] = (11,),
        init_values: float = 1.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("encoder depth must be >= 1")
        for index in out_indices:
            if index < 0 or index >= depth:
                raise ValueError(f"out_indices entry {index} is outside 0..{depth - 1}")
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_indices = tuple(out_indices)
        self.gradient_checkpointing = gradient_checkpointing
        grid = img_size // patch_size
        if grid < 1:
            raise ValueError("img_size must be >= patch_size")
        self.patch_embed = PatchEmbed(patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + grid * grid, embed_dim))
        self.blocks = nn.ModuleList(
            Block(embed_dim, num_heads, mlp_ratio, init_values=init_values) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_module)

    def interpolate_pos_embed(self, grid_h: int, grid_w: int) -> Tensor:
        cls, patch = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        count = patch.shape[1]
        side = int(round(count**0.5))
        if side * side != count:
            raise RuntimeError(f"pos_embed has {count} patches, which is not a square grid")
        patch = patch.reshape(1, side, side, -1).permute(0, 3, 1, 2)
        if side != grid_h or side != grid_w:
            patch = F.interpolate(patch, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        patch = patch.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)
        return torch.cat((cls, patch), dim=1)

    def forward_tokens(
        self,
        image: Tensor,
        token_bias: Tensor | None = None,
    ) -> tuple[Tensor, dict[int, Tensor], int, int]:
        """Encode one camera.

        ``image`` is ``[B, 3, H, W]``. ``H`` and ``W`` are padded up to a
        multiple of the patch size inside the caller; this conv then drops no
        pixels. ``token_bias`` is ``[B, L, C]`` added to the patch tokens
        (not the class token) before the blocks, which is the pose injection.

        Returns patch tokens ``[B, L, C]`` after the final norm, a dict of
        intermediate patch tokens keyed by block index (also normed), and the
        grid shape.
        """
        tokens = self.patch_embed(image)
        batch, length, _ = tokens.shape
        grid_h = image.shape[-2] // self.patch_size
        grid_w = image.shape[-1] // self.patch_size
        if grid_h * grid_w != length:
            raise RuntimeError("patch grid does not match the image; pad to the patch size first")
        cls = self.cls_token.expand(batch, -1, -1)
        hidden = torch.cat((cls, tokens), dim=1)
        hidden = hidden + self.interpolate_pos_embed(grid_h, grid_w)
        if token_bias is not None:
            if token_bias.shape[:2] != (batch, length):
                raise ValueError(
                    f"pose bias shape {tuple(token_bias.shape)} != "
                    f"{(batch, length, self.embed_dim)}"
                )
            hidden = hidden.clone()
            hidden[:, 1:] = hidden[:, 1:] + token_bias
        layers: dict[int, Tensor] = {}
        for index, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
            if index in self.out_indices:
                layers[index] = self.norm(hidden[:, 1:])
        hidden = self.norm(hidden)
        return hidden[:, 1:], layers, grid_h, grid_w


def _init_module(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
