# models/base_layers.py

import torch
import torch.nn as nn
from einops import rearrange, repeat


def pair(x):
    return x if isinstance(x, tuple) else (x, x)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.attn = nn.Sequential(nn.LayerNorm(dim), Attention(dim, heads))
        self.ff = nn.Sequential(nn.LayerNorm(dim), FeedForward(dim, dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class PatchEmbed(nn.Module):
    """
    Flexible Patch Embedding:
    - If patch_size is int → behaves as ViT embedding
    - If patch_size is tuple/list → creates multi-branch embeddings (CrossViT)
    """

    def __init__(self, img_size=64, patch_size=16, in_chans=3, embed_dim=256):
        super().__init__()

        img_size = to_2tuple(img_size)

        # ensure embed_dim supports either one or multiple patch sizes
        if isinstance(patch_size, int):
            patch_size = [patch_size]
        if isinstance(embed_dim, int):
            embed_dim = [embed_dim] * len(patch_size)

        assert len(embed_dim) == len(
            patch_size
        ), "embed_dim list must match patch branches"

        self.branches = nn.ModuleList()
        self.num_branches = len(patch_size)
        self.num_patches = []

        for p, dim in zip(patch_size, embed_dim):
            p = to_2tuple(p)
            n_patches = (img_size[0] // p[0]) * (img_size[1] // p[1])
            self.num_patches.append(n_patches)

            # unified method (Conv2D stride=patch), valid for ViT and CrossViT
            self.branches.append(nn.Conv2d(in_chans, dim, kernel_size=p, stride=p))

    def forward(self, x):
        out = []
        for conv in self.branches:
            y = conv(x)  # (B, E, H', W')
            y = y.flatten(2).transpose(1, 2)  # → (B, N, E)
            out.append(y)

        return (
            out[0] if self.num_branches == 1 else out
        )  # single branch = ViT, multi=CrossViT
