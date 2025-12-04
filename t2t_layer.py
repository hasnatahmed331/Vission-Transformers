# models/t2t_layers.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------- T2T internal transformer block ----------------- #


class T2TTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: (B, N, C)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop(attn_out)
        x_norm = self.norm2(x)
        x = x + self.drop(self.mlp(x_norm))
        return x


class T2TModule(nn.Module):
    """
    Progressive tokenization:
      x -> conv(7x7, s=4) -> tokens -> transformer
        -> conv(3x3, s=2 or 1) -> tokens -> transformer
        -> conv(3x3, s=2 or 1) -> tokens -> transformer
        -> final tokens (B, N, token_dim)
    """

    def __init__(
        self,
        img_size=160,
        in_chans=3,
        token_dim=64,
        t2t_specs=((7, 4), (3, 2), (3, 2)),
        drop=0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.token_dim = token_dim
        self.t2t_specs = t2t_specs

        convs = []
        for i, (k, s) in enumerate(t2t_specs):
            in_c = in_chans if i == 0 else token_dim
            convs.append(
                nn.Conv2d(
                    in_channels=in_c,
                    out_channels=token_dim,
                    kernel_size=k,
                    stride=s,
                    padding=k // 2,
                    bias=False,
                )
            )
        self.convs = nn.ModuleList(convs)

        self.blocks = nn.ModuleList(
            [
                T2TTransformerBlock(token_dim, num_heads=4, mlp_ratio=2.0, drop=drop)
                for _ in t2t_specs
            ]
        )

        # compute final H, W and num_patches
        H = W = img_size
        for k, s in t2t_specs:
            pad = k // 2
            H = math.floor((H + 2 * pad - k) / s + 1)
            W = math.floor((W + 2 * pad - k) / s + 1)
        self.final_hw = (H, W)
        self.num_patches = H * W

    def forward(self, x):
        # x: (B, C, H, W)
        B = x.shape[0]
        for conv, block in zip(self.convs, self.blocks):
            x = conv(x)  # (B, token_dim, H', W')
            B, C, H, W = x.shape
            tokens = x.flatten(2).transpose(1, 2)  # (B, N, C)
            tokens = block(tokens)
            x = tokens.transpose(1, 2).reshape(B, C, H, W)

        tokens = x.flatten(2).transpose(1, 2)  # (B, N, token_dim)
        return tokens


# ----------------- Public T2T patch embedding ----------------- #


class T2TPatchEmbed(nn.Module):
    """
    ViT-style patch embedding implemented via T2T progressive tokenization.

    Output:
      - x: (B, N, embed_dim)
      - attributes: num_patches
    """

    def __init__(
        self,
        img_size=160,
        in_chans=3,
        embed_dim=384,
        token_dim=64,
        t2t_specs=((7, 4), (3, 2), (3, 2)),
        drop=0.0,
    ):
        super().__init__()

        self.t2t = T2TModule(
            img_size=img_size,
            in_chans=in_chans,
            token_dim=token_dim,
            t2t_specs=t2t_specs,
            drop=drop,
        )

        self.num_patches = self.t2t.num_patches
        self.proj = nn.Linear(token_dim, embed_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: (B, C, H, W)
        tokens = self.t2t(x)  # (B, N, token_dim)
        tokens = self.proj(tokens)  # (B, N, embed_dim)
        tokens = self.drop(tokens)
        return tokens  # (B, N, embed_dim)
