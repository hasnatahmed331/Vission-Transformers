# models/vision_transformer.py

import torch
import torch.nn as nn
from einops import repeat
from .base_layers import PatchEmbed, TransformerBlock
from .t2t_layer import T2TPatchEmbed


class ViT(nn.Module):
    def __init__(
        self,
        image_size=64,
        patch_size=16,
        in_chans=3,
        dim=256,
        depth=6,
        heads=8,
        mlp_ratio=4.0,
        num_classes=10,
        dropout=0.1,
        use_t2t: bool = False,
        t2t_token_dim: int = 64,
        t2t_specs=((7, 4), (3, 2), (3, 2)),
    ):
        super().__init__()

        if use_t2t:
            # T2T-based patch embedding
            self.patch_embed = T2TPatchEmbed(
                img_size=image_size,
                in_chans=in_chans,
                embed_dim=dim,
                token_dim=t2t_token_dim,
                t2t_specs=t2t_specs,
                drop=dropout,
            )
            num_patches = self.patch_embed.num_patches
        else:
            # Conv/linear patch embedding
            self.patch_embed = PatchEmbed(
                img_size=image_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=dim,
            )
            num_patches = self.patch_embed.num_patches  # from your existing class

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))

        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, mlp_ratio) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        b = x.size(0)
        x = self.patch_embed(x)  # (B, N, D)

        cls = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding

        for blk in self.blocks:
            x = blk(x)

        return self.head(self.norm(x[:, 0]))  # CLS token
