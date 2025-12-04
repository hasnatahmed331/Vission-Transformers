# main.py

import torch
from model_factory import ModelFactory

if __name__ == "__main__":
    # Example 1: plain ViT
    vit = ModelFactory.build(
        model_type="vit",
        img_size=64,
        patch_size=8,
        embed_dim=128,
        depth=4,
        num_heads=4,
        num_classes=10,
    )

    # Example 2: CrossViT with control over encoders + cross-attention
    crossvit = ModelFactory.build(
        model_type="crossvit",
        img_size=64,
        num_classes=10,
        cross_patch_sizes=(8, 16),  # small & large patches
        cross_dims=(128, 256),  # branch dims
        cross_heads=(4, 4),  # heads per branch
        cross_depth=[  # (small_depth, large_depth, cross_att_depth)
            (1, 3, 1),
            (1, 3, 1),
            (1, 3, 1),
        ],
        cross_mlp_ratio=(3.0, 3.0, 1.0),
    )

    x = torch.randn(2, 3, 64, 64)
    out_vit = vit(x)
    out_cross = crossvit(x)

    print("ViT output:", out_vit.shape)  # [B, num_classes]
    print("CrossViT output:", out_cross.shape)  # [B, num_classes]
