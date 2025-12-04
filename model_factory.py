# model_factory.py

from typing import Tuple, List
from .vit import ViT
from .cross_vit import CrossViT


class ModelFactory:
    @staticmethod
    def build(
        model_type: str,
        img_size: int = 64,
        num_classes: int = 10,
        # ViT-specific
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        # CrossViT-specific
        cross_patch_sizes: Tuple[int, int] = (8, 16),
        cross_dims: Tuple[int, int] = (128, 256),
        cross_heads: Tuple[int, int] = (4, 4),
        cross_depth: List[Tuple[int, int, int]] = None,
        cross_mlp_ratio: Tuple[float, float, float] = (3.0, 3.0, 1.0),
        # common
        dropout: float = 0.1,
        qkv_bias: bool = True,
        # ---- NEW ----
        use_t2t: bool = False,
        t2t_token_dim: int = 64,
        t2t_specs_vit=((7, 4), (3, 2), (3, 2)),
        t2t_specs_small=None,
        t2t_specs_large=None,
    ):

        if model_type == "vit":
            print(
                f"Building ViT | patches: {patch_size}, dim: {embed_dim}, "
                f"layers: {depth}, heads: {num_heads}, use_t2t={use_t2t}"
            )
            return ViT(
                image_size=img_size,
                patch_size=patch_size,
                dim=embed_dim,
                depth=depth,
                heads=num_heads,
                mlp_ratio=4,
                num_classes=num_classes,
                dropout=dropout,
                use_t2t=use_t2t,
                t2t_token_dim=t2t_token_dim,
                t2t_specs=t2t_specs_vit,
            )

        elif model_type == "crossvit":
            if cross_depth is None:
                cross_depth = [(1, 3, 1), (1, 3, 1), (1, 3, 1)]

            print(
                "Building CrossViT | "
                f"patches={cross_patch_sizes}, dims={cross_dims}, "
                f"heads={cross_heads}, depth={cross_depth}, use_t2t={use_t2t}"
            )

            return CrossViT(
                img_size=img_size,
                patch_size=cross_patch_sizes,
                in_chans=3,
                num_classes=num_classes,
                embed_dim=cross_dims,
                depth=cross_depth,
                num_heads=cross_heads,
                mlp_ratio=cross_mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=dropout,
                attn_drop_rate=dropout,
                drop_path_rate=dropout,
                use_t2t=use_t2t,
                t2t_token_dim=t2t_token_dim,
                t2t_specs_small=t2t_specs_small,
                t2t_specs_large=t2t_specs_large,
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")
