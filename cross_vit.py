# models/crossvit.py

import torch
import torch.nn as nn
from typing import List, Tuple

from .base_layers import PatchEmbed, TransformerBlock
from .t2t_layer import T2TPatchEmbed


class DropPath(nn.Module):
    """
    Stochastic depth / drop-path.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class CrossAttention(nn.Module):
    """
    Single-token cross-attention: CLS attends to all tokens.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale=None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: (B, N, C), first token is CLS
        B, N, C = x.shape

        # CLS as query
        q = self.wq(x[:, :1])  # (B, 1, C)
        q = q.view(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # all tokens as key/value
        k = (
            self.wk(x)
            .view(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.wv(x)
            .view(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, 1, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (B, 1, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class CrossAttentionBlock(nn.Module):
    """
    LN -> cross-attention (CLS only) + residual.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale=None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # x: (B, N, C) -> only CLS gets updated
        cls_updated = self.attn(self.norm1(x))
        return x[:, :1] + self.drop_path(cls_updated)


class MultiScaleBlock(nn.Module):
    """
    One multi-scale stage:

      - For each branch:
          stack of self-attention blocks.
      - Cross-branch fusion:
          CLS token from each branch attends to other branch's patch tokens,
          then projected back.
    """

    def __init__(
        self,
        dim: List[int],
        num_patches: List[int],
        depth: Tuple[int, int, int],  # (depth_branch0, depth_branch1, depth_cross)
        num_heads: List[int],
        mlp_ratio: Tuple[float, float, float],
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert (
            len(dim) == len(num_patches) == len(num_heads) == 2
        ), "This implementation assumes 2 branches."

        self.num_branches = len(dim)
        d_b0, d_b1, d_cross = depth

        # --- Per-branch transformer stacks ---
        self.branch_blocks = nn.ModuleList()
        for b, d in enumerate(dim):
            blocks = []
            num_layer = depth[b]  # first two entries: per-branch depth
            for _ in range(num_layer):
                blocks.append(
                    TransformerBlock(
                        dim=d,
                        heads=num_heads[b],
                        mlp_ratio=mlp_ratio[b],
                        dropout=drop,
                    )
                )
            self.branch_blocks.append(
                nn.Sequential(*blocks) if len(blocks) > 0 else nn.Identity()
            )

        # --- Projections of CLS tokens between branches ---
        self.projs = nn.ModuleList()
        for b in range(self.num_branches):
            in_dim = dim[b]
            out_dim = dim[(b + 1) % self.num_branches]
            if in_dim == out_dim:
                self.projs.append(nn.Identity())
            else:
                self.projs.append(
                    nn.Sequential(
                        norm_layer(in_dim),
                        act_layer(),
                        nn.Linear(in_dim, out_dim),
                    )
                )

        # --- Cross-attention fusion modules ---
        self.fusion = nn.ModuleList()
        for b in range(self.num_branches):
            tgt_dim = dim[(b + 1) % self.num_branches]
            nh = num_heads[(b + 1) % self.num_branches]
            fusion_layers = []
            for _ in range(d_cross):
                fusion_layers.append(
                    CrossAttentionBlock(
                        dim=tgt_dim,
                        num_heads=nh,
                        mlp_ratio=mlp_ratio[b],
                        qkv_bias=qkv_bias,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path,
                        norm_layer=norm_layer,
                    )
                )
            self.fusion.append(
                nn.Sequential(*fusion_layers)
                if len(fusion_layers) > 0
                else nn.Identity()
            )

        # --- Project fused CLS back to original dims ---
        self.revert_projs = nn.ModuleList()
        for b in range(self.num_branches):
            in_dim = dim[(b + 1) % self.num_branches]
            out_dim = dim[b]
            if in_dim == out_dim:
                self.revert_projs.append(nn.Identity())
            else:
                self.revert_projs.append(
                    nn.Sequential(
                        norm_layer(in_dim),
                        act_layer(),
                        nn.Linear(in_dim, out_dim),
                    )
                )

    def forward(self, xs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        xs: list of [x_small, x_large], each (B, N, C)
        """
        assert len(xs) == self.num_branches

        # 1) Per-branch Transformer encoding
        encoded = []
        for x, block in zip(xs, self.branch_blocks):
            encoded.append(block(x))

        # 2) Project CLS tokens to the other branch's dim
        proj_cls = []
        for x, proj in zip(encoded, self.projs):
            proj_cls.append(proj(x[:, :1, :]))  # (B, 1, dim_other)

        # 3) Cross attention: each branch CLS attends to other branch's tokens
        outs = []
        for b in range(self.num_branches):
            other = (b + 1) % self.num_branches
            other_tokens = encoded[other][:, 1:, :]  # (B, N_other, C_other)
            fusion_in = torch.cat([proj_cls[b], other_tokens], dim=1)
            fused_cls = self.fusion[b](fusion_in)  # (B, 1, C_other)
            reverted_cls = self.revert_projs[b](fused_cls[:, :1, :])  # (B, 1, C_b)
            this_tokens = encoded[b][:, 1:, :]  # keep original patches
            outs.append(torch.cat([reverted_cls, this_tokens], dim=1))

        return outs


class CrossViT(nn.Module):
    """
    Dual-branch CrossViT with shared PatchEmbed backend or T2T backend.
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size=(8, 16),
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim=(128, 256),
        depth=((1, 3, 1), (1, 3, 1), (1, 3, 1)),
        num_heads=(4, 4),
        mlp_ratio=(3.0, 3.0, 1.0),
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=nn.LayerNorm,
        # ---- NEW ----
        use_t2t: bool = False,
        t2t_token_dim: int = 64,
        t2t_specs_small=None,
        t2t_specs_large=None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_branches = len(patch_size)
        self.embed_dim = list(embed_dim)
        self.use_t2t = use_t2t

        if t2t_specs_small is None:
            # fine-scale branch, less downsampling
            t2t_specs_small = ((7, 4), (3, 1), (3, 1))
        if t2t_specs_large is None:
            # coarse-scale branch, more downsampling
            t2t_specs_large = ((7, 4), (3, 2), (3, 2))

        # ---------------- Patch embedding backend ---------------- #

        if not use_t2t:
            # original conv-based multi-branch PatchEmbed
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            num_patches = self.patch_embed.num_patches_per_branch  # list
        else:
            # two independent T2T embedders, similar to CrossViT_T2T_160
            self.patch_embed = nn.ModuleList(
                [
                    T2TPatchEmbed(
                        img_size=img_size,
                        in_chans=in_chans,
                        embed_dim=embed_dim[0],
                        token_dim=t2t_token_dim,
                        t2t_specs=t2t_specs_small,
                        drop=drop_rate,
                    ),
                    T2TPatchEmbed(
                        img_size=img_size,
                        in_chans=in_chans,
                        embed_dim=embed_dim[1],
                        token_dim=t2t_token_dim,
                        t2t_specs=t2t_specs_large,
                        drop=drop_rate,
                    ),
                ]
            )
            num_patches = [pe.num_patches for pe in self.patch_embed]

        # CLS + positional embeddings per branch
        self.cls_tokens = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1, d)) for d in self.embed_dim]
        )
        self.pos_embeds = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, 1 + n, d))
                for n, d in zip(num_patches, self.embed_dim)
            ]
        )
        self.pos_drop = nn.Dropout(drop_rate)

        # Multi-scale stages as before
        self.blocks = nn.ModuleList()
        for block_cfg in depth:
            self.blocks.append(
                MultiScaleBlock(
                    dim=self.embed_dim,
                    num_patches=num_patches,
                    depth=block_cfg,
                    num_heads=list(num_heads),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer,
                )
            )

        self.norm = nn.ModuleList([norm_layer(d) for d in self.embed_dim])
        self.head = nn.ModuleList(
            [
                nn.Linear(d, num_classes) if num_classes > 0 else nn.Identity()
                for d in self.embed_dim
            ]
        )

        for pe in self.pos_embeds:
            nn.init.trunc_normal_(pe, std=0.02)
        for ct in self.cls_tokens:
            nn.init.trunc_normal_(ct, std=0.02)

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        B = x.size(0)
        xs: List[torch.Tensor] = []

        # Tokenize per branch
        if self.use_t2t:
            # ModuleList of T2TPatchEmbed
            branch_tokens = [pe(x) for pe in self.patch_embed]  # list of (B, N, C)
        else:
            # Shared PatchEmbed returns list directly
            branch_tokens = self.patch_embed(x)  # list of (B, N, C)

        for pe, ct, tokens in zip(self.pos_embeds, self.cls_tokens, branch_tokens):
            cls = ct.expand(B, -1, -1)  # (B, 1, C)
            tokens = torch.cat([cls, tokens], dim=1)
            tokens = tokens + pe
            tokens = self.pos_drop(tokens)
            xs.append(tokens)

        # Multi-scale + cross-attention stages
        for blk in self.blocks:
            xs = blk(xs)

        # Final norm + CLS extraction
        xs = [norm(xb) for xb, norm in zip(xs, self.norm)]
        cls_feats = [xb[:, 0] for xb in xs]
        return cls_feats
