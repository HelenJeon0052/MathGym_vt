import torch
import torch.nn as nn

from models.decoder import MLPDecoder
from models.encoder import HierarchicalEncoder3D

"""
Residual Connections
feature map u : Represents the hidden state (the processed 3D features) at "time" (depth) t
P(x) : act as the weight matrix || friction > how much of the previous state is retained
Q(x) : represents the external input || forcing function from specific 3D spatial features of the medical scan
"""


class ViT3DPatchEmbed(nn.Module):
    """
    transforms 3D medical volumes into sequences of tokens
    u = f(x, y, z)
    """

    def __init__(self, patch_size=4, in_channels=1, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels=in_channels, emded_dim=embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x):
        # [B, dim, Dd, Ll, Ww]
        x = self.proj(x)
        B, C, Dd, Ll, Ww = x.shape
        tokens = x.flatten(2).transpose(1, 2) # [B, T, C]

        return tokens, (Dd, Ll, Ww), x


class PatchMerging3D(nn.Module):
    """
    Downsample by 2
    Increase channels
    stride = 2 > why?
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, feat):
        # feat : [B, C, D, L, W]
        feat = self.conv(feat)
        print(f'feat.shape: {feat.shape}') # [B, out, D/2, L/2, W/2]
        B, C, D, L, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)

        return tokens, (D, L, W), feat


class NeuralODEBlock(nn.Module):
    """
    u' + P(x)u = Q(x)
    u' = f(u, t)
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.lin = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x(t+1) = x(t) + \int f(x(t)) dt
        return self.lin(torch.relu(x))


class Light3DVit(nn.Module):
    def __init__(self,
                 in_channels=4,
                 num_classes=3,
                 embed_dim=(48, 96, 192, 384),
                 depths=(2, 2, 2, 2),
                 sr_ratios=(4, 2, 1, 1),
                 block_type='sr',
                 ode_mode='strang',
                 ode_steps_attn=2,
                 ode_steps_mlp=1,
                 ode_steps_fric=1,
                 use_friction=True,
                 friction_position='mid',
                 patch_size=4):
        super().__init__()
        self.encoder = HierarchicalEncoder3D(
            in_channels=in_channels,
            embed_dim=list(embed_dim),
            depth=list(depths),
            sr_ratio=list(sr_ratios),
            mlp_ratio=4.0,
            dropout=0.0,
            attn_drop=0.0,
            block_type=block_type,
            ode_mode=ode_mode,
            ode_steps_attn=ode_steps_attn,
            ode_steps_mlp=ode_steps_mlp,
            ode_steps_fric=ode_steps_fric,
            use_friction=use_friction,
            friction_position=friction_position,
            patch_size=patch_size
        )
        self.decoder = MLPDecoder(list(embed_dim)[::-1], num_classes=num_classes)

    def forward(self, x):
        feats = self.encoder(x)
        out = self.decoder(feats)

        return out