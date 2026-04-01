import torch
import torch.nn as nn
import torch.nn.functional as F


def _choose_groupnorm_vit(num_channels: int, max_groups: int =8) -> int:
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1

class ConvGNact2D(nn.Module):
    def __init__(self, in_channel, out_channels, dropout=0.0):
        super().__init__()
        g = _choose_groupnorm_vit(num_channels=out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g, out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout < 0 else nn.Identity(),
        )
    def forward(self, x: torch.Tensor) ->  torch.Tensor:
        return self.block(x)

class FusionBlock2D(nn.Module):
    """
    upsampled feature + skip > refined feature
    """
    def __init__(self, in_channel, out_channels, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNact2D(in_channel, out_channels, dropout=dropout),
            ConvGNact2D(out_channels, out_channels, dropout=dropout),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        print(f'before concat: {x.shape}')
        x = torch.cat([x, skip], dim=1)
        print(f'after concat: {x.shape}')
        return self.block(x)

class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = MLP(dim)
        
    def forward(self, t):
        if t.ndim == 4:
            t = t[:, 0, 0, 0]
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        
        print(f'expected t: [b, dim] | {t.shape}')
        return self.mlp(t)

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio, dropout=0.0):
        super().__init__()
        ratio = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, ratio)
        self.fc2 = nn.Linear(ratio, dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GeLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x

class AttentionMLP(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiHeadAttention(dim, num_heads, dropout=dropout, attn_drop=attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)
        
    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x

class MLPDecoder(nn.Module):
    """
    features: [f1, f2, f3, f4] (low to high)
    decode : high to low and output logits
    """

    def __init__(self, embed_dim, num_classes, dropout=0.0):
        super().__init__()
        # embed_dim_rev = [c4, c3, c2, c1]
        c4, c3, c2, c1 = embed_dim

        self.up43 = FusionBlock2D(c4 + c3, c3, dropout=dropout)
        self.up32 = FusionBlock2D(c3 + c2, c2, dropout=dropout)
        self.up21 = FusionBlock2D(c2 + c1, c1, dropout=dropout)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, feats, out_size):
        f1, f2, f3, f4 = feats


        x = F.interpolate(f4, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        x = self.up43(x, f3)

        x = F.interpolate(x, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.up32(x, f2)

        x = F.interpolate(x, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        x = x + self.up21(x, f1)
        x = x + self.ref21(x)

        logits = self.head(x)

        if out_size is not None:
            logits = F.interpolate(logits, size=out_size, mode='bilinear', align_corners=False)

        return logits

class ViT2DPatchEmbed(nn.Module):
    """
    transforms 2D medical volumes into sequences of tokens
    """

    def __init__(self, patch_size, in_channel, embed_dim, kernel=None, stride=None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels=in_channel, emded_dim=embed_dim, kernel_size=kernel, stride=stride, bias=False)

    def forward(self, x):
        # [B, dim, Ll, Ww] > [b, n ,d]
        x = self.proj(x)
        B, C, L, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)

        return tokens, (L, W), x


class PatchMerging2D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, feat):
        # feat : [B, C, L, W]
        feat = self.conv(feat)
        print(f'feat.shape: {feat.shape}') # [B, out, L/2, W/2]
        B, C, L, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)

        return tokens, (L, W), feat

class ViTEndPointDenoiser(nn.Module):
    """
    predict x0_hat = D_theta(x_t, t) in a direct way
    expected:
     - input : x_t [b, c, l, w], t [b] || [b, 1] || [b, 1, 1, 1]
     - output : x0_hat [b, c, l, W]
    """

    def __init__(self, img_size=256, patch=4, in_channel=1, dim=512, depth=8, num_heads=8, mlp_ratio=3.0, dropout=0.0, attn_drop=0.0):
        super().__init__()
        if img_size % patch != 0:
            raise ValueError ('patch * img_size = int')

        self.img_size = img_size
        self.patch = patch
        self.in_channel = in_channel
        self.grid = img_size // patch
        self.num_tokens = self.grid * self.grid

        self.patch_embed = ViT2DPatchEmbed(in_channel=in_channel, kerner_size=self.patch, stride=self.patch, bias=True)
        self.position = nn.Parameter(torch.zeros(1, self.num_tokens, dim))

        self.time = TimeEmbed(dim)

        self.blocks = nn.ModuleList(
            [AttentionMLP(
                dim=dim, num_heads=num_heads, cond_dim=dim, mlp_ratio=mlp_ratio, dropout=dropout
            ) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, ((patch**2*in_channel)))

        nn.init.trunc_normal_(self.position, std=.02)
    
    def forward(self, x_t, t):
        B = x_t.shape[0]
        cond = self.time(t)
        token = self.patch_embed(x_t).flatten(2).transpose(1, 2)
        token = token + self.position

        for blk in self.blocks:
            token = blk(token, cond)
        
        token = self.norm(token)
        out = self.head(token)

        out = out.view(B, self.num_tokens, self.in_channel, self.patch, self.patch)
        out = out.permute(0,2,1,3,4).contiguous()
        out = out.view(B, self.in_channel, self.grid, self.grid, self.patch, self.patch)
        out = out.permute(0,1,2,4,3,5).contiguous().view(B, self.in_channel, self.img_size, self.img_size)

        return out