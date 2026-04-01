import torch
import torch.nn as nn

from models.attention import AttentionMLP, SRtransformerBlock3D, AttentionField3D, MLPField, FrictionField
from models.splitting import SplitODEBlock
from models.vit_3d import ViT3DPatchEmbed, PatchMerging3D


class HierarchicalEncoder3D(nn.Module):
    """
    returns :
     - 3D feature maps
     - [f1, f2, f3, f4]
    """
    def __init__(self,
                 in_channels,
                 embed_dim,
                 depth,
                 sr_ratio,
                 num_heads=None,
                 mlp_ratio=3.0,
                 dropout=0.0,
                 attn_drop=0.0,
                 block_type='sr',
                 ode_steps_attn=2,
                 ode_steps_mlp=1,
                 ode_steps_fric=1,
                 use_friction=True,
                 friction_position='mid',
                 patch_size=4):
        super().__init__()
        assert len(embed_dim) == len(depth) == len(sr_ratio)
        self.num_stages = len(embed_dim)
        if num_heads is None:
            num_heads = [max(1, d // 64) for d in embed_dim]

        self.patch_embed = ViT3DPatchEmbed(in_channels, embed_dim[0], patch_size=patch_size)

        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()

        for i in range(self.num_stages):
            dim = embed_dim[i]
            hd = num_heads[i]
            sr = sr_ratio[i]
            dth = depth[i]

            

            stage_blocks = nn.ModuleList()

            
            stage_blocks.append(AttentionMLP(dim, 1 << i, sr))

            for _ in range(dth):
                if block_type == 'sr':
                    stage_blocks.append(SRtransformerBlock3D(dim, num_heads=hd, sr_ratio=sr, mlp_ratio=mlp_ratio, dropout=dropout, attn_drop=attn_drop))
                else:
                    attn_field = AttentionField3D(dim, num_heads=hd, sr_ratio=sr, dropout=dropout, attn_drop=attn_drop)
                    mlp_field = MLPField(dim, mlp_ratio=mlp_ratio, dropout=dropout)
                    fric_field = FrictionField(dim)
                    stage_blocks.append(
                        SplitODEBlock(
                            attn_field=attn_field,
                            mlp_field=mlp_field,
                            fric_field=fric_field,
                            steps_attn=ode_steps_attn,
                            steps_mlp=ode_steps_mlp,
                            steps_fric=ode_steps_fric,
                            use_friction=use_friction,
                            friction_position=friction_position,
                        )
                    )
            self.stages.append(stage_blocks)

            if i < self.num_stages -1 :
                self.downs.append(PatchMerging3D(embed_dim[i], embed_dim[i+1]))

    def _tokens_to_feat(self, tok, grid):
        B, N, C = tok.shape
        D, L, W = grid

        assert N == D * L * W, (f'Token count mismatch: {N}, {grid} equivalent to {D*L*W}')
    
        feat = tok.transpose(1, 2).reshape(B, C, D, L, W)
        return feat
    
    def forward(self, x):

        patch_out = self.patch_embed(x)
        feats = []


        if isinstance(patch_out, (tuple, list)) and len(patch_out) == 3:
            tok, grid, feat = patch_out
        elif isinstance(patch_out, (tuple, list)) and len(patch_out) == 2:
            tok, grid = patch_out
            feat = self._tokens_to_feat(tok, grid) 
        else:
            raise ValueError('ViTPatchEmbed must contain returns (tok, grid, feat) | (tok, grid)')

        for i, stage_blocks in enumerate(self.stages):
            if i > 0:
                down = self.downs[i -1](feat)

                if isinstance(down, (tuple, list)) and len(down) == 3:
                    tok, grid, feat = down
                elif isinstance(down, (tuple, list)) and len(down) == 2:
                    tok, grid = down
                    feat = self._tokens_to_feat(tok, grid)
                else:
                    raise ValueError('PatchingMerge must contain returns (tok, grid, feat) | (tok, grid)')
            
            for blk in stage_blocks:
                tok = blk(tok, grid)

            feat = self._tokens_to_feat(tok, grid)
            feats.append(feat)
        

        feat_last = feats[-1]
        print([f.shape for f in feats])

        return feat_last, feats