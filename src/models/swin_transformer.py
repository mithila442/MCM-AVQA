#src/models/swin_transformer.py
import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer

class SwinTransformerBackbone(nn.Module):
    def __init__(self, checkpoint_path='checkpoints/swin_small_patch4_window7_224.pth'):
        super().__init__()
        print(f"🔍 Initializing SwinTransformer directly from checkpoint:")
        print(f"   Checkpoint path: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get('model', checkpoint)
        patch_shape = state_dict['patch_embed.proj.weight'].shape
        print(f"   patch_embed.proj.weight shape: {patch_shape}")

        # Always construct official Swin-Small
        self.backbone = SwinTransformer(
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=0,
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
        )

        # Custom: Load only matching shapes
        model_state = self.backbone.state_dict()
        filtered_state = {}
        mismatched = []

        for key in state_dict:
            if key in model_state and state_dict[key].shape == model_state[key].shape:
                filtered_state[key] = state_dict[key]
            else:
                mismatched.append(key)
        print(f"   Loading {len(filtered_state)} / {len(model_state)} keys from checkpoint; {len(mismatched)} mismatched or missing.")

        self.backbone.load_state_dict(filtered_state, strict=False)
        print("   ⚠️ Some layers (typically classification or relative pos, or new layers in timm) may be randomly initialized.")

        self.output_channels = 768  # Swin-Small

    def forward(self, x):
        x = self.backbone.patch_embed(x)
        if hasattr(self.backbone, 'pos_drop'):
            x = self.backbone.pos_drop(x)
        for i, layer in enumerate(self.backbone.layers):
            x = layer(x)
        x = self.backbone.norm(x)
        if x.dim() == 3:
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        elif x.dim() == 4:
            if x.shape[-1] == self.output_channels:
                x = x.permute(0, 3, 1, 2).contiguous()
        return x