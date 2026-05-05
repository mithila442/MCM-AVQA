# src/models/late_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateFusion(nn.Module):
    """
    Simple late-fusion module for the baseline:
    - Projects visual and audio features to a common dimension
    - Concatenates them
    - Applies a small MLP to produce fused features
    """
    def __init__(self, visual_dim=1024, audio_dim=128, fusion_dim=1024):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)

        # concatenated size = 2 * fusion_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(2 * fusion_dim, fusion_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, visual_feat, audio_feat):
        """
        Args:
            visual_feat: [B, visual_dim]
            audio_feat:  [B, audio_dim]

        Returns:
            fused: [B, fusion_dim]
        """
        visual_proj = self.visual_proj(visual_feat)   # [B, D]
        audio_proj = self.audio_proj(audio_feat)      # [B, D]

        fused_input = torch.cat([visual_proj, audio_proj], dim=1)  # [B, 2D]
        fused = self.fusion_layer(fused_input)                     # [B, D]
        return fused