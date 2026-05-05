# src/models/avm.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioVisualMixer(nn.Module):
    """
    Audio-Visual Mixer without confidence:
    uses audio features to modulate visual features via channel-wise attention.
    """

    def __init__(self, visual_dim=256, audio_dim=256):
        super().__init__()
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim

        # Project audio features to visual_dim (query)
        self.audio_proj = nn.Linear(audio_dim, visual_dim)
        # Project global visual features to visual_dim (keys)
        self.visual_proj = nn.Linear(visual_dim, visual_dim)

    def forward(self, visual_feat, audio_feat):
        """
        Args:
            visual_feat: [B, visual_dim, H, W]
            audio_feat:  [B, audio_dim]

        Returns:
            enhanced_visual: [B, visual_dim, H, W]
        """
        B, C, H, W = visual_feat.shape

        # ------ Audio query ------
        audio_query = self.audio_proj(audio_feat)           # [B, C]

        # ------ Visual keys (global pooled) ------
        visual_global = visual_feat.mean(dim=(2, 3))        # [B, C]
        visual_keys = self.visual_proj(visual_global)       # [B, C]

        # ------ Channel-wise attention weights ------
        attn_scores = audio_query * visual_keys             # [B, C]
        attn_weights = torch.sigmoid(attn_scores)           # [B, C]
        attention_map = attn_weights.view(B, C, 1, 1)       # [B, C, 1, 1]

        # ------ Apply audio-guided enhancement ------
        audio_guided = visual_feat * attention_map
        enhanced_visual = visual_feat + audio_guided        # residual

        return enhanced_visual