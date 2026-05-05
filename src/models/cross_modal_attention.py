# src/models/cross_modal_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioVisualMixer(nn.Module):
    """
    Dual-Reliability Auditory Attention Mixer for projected joint-attn features.
    """

    def __init__(self, visual_dim=256, audio_dim=256):
        super().__init__()
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim

        # Project (audio_joint + audio reliability) to visual_dim
        self.audio_proj = nn.Linear(audio_dim + 1, visual_dim)
        self.visual_proj = nn.Linear(visual_dim, visual_dim)
        self.visual_rel_gate = nn.Linear(1, visual_dim)  # [B, 1] -> [B, visual_dim]

    def forward(self, visual_feat, audio_feat, audio_reliability, visual_reliability):
        """
        visual_feat: [B, visual_dim, H, W] (projected)
        audio_feat: [B, audio_dim]          (projected, e.g. after joint attn)
        audio_reliability: [B, 1]
        visual_reliability: [B, 1]
        """
        B, C, H, W = visual_feat.shape

        # ------ Audio Query (reliability only here) ------
        audio_with_confidence = torch.cat([audio_feat, audio_reliability], dim=1)  # [B, audio_dim+1]
        audio_query = self.audio_proj(audio_with_confidence)  # [B, C]

        # ------ Visual Keys (reliability only here) ------
        visual_global = visual_feat.mean(dim=(2, 3))  # [B, C]
        visual_keys = self.visual_proj(visual_global)  # [B, C]

        visual_confidence_gates = torch.sigmoid(self.visual_rel_gate(visual_reliability))  # [B, C]
        visual_keys_reliable = visual_keys * visual_confidence_gates  # [B, C]

        # ------ Auditory Attention Weights (channel-wise) ------
        auditory_attention_scores = audio_query * visual_keys_reliable  # [B, C]
        auditory_attention_weights = torch.sigmoid(auditory_attention_scores)  # [B, C]
        attention_map = auditory_attention_weights.view(B, C, 1, 1)  # [B, C, 1, 1]

        # ------ Apply audio-guided enhancement (no extra reliability scalars) ------
        audio_guided_enhancement = visual_feat * attention_map
        enhanced_visual = visual_feat + audio_guided_enhancement
        return enhanced_visual