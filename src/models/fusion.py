# src/models/fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityFusion(nn.Module):
    def __init__(self, visual_dim=1024, audio_dim=128, fusion_dim=1024):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)

        # BEFORE: fusion_dim * 2 + 2 (when concat had visual, audio, vis_rel, aud_rel)
        # NOW: only visual_scaled and audio_scaled → size 2D
        self.audio_enhance_gate = nn.Linear(fusion_dim * 2, fusion_dim)

        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, visual_feat, audio_feat, visual_reliability, audio_reliability):
        visual_proj = self.visual_proj(visual_feat)  # [B, D]
        audio_proj = self.audio_proj(audio_feat)  # [B, D]

        # 1) Scale each modality by its reliability
        visual_scaled = visual_proj * visual_reliability  # [B, D]
        audio_scaled = audio_proj * audio_reliability  # [B, D]

        # 2) Learn audio gating from ONLY the reliability-scaled features
        concat = torch.cat([visual_scaled, audio_scaled], dim=1)  # [B, 2D]
        audio_gate = torch.sigmoid(self.audio_enhance_gate(concat))  # [B, D]

        # 3) Visual-dominant fusion with reliability-aware audio enhancement
        fused = visual_scaled + audio_gate * audio_scaled
        fused = self.fusion_layer(fused)
        return fused
