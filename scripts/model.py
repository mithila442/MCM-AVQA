# scripts/enhanced_model.py

import torch
import torch.nn as nn
import sys
import os

sys.path.append('/home/elx12/MCM-AVQA') #replace with your home directory

from src.models.swin_transformer import SwinTransformerBackbone
from src.models.vggish_backbone import VGGishBackbone
from src.models.cross_modal_attention import AudioVisualMixer
from src.models.fusion import ModalityFusion
from artifact_reliability import CheckpointBasedReliabilityModel


class EnhancedAVQAWithCheckpoint(nn.Module):
    def __init__(self, swin_cfg, attention_cfg, fusion_cfg, visual_reliability_cfg):
        """
        Args:
            swin_cfg (dict): Swin configuration from YAML
            attention_cfg (dict): Cross-modal attention config from YAML
            fusion_cfg (dict): Fusion configuration from YAML
            visual_reliability_cfg (dict): Visual reliability model config from YAML
        """
        super().__init__()

        # ============ Extract Config Values ============
        swin_checkpoint = swin_cfg.get('checkpoint_path')
        visual_dim = attention_cfg.get('visual_dim', 768)
        audio_dim = attention_cfg.get('audio_dim', 128)
        hidden_dim = attention_cfg.get('hidden_dim', 512)
        fusion_feature_dim = fusion_cfg.get('feature_dim', 512)

        # Visual reliability config
        rel_checkpoint = visual_reliability_cfg.get('checkpoint_path')
        rel_num_heads = visual_reliability_cfg.get('num_heads', 3)
        rel_hidden_dim = visual_reliability_cfg.get('hidden_dim', 16)

        # ============ Backbones ============
        self.visual_backbone = SwinTransformerBackbone(
            checkpoint_path=swin_checkpoint
        )
        self.audio_backbone = VGGishBackbone()

        # ============ Dimension Reductions ============
        self.visual_reduction = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj_to_mixer = nn.Linear(audio_dim, hidden_dim)

        # ============ AudioVisualMixer (spatial stage) ============
        self.mixer = AudioVisualMixer(
            visual_dim=hidden_dim,
            audio_dim=hidden_dim
        )

        # ============ ModalityFusion (global stage) ============
        self.fusion = ModalityFusion(
            visual_dim=hidden_dim,
            audio_dim=audio_dim,
            fusion_dim=fusion_feature_dim
        )

        # ============ Quality Regressor ============
        self.regressor = nn.Linear(fusion_feature_dim, 1)

        # ============ Reliability Model ============
        self.visual_reliability_model = CheckpointBasedReliabilityModel(
            checkpoint_path=rel_checkpoint,
            num_heads=rel_num_heads,
            hidden_dim=rel_hidden_dim
        )

    def forward(self, video_frames, waveform, audio_reliability):
        """
        Args:
            video_frames: [B, T, 3, H, W]
            waveform: [B, audio_len]
            audio_reliability: [B, 1]

        Returns:
            dict with keys: prediction, visual_reliability, audio_reliability, artifact_probs
        """
        B, T, C, H, W = video_frames.shape

        # ============ Audio Processing ============
        audio_feat = self.audio_backbone(waveform)  # [B, 128]

        if audio_feat.dim() == 1:
            # If 1D (single sample), unsqueeze to [1, feature_dim]
            audio_feat = audio_feat.unsqueeze(0)

        if audio_feat.dim() > 2:
            # If more than 2D, flatten everything except first dim
            audio_feat = audio_feat.view(audio_feat.shape[0], -1)

        # Now audio_feat is [N, feature_dim] where N might not equal B
        current_batch_size = audio_feat.shape[0]
        current_feature_dim = audio_feat.shape[1]

        # Handle feature dimension: ensure it's exactly 128
        if current_feature_dim < 128:
            pad_size = 128 - current_feature_dim
            audio_feat = torch.nn.functional.pad(audio_feat, (0, pad_size), mode='constant', value=0.0)
        elif current_feature_dim > 128:
            audio_feat = audio_feat[:, :128]

        # Handle batch dimension: ensure it matches B
        if current_batch_size < B:
            # Repeat last sample to fill batch
            repeat_count = B - current_batch_size
            audio_feat = torch.cat([
                audio_feat,
                audio_feat[-1:].expand(repeat_count, -1)
            ], dim=0)
        elif current_batch_size > B:
            # Trim to batch size
            audio_feat = audio_feat[:B]

        # Final assertion
        assert audio_feat.shape == (B, 128), f"Audio feat shape mismatch: expected (B={B}, 128), got {audio_feat.shape}"

        # ============ Visual Reliability ============
        visual_reliability, artifact_probs = self.visual_reliability_model(video_frames)

        # ============ Per-Frame Processing with Mixer ============
        visual_feats = []

        for t in range(T):
            frame_t = video_frames[:, t]  # [B, 3, H, W]

            visual_feat_t = self.visual_backbone(frame_t)

            if visual_feat_t.dim() == 4:
                if visual_feat_t.shape[1] != 768:
                    visual_feat_t = visual_feat_t.permute(0, 3, 1, 2).contiguous()

            if visual_feat_t.shape[2] != 7 or visual_feat_t.shape[3] != 7:
                visual_feat_t = torch.nn.functional.adaptive_avg_pool2d(visual_feat_t, (7, 7))

            # Project visual from 768 to hidden_dim
            visual_feat_pool = visual_feat_t.mean(dim=(2, 3))
            visual_feat_hidden = self.visual_reduction(visual_feat_pool)

            visual_feat_spatial = visual_feat_hidden.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)

            # Project audio from 128 to hidden_dim
            audio_feat_hidden = self.audio_proj_to_mixer(audio_feat)

            # Apply mixer
            modulated_visual_t = self.mixer(
                visual_feat_spatial,
                audio_feat_hidden,
                audio_reliability,
                visual_reliability
            )

            visual_pool_t = modulated_visual_t.mean(dim=(2, 3))
            visual_feats.append(visual_pool_t)

        # ============ Temporal Aggregation ============
        visual_feats_stacked = torch.stack(visual_feats, dim=1)  # [B, T, hidden_dim]
        visual_temporal = visual_feats_stacked.mean(dim=1)  # [B, hidden_dim]

        # ============ ModalityFusion ============
        fused = self.fusion(
            visual_temporal,
            audio_feat,
            visual_reliability,
            audio_reliability
        )
        
        fused = torch.nn.functional.layer_norm(fused, normalized_shape=(fused.shape[-1],), eps=1e-05)

        # ============ Quality Prediction ============
        pred = self.regressor(fused)
        if pred.dim() > 1:
            pred = pred.squeeze(-1)

        return {
            'prediction': pred,
            'visual_reliability': visual_reliability,
            'audio_reliability': audio_reliability,
            'artifact_probs': artifact_probs
        }
