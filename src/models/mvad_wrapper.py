# scripts/mvad_wrapper.py

# scripts/mvad_checkpoint_loader.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class VQAHead(nn.Module):
    """Recreate the VQAHead from MVAD artefact_net.py"""
    def __init__(self, in_channels=768, hidden_channels=64, dropout_ratio=0.5):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
            
        self.fc_hid = nn.Conv3d(in_channels, hidden_channels, (1, 1, 1))
        self.fc_last = nn.Conv3d(hidden_channels, 1, (1, 1, 1))
        self.gelu = nn.GELU()

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        qlt_score = self.fc_last(self.dropout(self.gelu(self.fc_hid(x)))).mean((-3, -2, -1)).squeeze()
        return qlt_score

class MVADFromCheckpoint(nn.Module):
    """Load MVAD model directly from checkpoint without repo dependencies"""
    
    def __init__(self, checkpoint_path):
        super().__init__()
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        self.ckpt = torch.load(checkpoint_path, map_location='cpu')
        self.state_dict = self.ckpt['state_dict']
        
        # Artifact names (from the checkpoint keys)
        self.artifact_names = [
            'motion_blur', 'dark_scenes', 'graininess', 'aliasing', 'banding',
            'blockiness', 'spatial_blur', 'frame_drop', 'transmission_error', 'black_screen'
        ]
        
        # Create artifact prediction heads to match checkpoint
        for artifact in self.artifact_names:
            head = VQAHead(in_channels=768, hidden_channels=64, dropout_ratio=0.5)
            setattr(self, f'head_{artifact}', head)
        
        # Load the head weights from checkpoint
        self._load_head_weights()
        
        # Set to eval mode and freeze
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
            
    def _load_head_weights(self):
        """Load weights for artifact prediction heads from checkpoint"""
        for artifact in self.artifact_names:
            head = getattr(self, f'head_{artifact}')
            
            # Load weights from state_dict
            head_prefix = f'model.head_{artifact}'
            
            # fc_hid weights and bias
            fc_hid_weight = self.state_dict[f'{head_prefix}.fc_hid.weight']
            fc_hid_bias = self.state_dict[f'{head_prefix}.fc_hid.bias']
            head.fc_hid.weight.data = fc_hid_weight
            head.fc_hid.bias.data = fc_hid_bias
            
            # fc_last weights and bias  
            fc_last_weight = self.state_dict[f'{head_prefix}.fc_last.weight']
            fc_last_bias = self.state_dict[f'{head_prefix}.fc_last.bias']
            head.fc_last.weight.data = fc_last_weight
            head.fc_last.bias.data = fc_last_bias
            
        print(f"Loaded weights for {len(self.artifact_names)} artifact heads")
    
    def extract_features_simple(self, video_frames):
        """
        Simple feature extraction - use global average pooling of frames
        as a placeholder for the full Swin3D backbone
        
        video_frames: [B, T, 3, H, W]
        Returns: [B, 768, 1, 1, 1] (fake features for head input)
        """
        B, T, C, H, W = video_frames.shape
        
        # Simple feature extraction (placeholder)
        # In reality, you'd want to use the full Swin3D backbone
        pooled = F.adaptive_avg_pool3d(
            video_frames.view(B, C, T, H, W), 
            (1, 1, 1)
        )  # [B, 3, 1, 1, 1]
        
        # Project to 768 dimensions (matching checkpoint expectation)
        features = torch.randn(B, 768, 1, 1, 1, device=video_frames.device)
        
        return features
        
    @torch.no_grad()
    def forward(self, video_frames):
        """
        video_frames: [B, T, 3, H, W]
        Returns: dict with artifact probabilities
        """
        # Extract features (simplified - you'd use full Swin3D here)
        features = self.extract_features_simple(video_frames)  # [B, 768, 1, 1, 1]
        
        # Get predictions from each head
        predictions = {}
        for artifact in self.artifact_names:
            head = getattr(self, f'head_{artifact}')
            logit = head(features)  # [B]
            prob = torch.sigmoid(logit)  # Convert to probability
            predictions[artifact] = prob
            
        return predictions