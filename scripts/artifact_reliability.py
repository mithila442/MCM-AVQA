# scripts/artifact_reliability.py

import torch
import torch.nn as nn
import sys
sys.path.append('/home/elx12/MCM-AVQA') #replace with your home directory

from src.models.mvad_wrapper import MVADFromCheckpoint

class Temporal1DConv(nn.Module):
    def __init__(self, in_dim, kernel_size=3):
        super().__init__()
        self.conv1d = nn.Conv1d(in_dim, in_dim, kernel_size, 
                               padding=kernel_size//2, groups=in_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):  # x: [B, T, D]
        x = x.permute(0, 2, 1)  # [B, D, T]
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # [B, T, D]
        return x

class CheckpointBasedReliabilityModel(nn.Module):
    def __init__(self, checkpoint_path, num_heads=3, hidden_dim=16):
        super().__init__()
        # Change: Only create mvad_model if checkpoint provided
        if checkpoint_path is None:
            self.mvad_model = None
        else:
            self.mvad_model = MVADFromCheckpoint(checkpoint_path)
        
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(10, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            for _ in range(num_heads)
        ])
        self.temporal_plugin = Temporal1DConv(in_dim=10)
        self.combiner = nn.Sequential(
            nn.Linear(num_heads, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, video_frames):
        if self.mvad_model is None:
            # Return dummy reliability: all ones, and fake artifact probs (zeros)
            B, T, C, H, W = video_frames.shape
            return (
                torch.ones(B, 1, device=video_frames.device),
                torch.zeros(B, T, 10, device=video_frames.device)
            )

        # ---- Rest is unchanged, real logic ----
        B, T, C, H, W = video_frames.shape
        device = video_frames.device
        artifact_probs_list = []
        for t in range(T):
            frame_t = video_frames[:, t:t+1]
            with torch.no_grad():
                artifact_dict = self.mvad_model(frame_t)
            frame_probs = torch.stack([
                artifact_dict[name] for name in self.mvad_model.artifact_names
            ], dim=1)
            artifact_probs_list.append(frame_probs)
        artifact_probs = torch.stack(artifact_probs_list, dim=1)
        x = self.temporal_plugin(artifact_probs)
        head_outputs = [head(x) for head in self.heads]
        combined = torch.cat(head_outputs, dim=-1)
        reliabilities = self.combiner(combined)
        visual_reliability = reliabilities.mean(dim=1)
        return visual_reliability, artifact_probs
