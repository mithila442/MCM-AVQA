# src/models/vggish_backbone.py

import torch
import torch.nn as nn
from torchvggish import vggish
from torchvggish.vggish_input import waveform_to_examples

class VGGishBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vggish()
        self.output_dim = 128  # Output of final VGGish layer

    def forward(self, waveform_batch):
        """
        Args:
            waveform_batch: Tensor of shape [B, num_samples] — mono 16kHz waveform
        Returns:
            Tensor [B, 128] — one mean VGGish feature vector per sample
        """
        device = waveform_batch.device
        batch_feats = []

        for w in waveform_batch:
            # Convert waveform to log mel spectrogram patches
            examples = waveform_to_examples(w.cpu().numpy(), sample_rate=16000)  # → [N, 96, 64]
            examples = examples.unsqueeze(1).squeeze(2).to(device).float()  # [N, 1, 96, 64]

            # Patch PCA tensors to correct device
            if hasattr(self.model, 'pproc'):
                self.model.pproc._pca_matrix = self.model.pproc._pca_matrix.to(device)
                self.model.pproc._pca_means = self.model.pproc._pca_means.to(device)

            with torch.no_grad():
                vgg_features = self.model(examples)  # [N, 128]

            mean_feat = vgg_features.mean(dim=0)  # [128]
            batch_feats.append(mean_feat)

        return torch.stack(batch_feats, dim=0)  # [B, 128]