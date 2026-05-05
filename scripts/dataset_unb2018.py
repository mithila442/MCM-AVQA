# scripts/datasets.py

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchaudio
import csv
import os
import random
import numpy as np
from torch.utils.data import Dataset
import cv2
from src.data.audio_utils import extract_audio_from_video
import logging

# ========== SUPPRESS MOVIEPY & FFMPEG LOGGING ==========
logging.getLogger('moviepy').setLevel(logging.CRITICAL)
logging.getLogger('moviepy.video.io.ffmpeg_reader').setLevel(logging.CRITICAL)
os.environ['FFMPEG_LOGLEVEL'] = 'quiet'

from contextlib import contextmanager


@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr."""
    import sys
    import os

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        devnull = open(os.devnull, 'w')
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()


def sample_linspace_frames_from_video(video_path: str, num_frames: int = 8, size: int = 224) -> torch.Tensor:
    """
    Decode frames from AVI and sample num_frames evenly spaced.
    Returns: Tensor [num_frames, 3, size, size] float32 in [0,1]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return torch.zeros(num_frames, 3, size, size)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames is None or total_frames <= 0:
        total_frames = 0

    if total_frames > 0:
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int).tolist()
    else:
        indices = list(range(num_frames))

    frames = []
    last_good = None

    for idx in indices:
        if total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ok, frame = cap.read()
        if not ok or frame is None:
            frames.append(None)
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        last_good = frame
        frames.append(frame)

    cap.release()

    # If everything failed
    if all(f is None for f in frames):
        return torch.zeros(num_frames, 3, size, size)

    # Fill missing frames using last available / first available
    first_avail = next((f for f in frames if f is not None), None)
    if first_avail is None:
        return torch.zeros(num_frames, 3, size, size)

    fixed = []
    prev = first_avail
    for f in frames:
        if f is None:
            f = prev
        else:
            prev = f
        fixed.append(f)

    arr = np.stack(fixed, axis=0).astype(np.float32) / 255.0  # [T,H,W,C] in [0,1]
    video_frames = torch.from_numpy(arr).permute(0, 3, 1, 2)  # [T,3,H,W]
    return video_frames


class AVQADataset(Dataset):
    def __init__(self,
                 video_dir,
                 mos_csv_path,
                 scoreq_reliability_csv_path,
                 train=False):
        """
        Args:
            video_dir (str): Directory containing .avi video files
            mos_csv_path (str): Path to UnB-AVQ-2018-Experiment3.csv
            scoreq_reliability_csv_path (str): Path to scoreq_reliability.csv
            train (bool): Whether to apply data augmentation
        """
        self.video_dir = video_dir
        self.train = train

        # ========== Find all .avi video files ==========
        self.video_files = sorted([
            os.path.join(video_dir, f) for f in os.listdir(video_dir)
            if f.endswith(".avi") and os.path.isfile(os.path.join(video_dir, f))
        ])
        print(f"✅ Found {len(self.video_files)} .avi files in {video_dir}")
        print(f"   Example: {os.path.basename(self.video_files[0]) if self.video_files else 'N/A'}")

        # ========== Load Mqs (Mean Quality Score) ==========
        self.quality_map = {}
        with open(mos_csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                testfile = row.get('testFile')
                if not testfile:
                    continue
                testfile = testfile.strip()
                try:
                    mqs = float(row['Mqs'])
                    if np.isfinite(mqs):
                        self.quality_map[testfile] = mqs
                    else:
                        print(f"⚠️  Skipping {testfile}: Mqs is non-finite ({mqs})")
                except (ValueError, KeyError):
                    print(f"⚠️  Skipping row with invalid Mqs for {testfile}")
                    continue

        print(f"✅ Loaded Mqs for {len(self.quality_map)} samples from {mos_csv_path}")

        # ========== Load ScoreQ Audio Reliability ==========
        self.reliability_map = {}
        with open(scoreq_reliability_csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                filename = row.get('filename')
                if not filename:
                    continue
                filename = filename.strip()
                try:
                    reliability = float(row['scoreq_reliability'])
                    if np.isfinite(reliability):
                        self.reliability_map[filename] = reliability
                    else:
                        self.reliability_map[filename] = 0.5
                except (ValueError, KeyError):
                    print(f"⚠️  Skipping row with invalid reliability for {filename}")
                    self.reliability_map[filename] = 0.5

        print(f"✅ Loaded ScoreQ reliability for {len(self.reliability_map)} samples from {scoreq_reliability_csv_path}")
        print(f"   Example entry: {list(self.reliability_map.items())[0] if self.reliability_map else 'N/A'}")

        # ========== Match video files to CSV entries ==========
        self.video_files_filtered = []
        for vf in self.video_files:
            basename_with_ext = os.path.basename(vf)
            video_name = basename_with_ext.replace('.avi', '').strip()
            if video_name in self.quality_map and video_name in self.reliability_map:
                self.video_files_filtered.append(vf)

        print(f"\n✅ Matched {len(self.video_files_filtered)} out of {len(self.video_files)} videos")
        print(f"   Videos have both Mqs (quality) and ScoreQ reliability")

        if len(self.video_files_filtered) == 0:
            print(f"❌ ERROR: No videos found with both quality and reliability data!")
            print(f"   Quality map keys (sample): {list(self.quality_map.keys())[:3]}")
            print(f"   Reliability map keys (sample): {list(self.reliability_map.keys())[:3]}")
            raise ValueError("❌ No videos found with both quality and reliability data!")

        self.video_files = self.video_files_filtered

        # ========== Video augmentation pipeline ==========
        self.video_transform = T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomRotation(degrees=15),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])

        # ========== Audio augmentation (waveform-safe) ==========
        self.audio_transforms = [
            lambda x: x + 0.005 * torch.randn_like(x) if random.random() < 0.5 else x,
            lambda x: x * random.uniform(0.8, 1.2) if random.random() < 0.5 else x,
        ]

    def _vol_aug(self, waveform):
        """Apply volume augmentation to waveform."""
        if random.random() < 0.3:
            try:
                w = waveform.unsqueeze(0)  # [1, T]
                vol = torchaudio.transforms.Vol(random.uniform(0.5, 1.5))
                w = vol(w)
                return w.squeeze(0)
            except Exception:
                return waveform
        return waveform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video_name = os.path.basename(video_path).replace('.avi', '').strip()

        # Get quality and reliability (same logic as your datasets.py)
        mqs = self.quality_map.get(video_name, 1.0)
        normalized_quality = (mqs - 1.0) / 4.0
        audio_reliability = self.reliability_map.get(video_name, 0.5)

        if not np.isfinite(audio_reliability):
            print(f"❌ DATASET: Non-finite reliability for {video_name}: {audio_reliability}")
            audio_reliability = 0.5

        if not np.isfinite(normalized_quality):
            print(f"❌ DATASET: Non-finite quality for {video_name}: {normalized_quality}")
            normalized_quality = 0.0

        # ========== Decode + sample 8 frames from AVI (datasets2.py style) ==========
        try:
            video_frames = sample_linspace_frames_from_video(video_path, num_frames=8, size=224)
        except Exception as e:
            print(f"⚠️  Error decoding frames from {video_path}: {e}")
            video_frames = torch.zeros(8, 3, 224, 224)

        # DEBUG: Check video frames
        if torch.isnan(video_frames).any() or torch.isinf(video_frames).any():
            print(f"❌ DATASET: NaN/Inf in video_frames for {video_name}")
            video_frames = torch.zeros(8, 3, 224, 224)

        # For validation / test, ensure correct size
        if not self.train:
            if video_frames.shape[2] != 224 or video_frames.shape[3] != 224:
                video_frames = F.interpolate(
                    video_frames.float(), size=(224, 224),
                    mode="bilinear", align_corners=False
                )

        # Apply video augmentation if training
        if self.train:
            aug_frames = []
            for f in video_frames:
                f = self.video_transform((f * 255).byte())
                aug_frames.append(f)
            video_frames = torch.stack(aug_frames, dim=0)

        assert video_frames.dim() == 4

        # ========== Load audio (from video) ==========
        MAX_AUDIO_LEN = 672000  # 42s at 16 kHz

        try:
            with suppress_stdout_stderr():
                waveform, sr = extract_audio_from_video(video_path, output_sr=16000)

            if waveform.ndim == 2:
                waveform = waveform.mean(dim=0)
            waveform = waveform.float()
        except Exception as e:
            print(f"⚠️  Error loading audio from {video_path}: {e}")
            waveform = torch.zeros(MAX_AUDIO_LEN, dtype=torch.float32)

        # DEBUG: Check audio after extraction
        if torch.isnan(waveform).any() or torch.isinf(waveform).any():
            print(f"❌ DATASET: NaN/Inf in extracted waveform for {video_name}")
            waveform = torch.zeros(MAX_AUDIO_LEN, dtype=torch.float32)

        # Pad / truncate to fixed length [T]
        cur_len = waveform.shape[0]
        if cur_len > MAX_AUDIO_LEN:
            waveform = waveform[:MAX_AUDIO_LEN]
        elif cur_len < MAX_AUDIO_LEN:
            waveform = F.pad(waveform, (0, MAX_AUDIO_LEN - cur_len))

        # DEBUG: Check after padding
        if torch.isnan(waveform).any() or torch.isinf(waveform).any():
            print(f"❌ DATASET: NaN/Inf in padded waveform for {video_name}")
            waveform = torch.zeros(MAX_AUDIO_LEN, dtype=torch.float32)

        # Safety: remove NaNs / infs
        if not torch.isfinite(waveform).all():
            print(f"⚠️  Non-finite samples in waveform for {video_name}, replacing with zeros")
            waveform = torch.zeros_like(waveform)

        # Optional waveform augmentations
        if self.train:
            for aug_idx, aug in enumerate(self.audio_transforms):
                try:
                    waveform_before = waveform.clone()
                    waveform = aug(waveform)

                    if torch.isnan(waveform).any() or torch.isinf(waveform).any():
                        print(f"❌ DATASET: Augmentation {aug_idx} produced NaN/Inf for {video_name}")
                        waveform = waveform_before
                except Exception as e:
                    print(f"⚠️  Audio augmentation {aug_idx} failed for {video_name}: {e}")

        # Final sanity check
        assert torch.isfinite(video_frames).all(), "Video frames contain non-finite values"
        assert torch.isfinite(waveform).all(), "Waveform contains non-finite values"
        assert torch.isfinite(torch.tensor(audio_reliability)), "Audio reliability is non-finite"
        assert torch.isfinite(torch.tensor(normalized_quality)), "Normalized quality is non-finite"

        return (
            video_frames,
            waveform,
            torch.tensor([audio_reliability], dtype=torch.float32),
            normalized_quality,
            video_name
        )

