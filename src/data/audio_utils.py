# src/data/audio_utils.py

import torch
import torchaudio
import torchaudio.transforms as T
from moviepy import VideoFileClip
import numpy as np
import librosa
import numpy as np
import tempfile
import os

def extract_audio_from_video(video_path, output_sr=16000):
    """
    Extract high-fidelity mono audio from a video file.
    Steps:
        - Demux audio using moviepy (writes to a temp file).
        - Load original audio using torchaudio for best decoding precision.
        - Resample to output_sr using high-quality torchaudio resampler.
        - Mono conversion via mean (if stereo).
    Returns:
        - waveform (Tensor): [1, T], mono float32 in [-1,1]
        - sample_rate (int): output_sr
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        try:
            # Write audio track to wav using moviepy (no resampling)
            clip = VideoFileClip(video_path)
            audio = clip.audio
            if audio is None:
                raise ValueError(f"No audio stream found in video: {video_path}")
            audio.write_audiofile(tmp_wav.name, fps=audio.fps, codec='pcm_s16le')
            del clip

            # Load using torchaudio for precision
            waveform, sr = torchaudio.load(tmp_wav.name)
            # Stereo to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            # Resample if needed
            if sr != output_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=output_sr)
                waveform = resampler(waveform)
            # RMS normalize (optional): uncomment the next two lines to normalize
            # rms = waveform.pow(2).mean().sqrt().item()
            # if rms > 0: waveform = waveform / rms

            return waveform, output_sr
        finally:
            os.remove(tmp_wav.name)


def load_audio(audio_path, target_sr=16000):
    """
    Load an audio file (.wav) and optionally resample it.
    Returns:
        - waveform (Tensor): [1, T], mono audio
        - sample_rate (int)
    """
    waveform, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
    return waveform.mean(0, keepdim=True), target_sr


def audio_to_log_melspectrogram(waveform, sample_rate, n_mels=64, n_fft=400, hop_length=160):
    """
    Converts raw waveform to a log-mel spectrogram.
    Matches configs for AST/VGGish-style audio input.
    Returns:
        - log_mel_spec (Tensor): [F, T]
    """
    mel_spec_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )
    mel_spec = mel_spec_transform(waveform)  # [1, F, T]
    log_mel_spec = torch.log10(mel_spec + 1e-10)
    return log_mel_spec.squeeze(0)  # [F, T]


def preprocess_audio(audio_path, target_sr=16000):
    """
    Loads a local .wav file and converts it to a log-mel spectrogram.
    """
    waveform, sr = load_audio(audio_path, target_sr=target_sr)
    return audio_to_log_melspectrogram(waveform, sr)


def preprocess_audio_from_video(video_path, target_sr=16000):
    """
    End-to-end audio preprocessing from a video file.
    Extracts audio and returns log-mel spectrogram.
    """
    waveform, sr = extract_audio_from_video(video_path, output_sr=target_sr)
    return audio_to_log_melspectrogram(waveform, sr)

def extract_waveform_from_video(video_path, target_sr=16000):
    audio, sr = librosa.load(video_path, sr=target_sr)
    return audio  # shape: [samples]