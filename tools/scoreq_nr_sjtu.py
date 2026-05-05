import os
import glob
import numpy as np
import pandas as pd
import soundfile as sf
import tempfile
import scoreq

# -----------------------------
# PATHS
# -----------------------------
test_dir = "./dist_audio_sjtu"
out_csv = "scoreq_reliability_sjtu.csv"

# -----------------------------
# SCOREQ NR SETTINGS
# -----------------------------
DATA_DOMAIN = "natural"   # use "synthetic" only if your audio is TTS/VC
MODE = "nr"               # NR predicts MOS-like quality directly (higher = better)

# Keep inputs ~10–15s to avoid memory issues
CLIP_ENABLE = True
CLIP_MAX_SEC = 15.0
CLIP_STRATEGY = "start"   # "start" is safest for consistency

# -----------------------------
# HELPERS
# -----------------------------
def clip_wav_to_15s(in_path: str, max_sec: float = 15.0, strategy: str = "start") -> str:
    """Clip to max_sec if needed; returns path (original or temp wav)."""
    data, sr = sf.read(in_path, always_2d=True)  # [T, C]
    max_len = int(max_sec * sr)

    if data.shape[0] <= max_len:
        return in_path

    start = 0 if strategy == "start" else max(0, (data.shape[0] - max_len) // 2)
    end = start + max_len
    clipped = data[start:end, :]

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    sf.write(tmp_path, clipped, sr)
    return tmp_path


def minmax01(x: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0,1]. If constant, returns zeros."""
    x = np.asarray(x, dtype=np.float64)
    mn = np.min(x)
    mx = np.max(x)
    den = (mx - mn)
    if den == 0:
        return np.zeros_like(x)
    return (x - mn) / den

# -----------------------------
# INIT SCOREQ (NR MODE)
# -----------------------------
scoreq_model = scoreq.Scoreq(data_domain=DATA_DOMAIN, mode=MODE)

# -----------------------------
# SCORE FILES
# -----------------------------
test_files = sorted(glob.glob(os.path.join(test_dir, "*.wav")))
if len(test_files) == 0:
    raise RuntimeError(f"No .wav files found in test_dir: {test_dir}")

results = []
tmp_paths_to_cleanup = []

for test_path in test_files:
    video_name = os.path.splitext(os.path.basename(test_path))[0]

    test_used = test_path
    if CLIP_ENABLE:
        test_used = clip_wav_to_15s(test_path, CLIP_MAX_SEC, CLIP_STRATEGY)
        if test_used != test_path:
            tmp_paths_to_cleanup.append(test_used)

    try:
        pred_mos = float(scoreq_model.predict(test_path=test_used, ref_path=None))
    except Exception as e:
        print(f"[WARN] Failed scoring {os.path.basename(test_path)}: {e}")
        continue

    results.append({"filename": video_name, "scoreq_nr": pred_mos})
    print(f"{os.path.basename(test_path)} -> SCOREQ_NR: {pred_mos:.4f}")

# Cleanup temp clipped files
for p in set(tmp_paths_to_cleanup):
    try:
        os.remove(p)
    except Exception:
        pass

if len(results) == 0:
    raise RuntimeError("No files were scored. Check your wavs and SCOREQ setup.")

df = pd.DataFrame(results)

# -----------------------------
# RELIABILITY = normalized SCOREQ prediction
# -----------------------------
pred = df["scoreq_nr"].astype(float).to_numpy()
df["scoreq_nr_01"] = minmax01(pred)
df["scoreq_reliability"] = np.clip(df["scoreq_nr_01"], 0.15, 1.0)

# -----------------------------
# SAVE (ONLY video name + reliability)
# -----------------------------
df[["filename", "scoreq_reliability"]].to_csv(out_csv, index=False)
print(f"\n✅ Saved reliability CSV: {out_csv}")