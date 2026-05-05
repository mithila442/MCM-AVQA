# scripts/evaluate_av_livesjtu.py

import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from thop import profile
from enhanced_model import EnhancedAVQAWithCheckpoint
from feat_weight import FeatureWeightingBaseline
from cwlf import CWLFBaseline
from datasets3 import LIVESJTUAVQADataset
import sys
sys.path.append('/home/elx12/mcm-avqa')


def four_pl_map(x, b1, b2, b3, b4):
    x = np.asarray(x, dtype=np.float64)
    return b2 + (b1 - b2) / (1.0 + np.exp((-x + b3) / (np.abs(b4) + 1e-12)))


def fit_4pl_mapping(pred, mos):
    pred = np.asarray(pred, dtype=np.float64)
    mos = np.asarray(mos, dtype=np.float64)
    p0 = [float(mos.max()), float(mos.min()), float(np.median(pred)), 0.2]
    lower = [float(mos.min()) - 0.25, float(mos.min()) - 0.25, float(pred.min()) - 10.0, 1e-3]
    upper = [float(mos.max()) + 0.25, float(mos.max()) + 0.25, float(pred.max()) + 10.0, 50.0]
    try:
        betas, _ = curve_fit(
            four_pl_map, pred, mos, p0=p0,
            bounds=(lower, upper),
            maxfev=20000
        )
        mapped = four_pl_map(pred, *betas)
        return mapped, betas, True
    except Exception:
        mm = (pred - pred.min()) / (pred.ptp() + 1e-12)
        mapped = mm * (mos.max() - mos.min()) + mos.min()
        return mapped, None, False


def rmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    # 2013-UnB-AVQ paths
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing Exp3 .avi videos (e.g., datasets/2013-UnB-AVQ/Exp3)")
    parser.add_argument("--mos_csv", type=str, required=True,
                        help="MOS CSV (e.g., datasets/2013-UnB-AVQ/UnB-AVQ-2013-Experiment3.csv)")
    parser.add_argument("--reliability_csv", type=str, required=True,
                        help="Reliability CSV with columns: filename, scoreq_reliability")

    # model ckpt
    parser.add_argument("--checkpoint", type=str, required=True)

    # test split indices created during training
    parser.add_argument("--test_indices", type=str, default="test_indices_enhanced.npy")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--artifact_ckpt", type=str, default=None,
                        help="Optional artifact reliability checkpoint, else internal default is used.")
    return parser.parse_args()

def compute_inference_flops(model, dataloader):
    model_cpu = model.cpu().eval()

    with torch.no_grad():
        video, waveform, audio_reliability, mos, names = next(iter(dataloader))

        # batch size 1 for per-sample FLOPs
        video = video[:1].float().cpu()
        waveform = waveform[:1].float().cpu()
        audio_reliability = audio_reliability[:1].float().cpu()

        flops, params = profile(
            model_cpu,
            inputs=(video, waveform, audio_reliability),
            verbose=False
        )

    return flops, params

def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.test_indices):
        raise FileNotFoundError(f"Test indices file not found: {args.test_indices}")
    test_indices = np.load(args.test_indices)

    # Dataset (NO augmentation for eval)
    base_dataset = dataset = LIVESJTUAVQADataset(
        video_dir="datasets/LIVE-SJTU_AVQA/Distorted",
        mos_xlsx_path="datasets/LIVE-SJTU_AVQA/MOS.xlsx",
        scoreq_reliability_csv_path="scoreq_reliability_sjtu.csv",
        train=False
    )

    test_dataset = Subset(base_dataset, test_indices)

    dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    # Model
    model = EnhancedAVQAWithCheckpoint(
        swin_cfg=config["swin"],
        attention_cfg=config["cross_modal_attention"],
        fusion_cfg=config["fusion"],
        visual_reliability_cfg=config["visual_reliability"],
        #checkpoint_path=args.artifact_ckpt
    ).to(device)
    
    # model = FeatureWeightingBaseline(
    #     swin_cfg=config["swin"],                          # ← FIX
    #     attention_cfg=config["cross_modal_attention"],    # ← FIX
    #     fusion_cfg=config["fusion"],                      # ← FIX
    #     visual_reliability_cfg=config["visual_reliability"] # ← FIX
    # ).to(device)
    
    # model= CWLFBaseline(
    #     swin_cfg=config["swin"],
    #     attention_cfg=config["cross_modal_attention"],
    #     fusion_cfg=config["fusion"],
    #     visual_reliability_cfg=config["visual_reliability"],
    # ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)

    # supports either {"model_state_dict": ...} or raw state_dict
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    flops, params = compute_inference_flops(model, dataloader)
    print(f"🚀 Inference FLOPs: {flops}")
    print(f"📦 Params: {params}")

    model = model.to(device)
    model.eval()

    all_preds, all_mos = [], []

    with torch.no_grad():
        for video, waveform, audio_reliability, mos, names in dataloader:
            video = video.to(device).float()
            waveform = waveform.to(device).float()
            audio_reliability = audio_reliability.to(device).float()
            mos = mos.to(device).float()

            outputs = model(video, waveform, audio_reliability)
            pred = outputs["prediction"].view(-1)

            all_preds.extend(pred.detach().cpu().numpy().tolist())
            all_mos.extend(mos.view(-1).detach().cpu().numpy().tolist())

    all_preds = np.array(all_preds, dtype=np.float64)
    all_mos = np.array(all_mos, dtype=np.float64)

    # Guard against constant vectors
    if len(np.unique(all_preds)) <= 1 or len(np.unique(all_mos)) <= 1:
        print("⚠️ Predictions or MOS are constant; correlations will be 0.")
        plcc_before = 0.0
        srocc = 0.0
    else:
        plcc_before = float(pearsonr(all_preds, all_mos)[0])
        srocc = float(spearmanr(all_preds, all_mos).correlation)

    mae_before = float(mean_absolute_error(all_mos, all_preds))
    rmse_before = rmse(all_mos, all_preds)

    mapped_preds, betas, ok = fit_4pl_mapping(all_preds, all_mos)

    if len(np.unique(mapped_preds)) <= 1 or len(np.unique(all_mos)) <= 1:
        plcc_after = 0.0
    else:
        plcc_after = float(pearsonr(mapped_preds, all_mos)[0])

    mae_after = float(mean_absolute_error(all_mos, mapped_preds))
    rmse_after = rmse(all_mos, mapped_preds)

    print("\nEvaluation Results (LIVE-SJTU Test Set):")
    print(f"SROCC (rank, invariant to mapping): {srocc:.4f}")
    print(f"PLCC  before mapping: {plcc_before:.4f} | RMSE: {rmse_before:.4f} | MAE: {mae_before:.4f}")
    print(f"PLCC  after  4PL   : {plcc_after:.4f} | RMSE: {rmse_after:.4f} | MAE: {mae_after:.4f}")
    if ok and betas is not None:
        b1, b2, b3, b4 = betas
        print(f"4PL betas: b1={b1:.4f}, b2={b2:.4f}, b3={b3:.4f}, b4={b4:.4f}")
    else:
        print("4PL fit failed; used min-max fallback for reporting.")

    # -------- plots (same as your original) --------
    mn, mx = float(all_mos.min()), float(all_mos.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(all_mos, all_preds, alpha=0.6, color="blue", label="raw predictions")
    plt.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Ideal: y=x')
    plt.xlabel("MOS (Ground Truth, normalized)")
    plt.ylabel("Predicted (raw)")
    plt.title(f"Predicted vs MOS — BEFORE 4PL\nPLCC={plcc_before:.3f}, RMSE={rmse_before:.3f}")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("scatter_livesjtu_pred_vs_mos_before4pl.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(all_mos, mapped_preds, alpha=0.6, color="teal", label="after 4PL")
    plt.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Ideal: y=x')
    plt.xlabel("MOS (Ground Truth, normalized)")
    plt.ylabel("Predicted (after 4PL mapping)")
    plt.title(f"Predicted vs MOS — AFTER 4PL\nPLCC={plcc_after:.3f}, RMSE={rmse_after:.3f}")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("scatter_livesjtu_pred_vs_mos_after4pl.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()