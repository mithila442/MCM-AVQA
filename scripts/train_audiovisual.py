import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import sys
import logging
import sys
# Add path for enhanced model
sys.path.append('/home/elx12/MCM-AVQA') #replace with your home directory
from model import EnhancedAVQAWithCheckpoint
from feat_weight import FeatureWeightingBaseline
from cwlf import CWLFBaseline
from dataset_unb2018 import AVQADataset
from dataset_unb2013 import UNBAVQ2013Exp3Dataset
from dataset_livesjtu import LIVESJTUAVQADataset
from thop import profile
# ========== SUPPRESS MOVIEPY & FFMPEG LOGGING ==========
logging.getLogger('moviepy').setLevel(logging.CRITICAL)
logging.getLogger('moviepy.video.io.ffmpeg_reader').setLevel(logging.CRITICAL)
os.environ['FFMPEG_LOGLEVEL'] = 'quiet'


class PearsonCorrLoss(nn.Module):
    def __init__(self):
        super(PearsonCorrLoss, self).__init__()

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        
        # ✅ Add small epsilon for stability
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)
        
        numerator = torch.sum((pred - pred_mean) * (target - target_mean))
        
        pred_var = torch.sum((pred - pred_mean) ** 2)
        target_var = torch.sum((target - target_mean) ** 2)
        
        denominator = torch.sqrt(pred_var * target_var + 1e-8)  # ← Add epsilon here
        
        corr = numerator / (denominator + 1e-8)
        
        # ✅ Clamp correlation to valid range
        corr = torch.clamp(corr, min=-1.0, max=1.0)
        
        return 1.0 - corr


def calculate_correlations(predictions, targets):
    try:
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        if len(np.unique(predictions)) == 1 or len(np.unique(targets)) == 1:
            return 0.0, 0.0
        plcc, _ = pearsonr(predictions, targets)
        srocc, _ = spearmanr(predictions, targets)
        plcc = float(plcc) if not np.isnan(plcc) else 0.0
        srocc = float(srocc) if not np.isnan(srocc) else 0.0
        return plcc, srocc
    except Exception as e:
        print(f"⚠️ Correlation calculation failed: {e}")
        return 0.0, 0.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/model_config.yaml")
    parser.add_argument("--dataset", type=str, default="unb-av")
    # parser.add_argument("--video_dir", type=str, default="datasets/exp3",
    #                     help="Directory containing .avi video files")
    parser.add_argument("--video_dir", type=str,
                        default="datasets/2013-UnB-AVQ/Exp3")
    # parser.add_argument("--mos_csv", type=str, default="datasets/exp3/UnB-AVQ-2018-Experiment3.csv",
    #                     help="Path to MOS ground truth CSV")
    parser.add_argument("--mos_csv", type=str,
                        default="datasets/2013-UnB-AVQ/UnB-AVQ-2013-Experiment3.csv")
    # parser.add_argument("--reliability_csv", type=str, default="scoreq_audio_reliability.csv",
    #                     help="Path to ScoreQ audio reliability CSV")
    parser.add_argument("--reliability_csv", type=str, default="scoreq_reliability_sjtu.csv",
                        help="Path to ScoreQ audio reliability CSV")
    parser.add_argument("--checkpoint_path", type=str, default="visual_artifacts_ckpt.ckpt",
                        help="Path to checkpoint for artifact detection")
    return parser.parse_args()


def save_checkpoint(epoch, model, optimizer, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    try:
        model_state = model.state_dict()
    except TypeError:
        model_state = nn.Module.state_dict(model)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    checkpoint_path = os.path.join(save_dir, f"enhanced_checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"✅ Enhanced model checkpoint saved at epoch {epoch} to {checkpoint_path}")


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    nan_count = 0
    
    with torch.no_grad():
        for video, waveform, audio_reliability, mos, names in tqdm(dataloader, desc="Validating", leave=False):
            video = video.to(device)
            waveform = waveform.to(device)
            audio_reliability = audio_reliability.to(device)
            mos = mos.to(device)
            
            # DEBUG: Check inputs
            if torch.isnan(video).any() or torch.isnan(waveform).any() or torch.isnan(mos).any():
                print(f"❌ EVAL: NaN in validation inputs for {names}")
                nan_count += 1
                continue
            
            outputs = model(video, waveform, audio_reliability)
            pred = outputs['prediction']
            
            # DEBUG: Check output
            if torch.isnan(pred).any():
                print(f"❌ EVAL: Model output NaN for {names}")
                nan_count += 1
                continue
            
            loss = criterion(pred.squeeze(), mos.squeeze())
            
            # DEBUG: Check loss
            if torch.isnan(loss):
                print(f"❌ EVAL: Loss is NaN for {names}")
                nan_count += 1
                continue
            
            total_loss += loss.item()
            all_predictions.extend(pred.cpu().numpy().flatten())
            all_targets.extend(mos.cpu().numpy().flatten())
    
    if nan_count > 0:
        print(f"⚠️  EVAL: Skipped {nan_count} samples due to NaN")
    
    if len(all_predictions) == 0:
        print(f"❌ EVAL: No valid predictions!")
        return float('nan'), 0.0, 0.0, 0.0, 0.0
    
    avg_loss = total_loss / len(dataloader)
    plcc, srocc = calculate_correlations(all_predictions, all_targets)
    return avg_loss, plcc, srocc, 0.0, 0.0



def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, device, config):
    print("🚀 Enhanced AVQA Training Started:")

    # ========== Ensure model is float32 ==========
    model = model.to(device).float()

    patience = 300
    best_val_loss = float('inf')
    best_val_plcc = float('-inf')
    best_val_srocc = float('-inf')
    best_epoch = 0
    epochs_since_improv = 0
    best_model_state = None

    for epoch in range(config["epochs"]):
        model.train()
        total_train_loss = 0.0
        train_predictions = []
        train_targets = []
        correlation_weight = 0.15
        mse_criterion = nn.MSELoss()
        corr_criterion = PearsonCorrLoss()
        train_nan_count = 0

        for video, waveform, audio_reliability, mos, names in tqdm(train_dataloader, desc=f"Epoch {epoch + 1} [Train]"):
            # ========== Convert all to float32 ==========
            video = video.to(device).float()
            waveform = waveform.to(device).float()
            audio_reliability = audio_reliability.to(device).float()
            mos = mos.to(device).float()

            # ========== DEBUG: Check inputs ==========
            if torch.isnan(video).any():
                print(f"❌ TRAIN: NaN in video for {names}")
                train_nan_count += 1
                continue
            if torch.isnan(waveform).any():
                print(f"❌ TRAIN: NaN in waveform for {names}")
                train_nan_count += 1
                continue
            if torch.isnan(audio_reliability).any():
                print(f"❌ TRAIN: NaN in audio_reliability for {names}")
                train_nan_count += 1
                continue
            if torch.isnan(mos).any():
                print(f"❌ TRAIN: NaN in MOS for {names}")
                train_nan_count += 1
                continue

            optimizer.zero_grad()
            outputs = model(video, waveform, audio_reliability)
            pred = outputs['prediction'].float()

            # ========== DEBUG: Check model output ==========
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                print(f"❌ TRAIN: Model output NaN/Inf for {names}")
                print(f"   Visual reliability: {outputs['visual_reliability']}")
                print(f"   Audio reliability: {outputs['audio_reliability']}")
                train_nan_count += 1
                continue

            mse_loss = mse_criterion(pred.squeeze(), mos.squeeze())
            corr_loss = corr_criterion(pred.squeeze(), mos.squeeze())

            # ========== DEBUG: Check losses ==========
            if torch.isnan(mse_loss):
                print(f"❌ TRAIN: MSE Loss is NaN for {names}")
                print(f"   Pred: {pred.squeeze()}")
                print(f"   MOS: {mos.squeeze()}")
                train_nan_count += 1
                continue
            if torch.isnan(corr_loss):
                print(f"❌ TRAIN: Correlation Loss is NaN for {names}")
                print(f"   Pred min/max: {pred.min()}/{pred.max()}")
                print(f"   MOS min/max: {mos.min()}/{mos.max()}")
                train_nan_count += 1
                continue

            loss = mse_loss + correlation_weight * corr_loss

            if torch.isnan(loss):
                print(f"❌ TRAIN: Combined loss is NaN for {names}")
                train_nan_count += 1
                continue

            loss.backward()
            
            # ========== DEBUG: Check gradients ==========
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"❌ TRAIN: Gradient norm is NaN/Inf: {grad_norm}")
                train_nan_count += 1
                optimizer.zero_grad()  # Skip this batch
                continue

            optimizer.step()
            total_train_loss += loss.item()
            train_predictions.extend(pred.detach().cpu().numpy().flatten())
            train_targets.extend(mos.cpu().numpy().flatten())

        if train_nan_count > 0:
            print(f"⚠️  TRAIN: Encountered {train_nan_count} NaN batches this epoch")

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_dataloader) if len(train_dataloader) > 0 else float('nan')
        train_plcc, train_srocc = calculate_correlations(train_predictions, train_targets)
        
        # ========== DEBUG: Check training metrics ==========
        if not np.isfinite(avg_train_loss):
            print(f"❌ TRAIN: avg_train_loss is non-finite: {avg_train_loss}")
        
        val_loss, val_plcc, val_srocc, _, _ = evaluate_model(model, val_dataloader, criterion, device)
        
        # ========== DEBUG: Check validation metrics ==========
        if not np.isfinite(val_loss):
            print(f"❌ VAL: val_loss is non-finite: {val_loss}")
        if not np.isfinite(val_plcc):
            print(f"❌ VAL: val_plcc is non-finite: {val_plcc}")
        if not np.isfinite(val_srocc):
            print(f"❌ VAL: val_srocc is non-finite: {val_srocc}")
        
        print(
            f"[Epoch {epoch + 1:3d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.7f}")
        print(
            f"{'':>11} Train PLCC: {float(train_plcc):.4f} | Val PLCC: {float(val_plcc):.4f} | Train SROCC: {float(train_srocc):.4f} | Val SROCC: {float(val_srocc):.4f}")

        # Early stopping logic: improvement if val_loss goes down OR val_plcc up
        improved = (np.isfinite(val_loss) and val_loss < best_val_loss) or \
                   (np.isfinite(val_plcc) and val_plcc > best_val_plcc) or \
                   (np.isfinite(val_srocc) and val_srocc > best_val_srocc)
                   
        # 🔹 Periodic checkpoint every 15 epochs (regardless of improvement)
        if (epoch + 1) % 15 == 0:
            save_checkpoint(epoch + 1, model, optimizer)
            print(f"💾 Periodic checkpoint saved at epoch {epoch + 1}")
        
        if improved:
            if np.isfinite(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
            if np.isfinite(val_plcc) and val_plcc > best_val_plcc:
                best_val_plcc = val_plcc
            if np.isfinite(val_srocc) and val_srocc > best_val_srocc:
                best_val_srocc = val_srocc
            best_epoch = epoch + 1
            epochs_since_improv = 0
            best_model_state = {
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            ckpt_path = os.path.join('checkpoints', f'enhanced_checkpoint_best_epoch_{best_epoch}.pth')
            torch.save(best_model_state, ckpt_path)
            print(f"✅ Saved improved model checkpoint to {ckpt_path}")
            print(
                f"✅ New best (val_loss: {best_val_loss:.5f}, val_plcc: {best_val_plcc:.5f}), val_srocc: {best_val_srocc:.5f}) at epoch {best_epoch}")
        else:
            epochs_since_improv += 1
            print(
                f"⚠️  No improvement (val_loss: {val_loss:.5f}, val_plcc: {val_plcc:.5f}), val_srocc: {val_srocc:.5f}) for {epochs_since_improv} epochs (best at epoch {best_epoch})")

        if epochs_since_improv >= patience:
            print(f"🛑 Early stopping at epoch {epoch + 1}. Best val_loss: {best_val_loss:.5f} at epoch {best_epoch}")
            ckpt_path = os.path.join('checkpoints', f'enhanced_early_stop_best_epoch_{best_epoch}.pth')
            torch.save(best_model_state, ckpt_path)
            print(f"✅ Saved best model checkpoint to {ckpt_path}")
            break

    print("🎉 Enhanced AVQA Training Complete!")

def compute_inference_flops(model, test_dataloader):
    model_cpu = model.cpu().eval()

    with torch.no_grad():
        video, waveform, audio_reliability, mos, names = next(iter(test_dataloader))
        video = video.float().cpu()
        waveform = waveform.float().cpu()
        audio_reliability = audio_reliability.float().cpu()

        flops, params = profile(
            model_cpu,
            inputs=(video, waveform, audio_reliability),
            verbose=False
        )

    return flops, params


def main():
    args = parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # ============ Load Config ============
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"🔧 Initializing Enhanced AVQA Model...")
    model = EnhancedAVQAWithCheckpoint(
        swin_cfg=config["swin"],
        attention_cfg=config["cross_modal_attention"],
        fusion_cfg=config["fusion"],
        visual_reliability_cfg=config["visual_reliability"]
    ).to(device)
    # print(f"🔧 Initializing Feature Weighting Model...")
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
    

    print(f"✅ Model initialized on device: {device}")

    # ============ Load Dataset (for split) ============
    print(f"\n📂 Loading dataset from:")
    print(f"   - Videos: {args.video_dir}")
    print(f"   - MOS CSV: {args.mos_csv}")
    print(f"   - Reliability CSV: {args.reliability_csv}")

    # Base dataset (no augmentation) just to define ordering & length
    if args.dataset=='unb-av':
        base_dataset = AVQADataset(
            video_dir=args.video_dir,
            mos_csv_path=args.mos_csv,
            scoreq_reliability_csv_path=args.reliability_csv,
            train=False  # no augmentation here
        )
    elif args.dataset=='unb-avc':
        base_dataset = UNBAVQ2013Exp3Dataset(
            video_dir=args.video_dir,
            mos_csv_path=args.mos_csv,
            scoreq_reliability_csv_path=(args.reliability_csv if args.reliability_csv else None),
            train=False
        )
    else:
        base_dataset = LIVESJTUAVQADataset(
            video_dir="datasets/LIVE-SJTU_AVQA/Distorted",
            mos_xlsx_path="datasets/LIVE-SJTU_AVQA/MOS.xlsx",
            scoreq_reliability_csv_path="scoreq_reliability_sjtu.csv",
            train=False
        )

    # ============ Train/Val/Test Split ============
    seed = config["SEED"]
    num_total = len(base_dataset)
    num_test = int(num_total * 0.1)
    num_val = int(num_total * 0.1)
    num_train = num_total - num_val - num_test

    np.random.seed(seed)
    indices = np.arange(num_total)
    np.random.shuffle(indices)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]

    print(f"\n📊 Dataset split - Train: {num_train}, Val: {num_val}, Test: {num_test}")
    np.save("train_indices_enhanced.npy", train_indices)
    np.save("val_indices_enhanced.npy", val_indices)
    np.save("test_indices_enhanced.npy", test_indices)
    print("✅ Index files saved")

    # ============ Create Train/Val/Test Datasets ============
    # Separate datasets so train can have augmentation while val/test do not
    if args.dataset=='unb-av':
        train_full = AVQADataset(
            video_dir=args.video_dir,
            mos_csv_path=args.mos_csv,
            scoreq_reliability_csv_path=args.reliability_csv,
            train=True,   # ✅ augmentation ON
        )
        
        val_full = AVQADataset(
            video_dir=args.video_dir,
            mos_csv_path=args.mos_csv,
            scoreq_reliability_csv_path=args.reliability_csv,
            train=False,  # ❌ no augmentation
        )
        
        test_full = AVQADataset(
            video_dir=args.video_dir,
            mos_csv_path=args.mos_csv,
            scoreq_reliability_csv_path=args.reliability_csv,
            train=False,  # ❌ no augmentation
        )
    
    elif args.dataset=='unb-avc':
        train_full = UNBAVQ2013Exp3Dataset(
            video_dir=args.video_dir,
            mos_csv_path=args.mos_csv,
            scoreq_reliability_csv_path=(args.reliability_csv if args.reliability_csv else None),
            train=True
        )

        val_full = UNBAVQ2013Exp3Dataset(
            video_dir=args.video_dir,
            mos_csv_path=args.mos_csv,
            scoreq_reliability_csv_path=(args.reliability_csv if args.reliability_csv else None),
            train=False
        )

        test_full = UNBAVQ2013Exp3Dataset(
            video_dir=args.video_dir,
            mos_csv_path=args.mos_csv,
            scoreq_reliability_csv_path=(args.reliability_csv if args.reliability_csv else None),
            train=False
        )
    else:
        train_full = LIVESJTUAVQADataset(
            video_dir="datasets/LIVE-SJTU_AVQA/Distorted",
            mos_xlsx_path="datasets/LIVE-SJTU_AVQA/MOS.xlsx",
            scoreq_reliability_csv_path="scoreq_reliability_sjtu.csv",
            train=True
        )


        val_full = LIVESJTUAVQADataset(
            video_dir="datasets/LIVE-SJTU_AVQA/Distorted",
            mos_xlsx_path="datasets/LIVE-SJTU_AVQA/MOS.xlsx",
            scoreq_reliability_csv_path="scoreq_reliability_sjtu.csv",
            train=False
        )

        test_full = LIVESJTUAVQADataset(
            video_dir="datasets/LIVE-SJTU_AVQA/Distorted",
            mos_xlsx_path="datasets/LIVE-SJTU_AVQA/MOS.xlsx",
            scoreq_reliability_csv_path="scoreq_reliability_sjtu.csv",
            train=False
        )

    # Apply the same split indices to each dataset
    train_dataset = torch.utils.data.Subset(train_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_full, val_indices)
    test_dataset = torch.utils.data.Subset(test_full, test_indices)

    # ============ Create DataLoaders ============
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=5
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=2
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=2
    )

    print(f"\n📦 DataLoaders created:")
    print(f"   - Train batches: {len(train_dataloader)} (batch size: {config['batch_size']})")
    print(f"   - Val batches: {len(val_dataloader)} (batch size: {config['batch_size']})")
    print(f"   - Test batches: {len(test_dataloader)} (batch size: {config['batch_size']})")

    flops, params = compute_inference_flops(model, test_dataloader)
    print(f"🚀 Inference FLOPs: {flops}")
    print(f"📦 Params: {params}")

    model = model.to(device)  # move it back to GPU for training

    # ============ Optimizer & Scheduler ============
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=5e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=5e-7
    )
    criterion = nn.L1Loss()

    print(f"\n⚙️  Training Configuration:")
    print(f"   - Optimizer: AdamW (lr={config['learning_rate']}, weight_decay=5e-3)")
    print(f"   - Scheduler: CosineAnnealingLR (T_max={config['epochs']}, eta_min=1e-6)")
    print(f"   - Loss: L1Loss + 0.15 * PearsonCorrLoss")
    print(f"   - Epochs: {config['epochs']}")
    print(f"   - Device: {device}\n")

    # ============ Train Model ============
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config=config
    )

if __name__ == "__main__":
    main()
