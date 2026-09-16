"""
High-Performance PyTorch Training for TrackNetV4 on Amateur Football Dataset.
Utilizes native NVIDIA CUDA (RTX 5060) + Mixed Precision (AMP).

Usage:
    uv run python train_torch.py [--epochs 25] [--batch_size 8] [--lr 1e-3]
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.tracknet_pytorch import TrackNetV4_PyTorch, custom_loss
from src.dataset.torch_dataset import FootballClipDataset, evaluate_detection_outcomes


def main():
    parser = argparse.ArgumentParser(description="Train TrackNetV4 on Amateur Football Dataset with PyTorch CUDA")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8 for 8GB VRAM)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--fusion_type", type=str, default="TypeA", choices=["TypeA", "TypeB"], help="Motion fusion layer type")
    parser.add_argument("--dataset_root", type=str, default="Dataset/football", help="Path to Dataset/football")
    parser.add_argument("--work_dir", type=str, default="models/football_torch", help="Output directory for checkpoints")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader subprocess workers (default: 2)")
    parser.add_argument("--tol", type=float, default=4.0, help="Pixel distance tolerance for F1 evaluation (default: 4.0)")
    parser.add_argument("--eval_freq", type=int, default=1, help="Validation frequency in epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to .pt checkpoint to resume from")
    parser.add_argument("--pos_weight", type=float, default=3.0, help="Positive class weight in focal loss (default: 3.0)")
    parser.add_argument("--no_augment", dest="augment", action="store_false", help="Disable training data augmentation")
    parser.set_defaults(augment=True)

    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    # 1. Device Setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print("=" * 65)
        print(f"  [GPU Acceleration ACTIVE] Device: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        print("=" * 65)
    else:
        device = torch.device("cpu")
        print("[WARNING] CUDA not detected. Running on CPU.")

    # 2. Datasets & Loaders
    print(f"\n[Dataset] Preparing datasets from '{args.dataset_root}'...")
    train_dataset = FootballClipDataset(
        dataset_root=args.dataset_root,
        mode="train",
        train_ratio=0.85,
        min_visible_frames=5,  # Filter clips with active ball play
        augment=args.augment,
    )
    val_dataset = FootballClipDataset(
        dataset_root=args.dataset_root,
        mode="val",
        train_ratio=0.85,
        min_visible_frames=5,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # 3. Model Architecture
    print(f"\n[Model] Building TrackNetV4 ({args.fusion_type})...")
    model = TrackNetV4_PyTorch(in_channels=9, out_channels=3, fusion_type=args.fusion_type).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total trainable parameters: {total_params:,}")

    # 4. Optimizer, Scaler & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    start_epoch = 0
    best_f1 = 0.0

    if args.resume and os.path.exists(args.resume):
        print(f"[Model] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_f1 = checkpoint.get("best_f1", checkpoint.get("f1", 0.0))

        # Check existing best checkpoint in work_dir so we don't overwrite with an inferior model
        best_path = os.path.join(args.work_dir, "tracknetv4_best.pt")
        if os.path.exists(best_path):
            try:
                best_ckpt = torch.load(best_path, map_location="cpu")
                best_f1 = max(best_f1, best_ckpt.get("best_f1", 0.0))
            except Exception:
                pass

        for _ in range(start_epoch):
            scheduler.step()
        print(f"[Model] Resumed at epoch {start_epoch}/{args.epochs} | Current LR: {scheduler.get_last_lr()[0]:.6f} | Best F1: {best_f1:.4f}")

    # 5. Main Training Loop
    print("\n" + "=" * 65)
    print(f"  Starting Training: {args.epochs} Epochs | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("=" * 65)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        step_count = 0
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}/{args.epochs:02d} [Train]")
        for x_batch, y_batch in pbar:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                y_pred = model(x_batch)
                loss = custom_loss(y_pred, y_batch, pos_weight=args.pos_weight)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            running_loss += loss_val
            step_count += 1
            pbar.set_postfix({"loss": f"{loss_val:.5f}", "avg_loss": f"{running_loss / step_count:.5f}"})

        scheduler.step()
        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / max(1, step_count)

        print(f"\n[Epoch {epoch + 1:02d}] Finished in {epoch_time:.1f}s | Train Loss: {epoch_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # 6. Validation Phase
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            TP = TN = FP1 = FP2 = FN = 0
            val_loss = 0.0
            val_steps = 0

            with torch.inference_mode():
                for x_val, y_val in tqdm(val_loader, desc="Validating", leave=False):
                    x_val = x_val.to(device, non_blocking=True)
                    y_val = y_val.to(device, non_blocking=True)

                    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                        pred_val = model(x_val)
                        loss = custom_loss(pred_val, y_val, pos_weight=args.pos_weight)

                    val_loss += loss.item()
                    val_steps += 1

                    # Compute outcome metrics with OpenCV contours
                    pred_np = pred_val.detach().cpu().numpy()
                    gt_np = y_val.detach().cpu().numpy()
                    tp, tn, fp1, fp2, fn = evaluate_detection_outcomes(pred_np, gt_np, tol=args.tol)
                    TP += tp
                    TN += tn
                    FP1 += fp1
                    FP2 += fp2
                    FN += fn

            total_cases = TP + TN + FP1 + FP2 + FN
            acc = (TP + TN) / total_cases if total_cases > 0 else 0
            prec = TP / (TP + FP1 + FP2) if (TP + FP1 + FP2) > 0 else 0
            rec = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            avg_val_loss = val_loss / max(1, val_steps)

            print(f"[Validation] Loss: {avg_val_loss:.5f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1-Score: {f1:.3f} (tol={args.tol}px)")

            # Save best checkpoint
            if f1 > best_f1:
                best_f1 = f1
                best_path = os.path.join(args.work_dir, "tracknetv4_best.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "fusion_type": args.fusion_type,
                }, best_path)
                print(f"  --> [NEW BEST MODEL] Saved to {best_path} (F1: {best_f1:.3f})")

        # Periodic checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(args.work_dir, f"tracknetv4_epoch_{epoch + 1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "f1": f1 if 'f1' in locals() else 0.0,
                "best_f1": best_f1,
                "fusion_type": args.fusion_type,
            }, ckpt_path)

    # Save final model
    final_path = os.path.join(args.work_dir, "tracknetv4_final.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "best_f1": best_f1,
        "fusion_type": args.fusion_type,
    }, final_path)
    print(f"\n[DONE] Training complete! Best model saved to: {os.path.join(args.work_dir, 'tracknetv4_best.pt')}")


if __name__ == "__main__":
    main()
