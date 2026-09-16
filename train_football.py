#!/usr/bin/env python
"""
Train / Fine-tune TrackNetV4 on Amateur Football Dataset.

Usage:
    uv run python train_football.py [--model_name TrackNetV4_TypeA] [--epochs 30] [--batch_size 2]
"""

import argparse
import datetime
import os
import sys

# Ensure TrackNetV4/src is in Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "TrackNetV4", "src"))

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adadelta
import numpy as np

from util import custom_loss, get_dataset, get_model, outcome
from models.TrackNetV4 import (
    MotionPromptLayer,
    FusionLayerTypeA,
    FusionLayerTypeB,
)


def main():
    parser = argparse.ArgumentParser(description="Train TrackNetV4 on Amateur Football Dataset")
    parser.add_argument("--model_name", type=str, default="TrackNetV4_TypeA", choices=["Baseline_TrackNetV2", "TrackNetV4_TypeA", "TrackNetV4_TypeB"], help="Model architecture")
    parser.add_argument("--dataset", type=str, default="football", help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--learning_rate", type=float, default=1.0, help="Learning rate for Adadelta (default: 1.0)")
    parser.add_argument("--height", type=int, default=288, help="Target image height (default: 288)")
    parser.add_argument("--width", type=int, default=512, help="Target image width (default: 512)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs (default: 30)")
    parser.add_argument("--tol", type=int, default=4, help="Tolerance distance for evaluation (default: 4)")
    parser.add_argument("--model_path", type=str, default=None, help="Pretrained .keras checkpoint to fine-tune")
    parser.add_argument("--work_dir", type=str, default="./models/football", help="Output directory for saved models")
    parser.add_argument("--save_freq", type=int, default=5, help="Save checkpoint every N epochs")

    args = parser.parse_args()

    # Unique work directory with timestamp if needed
    os.makedirs(args.work_dir, exist_ok=True)

    print("=" * 60)
    print("TrackNetV4 Football Training Configurations:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # 1. Load Model Architecture
    print(f"[Model] Initializing {args.model_name} ({args.width}x{args.height})...")
    model = get_model(args.model_name, args.height, args.width)

    # 2. Optionally load pretrained weights
    if args.model_path and os.path.exists(args.model_path):
        print(f"[Model] Loading pretrained weights from {args.model_path}...")
        custom_objects = {
            "custom_loss": custom_loss,
            "MotionPromptLayer": MotionPromptLayer,
            "FusionLayerTypeA": FusionLayerTypeA,
            "FusionLayerTypeB": FusionLayerTypeB,
        }
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_model = load_model(args.model_path, custom_objects=custom_objects)
            temp_weights = os.path.join(temp_dir, "weights.weights.h5")
            temp_model.save_weights(temp_weights)
            model.load_weights(temp_weights, skip_mismatch=True)
            print("[Model] Weights loaded successfully with skip_mismatch=True.")

    # 3. Load Datasets
    print(f"[Dataset] Loading '{args.dataset}' dataset (train/test)...")
    dataset_train = get_dataset(args.dataset, "train", height=args.height, width=args.width)
    dataset_test = get_dataset(args.dataset, "test", height=args.height, width=args.width)

    num_train_pairs = len(dataset_train)
    print(f"[Dataset] Training sample pairs available: {num_train_pairs}")
    if num_train_pairs == 0:
        print("[Error] No processed training data found! Run prepare_dataset.py first.")
        return

    # 4. Compile Model
    model.compile(
        loss=custom_loss,
        optimizer=Adadelta(learning_rate=args.learning_rate),
        metrics=["accuracy"],
    )

    # 5. Training Loop
    best_f1 = 0.0
    for epoch in range(args.epochs):
        print(f"\n======== Epoch {epoch + 1}/{args.epochs} ========")
        
        # Train on each clip
        train_loss = 0.0
        clip_count = 0
        for x_train, y_train in dataset_train:
            hist = model.fit(x_train, y_train, batch_size=args.batch_size, epochs=1, verbose=1)
            train_loss += hist.history["loss"][0]
            clip_count += 1
            del x_train, y_train

        avg_loss = train_loss / max(1, clip_count)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

        # Intermediate evaluation
        if len(dataset_test) > 0 and (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            TP = TN = FP1 = FP2 = FN = 0
            for x_test, y_test in dataset_test:
                y_pred = model.predict(x_test, batch_size=args.batch_size, verbose=0)
                tp, tn, fp1, fp2, fn = outcome(y_pred, y_test, args.tol)
                TP += tp
                TN += tn
                FP1 += fp1
                FP2 += fp2
                FN += fn
                del x_test, y_test, y_pred

            total_cases = TP + TN + FP1 + FP2 + FN
            acc = (TP + TN) / total_cases if total_cases > 0 else 0
            prec = TP / (TP + FP1 + FP2) if (TP + FP1 + FP2) > 0 else 0
            rec = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            print(f"[Eval] Epoch {epoch + 1} Metrics: Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}")

            # Save periodic checkpoint
            ckpt_path = os.path.join(args.work_dir, f"model_epoch_{epoch + 1}.keras")
            model.save(ckpt_path)
            print(f"[Checkpoint] Saved model checkpoint: {ckpt_path}")

            if f1 > best_f1:
                best_f1 = f1
                best_path = os.path.join(args.work_dir, "model_best.keras")
                model.save(best_path)
                print(f"[Checkpoint] New best model saved: {best_path} (F1: {best_f1:.3f})")

    # Save final model
    final_path = os.path.join(args.work_dir, "model_final.keras")
    model.save(final_path)
    print(f"\n[Done] Training completed! Final model saved to {final_path}")


if __name__ == "__main__":
    main()
