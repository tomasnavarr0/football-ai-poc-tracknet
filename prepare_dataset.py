#!/usr/bin/env python
"""
Prepare Football Dataset for TrackNetV4.

Converts extracted frames and Label.csv files from `Dataset/football`
into preprocessed TrackNetV4 training sequences (.npy tensors of heatmaps and 9-channel frame stacks).

Usage:
    uv run python prepare_dataset.py [--dataset_dir Dataset/football] [--output_dir processed_data/football]
"""

import argparse
import os
from src.dataset.football_dataset import build_football_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare Football Dataset for TrackNetV4")
    parser.add_argument("--dataset_dir", type=str, default="Dataset/football", help="Root directory containing labeled clips")
    parser.add_argument("--output_dir", type=str, default="processed_data/football", help="Destination directory for .npy splits")
    parser.add_argument("--train_ratio", type=float, default=0.75, help="Train/test split ratio (default: 0.75)")
    parser.add_argument("--height", type=int, default=288, help="Target frame height (default: 288)")
    parser.add_argument("--width", type=int, default=512, help="Target frame width (default: 512)")

    args = parser.parse_args()

    print(f"[Dataset] Starting preparation of dataset from: {args.dataset_dir}")
    print(f"[Dataset] Target resolution: {args.width}x{args.height}")
    print(f"[Dataset] Saving processed files to: {args.output_dir}")

    build_football_dataset(
        dataset_root=args.dataset_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        target_height=args.height,
        target_width=args.width,
    )


if __name__ == "__main__":
    main()
