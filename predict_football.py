#!/usr/bin/env python
"""
Run TrackNetV4 Inferences on Football Match Videos.

Outputs:
1. Annotated MP4 video with ball detection and trajectory trail.
2. CSV file with Frame, Visibility, X, Y coordinates.

Usage:
    uv run python predict_football.py --video_path videos/clip.mp4 --model_weights models/football/model_final.keras
"""

import argparse
import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "TrackNetV4", "src"))

from tensorflow.keras.models import load_model
from util import custom_loss
from models.TrackNetV4 import (
    MotionPromptLayer,
    FusionLayerTypeA,
    FusionLayerTypeB,
)


def predict_video(
    video_path: str,
    model_weights: str,
    output_dir: str = "./predictions",
    queue_length: int = 15,
    target_height: int = 288,
    target_width: int = 512,
):
    os.makedirs(output_dir, exist_ok=True)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    out_video_path = os.path.join(output_dir, f"{video_basename}_tracknet_pred.mp4")
    out_csv_path = os.path.join(output_dir, f"{video_basename}_trajectory.csv")

    # Load Model
    print(f"[TrackNet] Loading trained model from {model_weights}...")
    custom_objects = {
        "custom_loss": custom_loss,
        "MotionPromptLayer": MotionPromptLayer,
        "FusionLayerTypeA": FusionLayerTypeA,
        "FusionLayerTypeB": FusionLayerTypeB,
    }
    model = load_model(model_weights, custom_objects=custom_objects)
    print("[TrackNet] Model loaded successfully!")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (orig_w, orig_h))

    ratio_w = orig_w / float(target_width)
    ratio_h = orig_h / float(target_height)

    frames_buffer = []
    orig_frames_buffer = []
    history = []
    results = []

    pbar = tqdm(total=total_frames, desc="Running TrackNet inference")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_frames_buffer.append(frame.copy())

        # Preprocess frame for TrackNet (RGB, resized, channels first)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (target_width, target_height)).astype(np.float32) / 255.0
        img_ch_first = np.moveaxis(img_resized, -1, 0)
        frames_buffer.append(img_ch_first)

        if len(frames_buffer) == 3:
            # Stack 3 frames -> (1, 9, 288, 512)
            input_tensor = np.expand_dims(np.concatenate(frames_buffer, axis=0), axis=0)
            heatmap_preds = model.predict(input_tensor, verbose=0)[0]  # (3, 288, 512)

            # Predict position for middle frame (or last)
            pred_hm = heatmap_preds[1] * 255.0
            pred_hm_u8 = pred_hm.astype(np.uint8)

            contours, _ = cv2.findContours(pred_hm_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                best_cnt = max(contours, key=cv2.contourArea)
                M = cv2.moments(best_cnt)
                if M["m00"] > 0:
                    cx = int(round((M["m10"] / M["m00"]) * ratio_w))
                    cy = int(round((M["m01"] / M["m00"]) * ratio_h))
                    vis = 1
                else:
                    cx, cy, vis = -1, -1, 0
            else:
                cx, cy, vis = -1, -1, 0

            # Render middle frame
            curr_frame = orig_frames_buffer[1]
            if vis == 1:
                history.append((cx, cy))
                results.append({"Frame": frame_idx - 1, "Visibility": 1, "X": cx, "Y": cy})
            else:
                history.append(None)
                results.append({"Frame": frame_idx - 1, "Visibility": 0, "X": -1, "Y": -1})

            if len(history) > queue_length:
                history.pop(0)

            # Draw trajectory trail
            for k in range(1, len(history)):
                if history[k - 1] is not None and history[k] is not None:
                    thick = int(2 + 4 * (k / len(history)))
                    cv2.line(curr_frame, history[k - 1], history[k], (0, 255, 0), thick)

            # Draw current ball marker
            if vis == 1:
                cv2.circle(curr_frame, (cx, cy), 8, (0, 0, 255), -1)
                cv2.circle(curr_frame, (cx, cy), 15, (0, 255, 255), 2)
                cv2.putText(curr_frame, f"Football ({cx},{cy})", (cx + 15, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            writer.write(curr_frame)

            # Pop oldest
            frames_buffer.pop(0)
            orig_frames_buffer.pop(0)

        frame_idx += 1
        pbar.update(1)

    cap.release()
    writer.release()
    pbar.close()

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(out_csv_path, index=False)
    print(f"[Done] Trajectory CSV saved: {out_csv_path}")
    print(f"[Done] Output video saved: {out_video_path}")


def main():
    parser = argparse.ArgumentParser(description="TrackNetV4 Football Video Predictor")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input football match video")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to trained .keras model weights")
    parser.add_argument("--output_dir", type=str, default="./predictions", help="Output directory")
    parser.add_argument("--queue_length", type=int, default=15, help="Length of trajectory trail")

    args = parser.parse_args()
    predict_video(
        video_path=args.video_path,
        model_weights=args.model_weights,
        output_dir=args.output_dir,
        queue_length=args.queue_length,
    )


if __name__ == "__main__":
    main()
