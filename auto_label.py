#!/usr/bin/env python
"""
Auto-Labeling Pipeline using SAM 3.1 Multiplex for Football.

Reads videos from the `videos/` folder, splits long videos into manageable clips
(default 60 frames each, matching TrackNetV4 rally clips and fitting comfortably in 8GB VRAM),
tracks the ball using SAM 3.1 Multiplex (`sam3.1_multiplex_fp16.safetensors`),
carries over the ball position from Clip N to Clip N+1, applies physical filtering,
and generates `Label.csv` in TrackNetV4 format inside `Dataset/football/<video_name>/Clip<N>/`.

Usage:
    uv run python auto_label.py [--video "videos/clip (1).mp4"] [--clip_len 60]
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import glob
import math
import shutil
import gc
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.labeling.sam3_tracker import FootballSAMTracker
from src.labeling.ball_filter import BallTrajectoryFilter


def extract_clip_frames(
    cap: cv2.VideoCapture,
    output_dir: str,
    num_frames: int,
    start_frame_idx: int,
) -> int:
    """
    Extracts up to num_frames from cap starting at start_frame_idx into output_dir
    named 0000.jpg, 0001.jpg, ...
    """
    os.makedirs(output_dir, exist_ok=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
    saved_count = 0

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(output_dir, f"{saved_count:04d}.jpg")
        cv2.imwrite(out_path, frame)
        saved_count += 1

    return saved_count


def generate_annotated_video(
    frames_dir: str,
    records: list,
    output_video_path: str,
    fps: float = 30.0,
    trail_len: int = 15,
):
    """
    Renders an annotated video showing the detected ball trajectory and bounding box.
    """
    if not records:
        return

    first_frame_path = os.path.join(frames_dir, records[0]["file name"])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        return
    h, w = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    history = []

    for r in records:
        fn = r["file name"]
        vis = r["visibility"]
        x, y = r["x-coordinate"], r["y-coordinate"]

        img_path = os.path.join(frames_dir, fn)
        img = cv2.imread(img_path)
        if img is None:
            continue

        if vis == 1:
            history.append((x, y))
        else:
            history.append(None)

        if len(history) > trail_len:
            history.pop(0)

        # Draw trajectory trail
        for k in range(1, len(history)):
            if history[k - 1] is not None and history[k] is not None:
                thickness = int(2 + 3 * (k / len(history)))
                cv2.line(img, history[k - 1], history[k], (0, 255, 0), thickness)

        # Draw current ball circle
        if vis == 1:
            cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
            cv2.circle(img, (x, y), 14, (0, 255, 255), 2)
            cv2.putText(
                img,
                f"BALL ({x},{y})",
                (x + 15, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

        writer.write(img)

    writer.release()


def process_single_video(
    video_path: str,
    tracker: FootballSAMTracker,
    dataset_football_root: str,
    text_prompt: str = "small moving soccer ball on the pitch, ball in play",
    point_prompt=None,
    box_prompt=None,
    clip_len: int = 60,
    prob_thresh: float = 0.30,
    use_carryover: bool = False,
    max_frames: int = 0,
    save_annotated: bool = True,
):
    """
    Processes one video file by segmenting it into clips of `clip_len` frames:
    1. Extracts frames into Dataset/football/<video_name>/Clip<N>/
    2. Runs SAM 3.1 tracking (with enhanced text prompt targeting moving balls)
    3. Filters trajectory with physical plausibility constraints
    4. Writes Label.csv
    5. Frees GPU VRAM between clips to prevent CUDA OOM
    """
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    video_dataset_dir = os.path.join(dataset_football_root, video_basename)
    os.makedirs(video_dataset_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Video] Error: Cannot open video file {video_path}")
        return

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if max_frames > 0:
        total_video_frames = min(total_video_frames, max_frames)

    num_clips = int(math.ceil(total_video_frames / float(clip_len)))
    print(f"\n[Video] {video_path}: Total {total_video_frames} frames -> {num_clips} clip(s) of ~{clip_len} frames.")

    ball_filter = BallTrajectoryFilter(
        min_area=15,
        max_area=4000,
        min_circularity=0.25,
        min_aspect_ratio=0.40,
        max_speed_px=220.0,
        max_gap_interpolation=5,
        min_y_ratio=0.05,
    )

    last_known_ball_pos = point_prompt

    for clip_idx in range(1, num_clips + 1):
        clip_name = f"Clip{clip_idx}"
        clip_dir = os.path.join(video_dataset_dir, clip_name)
        label_csv_path = os.path.join(clip_dir, "Label.csv")

        start_frame_idx = (clip_idx - 1) * clip_len
        frames_to_extract = min(clip_len, total_video_frames - start_frame_idx)
        if frames_to_extract < 3:
            # Skip clips with fewer than 3 frames (TrackNet requires >= 3)
            continue

        print(f"\n--- [{video_basename}] {clip_name} (Frames {start_frame_idx} to {start_frame_idx + frames_to_extract - 1}) ---")

        # Check if this clip was already processed with good detections
        if os.path.exists(label_csv_path):
            try:
                df_existing = pd.read_csv(label_csv_path)
                vis_count = (df_existing["visibility"] == 1).sum()
                if vis_count > 3:
                    vis_rows = df_existing[df_existing["visibility"] == 1]
                    if len(vis_rows) > 0 and vis_rows.index[-1] >= len(df_existing) - 10:
                        last_r = vis_rows.iloc[-1]
                        last_known_ball_pos = (int(round(last_r["x-coordinate"])), int(round(last_r["y-coordinate"])))
                    else:
                        last_known_ball_pos = None
                    print(f"[AutoLabeler] {clip_name} already labeled ({vis_count}/{len(df_existing)} visible). Skipping.")
                    continue
                else:
                    print(f"[AutoLabeler] {clip_name} has only {vis_count} visible detections. Re-labeling with enhanced motion prompt...")
            except Exception:
                pass

        # Wipe incomplete frames from any prior interrupted run
        if os.path.exists(clip_dir):
            shutil.rmtree(clip_dir, ignore_errors=True)
        os.makedirs(clip_dir, exist_ok=True)

        # 1. Extract clip frames
        saved_count = extract_clip_frames(cap, clip_dir, frames_to_extract, start_frame_idx)
        if saved_count < 3:
            print(f"[{clip_name}] Too few frames ({saved_count}). Skipping.")
            continue

        # 2. Determine prompt for this clip
        clip_point = None
        clip_box = None
        clip_text = None

        if clip_idx == 1 and box_prompt is not None:
            clip_box = box_prompt
        elif clip_idx == 1 and point_prompt is not None:
            clip_point = point_prompt
        elif use_carryover and last_known_ball_pos is not None:
            clip_point = last_known_ball_pos
            print(f"[SAM 3.1] Tracking {clip_name} using carryover ball position: {clip_point}")
        else:
            clip_text = text_prompt
            print(f"[SAM 3.1] Tracking {clip_name} using text prompt: '{clip_text}'")

        # 3. Run SAM 3.1 tracking on this clip
        try:
            masks_per_frame = tracker.track_video(
                frames_dir=clip_dir,
                text_prompt=clip_text,
                point_prompt=clip_point,
                box_prompt=clip_box,
                output_prob_thresh=prob_thresh,
            )
        except Exception as e:
            print(f"[SAM 3.1] Tracking error on {clip_name}: {e}. Proceeding with empty detections for this clip.")
            masks_per_frame = {}

        # 4. Extract centroids and apply physical filters
        frame_files = sorted(glob.glob(os.path.join(clip_dir, "*.jpg")))
        raw_records = []

        for i, fpath in enumerate(frame_files):
            fname = os.path.basename(fpath)
            mask = masks_per_frame.get(i)
            cx, cy, area, is_valid = ball_filter.extract_centroid_from_mask(mask)

            raw_records.append({
                "frame_idx": i,
                "file name": fname,
                "visibility": 1 if is_valid else 0,
                "x": cx,
                "y": cy,
                "status": 0,
            })

        # Smooth trajectory and interpolate short occlusions
        smoothed_records = ball_filter.smooth_trajectory(raw_records)

        # Convert to final TrackNetV4 Label format
        final_records = []
        visible_count = 0
        for r in smoothed_records:
            vis = r["visibility"]
            x = r["x"] if vis == 1 else -1
            y = r["y"] if vis == 1 else -1
            if vis == 1:
                visible_count += 1
            final_records.append({
                "file name": r["file name"],
                "visibility": vis,
                "x-coordinate": x,
                "y-coordinate": y,
                "status": 0,
            })

        df_labels = pd.DataFrame(final_records)
        df_labels.to_csv(label_csv_path, index=False)
        print(f"[Dataset] Saved labels: {label_csv_path} (Visible frames: {visible_count}/{len(final_records)})")

        # Update last known position for carryover to the next clip
        vis_indices = [idx for idx, r in enumerate(final_records) if r["visibility"] == 1]
        if vis_indices and vis_indices[-1] >= len(final_records) - 10:
            last_vis_record = final_records[vis_indices[-1]]
            last_known_ball_pos = (last_vis_record["x-coordinate"], last_vis_record["y-coordinate"])
            print(f"[{clip_name}] End-of-clip ball position captured for carryover: {last_known_ball_pos}")
        else:
            last_known_ball_pos = None

        # 5. Optional visualization video
        if save_annotated:
            annotated_path = os.path.join(clip_dir, f"{clip_name}_annotated.mp4")
            generate_annotated_video(clip_dir, final_records, annotated_path, fps=fps)

        # 6. Explicit memory cleanup between clips
        del masks_per_frame
        del raw_records
        del smoothed_records
        del final_records
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cap.release()
    print(f"\n[AutoLabeler] Completed processing all clips for {video_path}!")


def main():
    parser = argparse.ArgumentParser(description="SAM 3.1 Football Ball Auto-Labeler with Clip Windowing")
    parser.add_argument("--video", type=str, default=None, help="Path to single video file. If omitted, processes all in videos/")
    parser.add_argument("--weights", type=str, default="sam3.1_multiplex_fp16.safetensors", help="Path to SAM 3.1 safetensors")
    parser.add_argument("--output_dir", type=str, default="Dataset/football", help="Target dataset root directory")
    parser.add_argument("--text_prompt", type=str, default="small moving soccer ball on the pitch, ball in play", help="Text prompt for open-vocabulary detection")
    parser.add_argument("--prob_thresh", type=float, default=0.30, help="Detection probability threshold in SAM 3 (default: 0.30)")
    parser.add_argument("--carryover", action="store_true", help="Enable carryover point prompt between clips")
    parser.add_argument("--point", type=str, default=None, help="Optional initial click 'X,Y' on frame 0 of Clip1")
    parser.add_argument("--box", type=str, default=None, help="Optional initial box 'X1,Y1,X2,Y2' on frame 0 of Clip1")
    parser.add_argument("--clip_len", type=int, default=60, help="Number of frames per clip (default: 60 frames, ~2s at 30fps)")
    parser.add_argument("--max_frames", type=int, default=0, help="Max total frames to process per video (0 for full video)")
    parser.add_argument("--no_video", action="store_true", help="Skip rendering annotated video")

    args = parser.parse_args()

    # Parse point / box if provided
    point_prompt = None
    if args.point:
        parts = [int(p.strip()) for p in args.point.split(",")]
        point_prompt = (parts[0], parts[1])

    box_prompt = None
    if args.box:
        parts = [int(p.strip()) for p in args.box.split(",")]
        box_prompt = (parts[0], parts[1], parts[2], parts[3])

    # Find videos
    if args.video:
        video_paths = [args.video]
    else:
        video_paths = sorted(glob.glob("videos/*.mp4") + glob.glob("videos/*.avi") + glob.glob("videos/*.mkv"))

    if not video_paths:
        print("No videos found in videos/ directory! Please add your video files to videos/.")
        return

    print(f"[AutoLabeler] Found {len(video_paths)} video(s) to process.")
    print(f"[AutoLabeler] Clip length: {args.clip_len} frames (~{args.clip_len/30.0:.1f}s)")
    print(f"[AutoLabeler] Prompt: '{args.text_prompt}' (prob_thresh={args.prob_thresh})")

    # Initialize SAM 3.1 Tracker once
    tracker = FootballSAMTracker(args.weights)

    for vpath in video_paths:
        print(f"\n=======================================================")
        print(f"Processing video: {vpath}")
        print(f"=======================================================")
        process_single_video(
            video_path=vpath,
            tracker=tracker,
            dataset_football_root=args.output_dir,
            text_prompt=args.text_prompt,
            point_prompt=point_prompt,
            box_prompt=box_prompt,
            clip_len=args.clip_len,
            prob_thresh=args.prob_thresh,
            use_carryover=args.carryover,
            max_frames=args.max_frames,
            save_annotated=not args.no_video,
        )

    print("\n[AutoLabeler] All videos processed successfully!")


if __name__ == "__main__":
    main()
