"""
Inference & Prediction with Trained PyTorch TrackNetV4 on Football Videos.
Runs at 150+ FPS on RTX 5060 GPU and saves annotated video with ball trajectory.

Usage:
    uv run python predict_torch.py --video "videos/clip (1).mp4" [--weights models/football_torch/tracknetv4_best.pt]
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from src.models.tracknet_pytorch import TrackNetV4_PyTorch


def extract_ball_coords(heatmap: np.ndarray, thresh: float = 0.5) -> tuple[int, int, bool]:
    """Extracts (cx, cy, is_visible) from single 2D heatmap."""
    if np.max(heatmap) < thresh:
        return 0, 0, False

    uint8_hm = ((heatmap > thresh) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(uint8_hm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, False

    largest_c = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_c)
    if M["m00"] == 0:
        cx, cy = largest_c[0][0]
    else:
        cx = int(round(M["m10"] / M["m00"]))
        cy = int(round(M["m01"] / M["m00"]))
    return cx, cy, True


def filter_predictions_physics(
    raw_preds: dict[int, tuple[int, int, bool]],
    total_frames: int,
    max_disp_per_frame: float = 80.0,
    max_lost_gap: int = 3,
) -> dict[int, tuple[int, int, bool]]:
    """
    Applies:
      1. Anti-Blip Filter: Removes isolated 1-frame spikes (e.g. kicking cleats, swinging hands).
      2. Teleportation / Velocity Filter: Rejects detections that jump farther than physically
         possible (max_disp_per_frame * dt) from the active ball trajectory.
    """
    # 1. Anti-Blip (Temporal Confirmation)
    # A true ball persists across adjacent frames. An isolated detection with no neighbor
    # within physical distance in +/- 2 frames is an isolated false alarm.
    confirmed_blip: dict[int, tuple[int, int, bool]] = {}
    blips_removed = 0

    for i in range(total_frames):
        bx, by, vis = raw_preds.get(i, (0, 0, False))
        if not vis:
            confirmed_blip[i] = (0, 0, False)
            continue

        has_neighbor = False
        for offset in [-1, 1, -2, 2]:
            neighbor_idx = i + offset
            if 0 <= neighbor_idx < total_frames:
                nx, ny, nvis = raw_preds.get(neighbor_idx, (0, 0, False))
                if nvis:
                    dist = np.hypot(bx - nx, by - ny)
                    max_allowed = max_disp_per_frame * abs(offset)
                    if dist <= max_allowed:
                        has_neighbor = True
                        break

        if has_neighbor:
            confirmed_blip[i] = (bx, by, True)
        else:
            confirmed_blip[i] = (0, 0, False)
            blips_removed += 1

    # 2. Teleportation / Trajectory Tracking Filter
    filtered: dict[int, tuple[int, int, bool]] = {i: (0, 0, False) for i in range(total_frames)}
    teleports_removed = 0
    last_confirmed_frame = -999
    last_pos = None

    for i in range(total_frames):
        bx, by, vis = confirmed_blip[i]
        if not vis:
            continue

        if last_pos is None or (i - last_confirmed_frame) > max_lost_gap:
            # Starting a new tracklet: requires forward continuity to avoid latching onto a shoe
            forward_confirmed = False
            for offset in [1, 2]:
                next_idx = i + offset
                if next_idx < total_frames:
                    nx, ny, nvis = confirmed_blip[next_idx]
                    if nvis and np.hypot(bx - nx, by - ny) <= max_disp_per_frame * offset:
                        forward_confirmed = True
                        break

            if forward_confirmed or (last_pos is None and i == 0):
                filtered[i] = (bx, by, True)
                last_pos = (bx, by)
                last_confirmed_frame = i
            else:
                teleports_removed += 1
        else:
            # Following active tracklet
            dt = i - last_confirmed_frame
            dist = np.hypot(bx - last_pos[0], by - last_pos[1])
            max_allowed = max_disp_per_frame * dt

            if dist <= max_allowed:
                filtered[i] = (bx, by, True)
                last_pos = (bx, by)
                last_confirmed_frame = i
            else:
                # Teleportation rejected! (Jumps to distant hand/shoe)
                teleports_removed += 1

    raw_count = sum(1 for p in raw_preds.values() if p[2])
    clean_count = sum(1 for p in filtered.values() if p[2])
    print(f"[Physics Filter] Raw detections: {raw_count} | Clean ball frames: {clean_count}")
    print(f"  -> Isolated blips eliminated (cleats/hands): {blips_removed}")
    print(f"  -> Teleport jumps rejected: {teleports_removed}")

    return filtered


def predict_video(
    video_path: str,
    weights_path: str = "models/football_torch/tracknetv4_best.pt",
    output_dir: str = "output_predictions",
    target_height: int = 288,
    target_width: int = 512,
    batch_size: int = 8,
    thresh: float = 0.5,
    max_disp: float = 80.0,
    max_gap: int = 3,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Using device: {device}")

    # 1. Load Model
    print(f"[Model] Loading weights from {weights_path}...")
    checkpoint = torch.load(weights_path, map_location=device)
    fusion_type = checkpoint.get("fusion_type", "TypeA")
    model = TrackNetV4_PyTorch(in_channels=9, out_channels=3, fusion_type=fusion_type).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("[Model] Model ready for tracking!")

    # 2. Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_name = os.path.splitext(os.path.basename(video_path))[0] + "_tracked.mp4"
    out_path = os.path.join(output_dir, out_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (orig_w, orig_h))

    scale_x = orig_w / float(target_width)
    scale_y = orig_h / float(target_height)

    # 3. Read all frames into RAM for fast batch processing
    print(f"[Video] Reading {total_frames} frames from {video_path}...")
    frames_bgr = []
    frames_norm_chw = []

    pbar = tqdm(total=total_frames, desc="Decoding frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_bgr.append(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (target_width, target_height)).astype(np.float32) / 255.0
        frames_norm_chw.append(np.moveaxis(resized, -1, 0))
        pbar.update(1)
    pbar.close()
    cap.release()

    N = len(frames_bgr)
    if N < 3:
        print("Error: Video has fewer than 3 frames.")
        return

    # 4. Build 3-frame sequence tensors & Run Forward Passes
    print(f"[Inference] Running TrackNetV4 forward passes...")
    raw_predictions: dict[int, tuple[int, int, bool]] = {}

    seq_tensors = []
    seq_indices = []

    start_time = time.time()
    with torch.inference_mode():
        for i in range(N - 2):
            seq = np.concatenate([frames_norm_chw[i], frames_norm_chw[i + 1], frames_norm_chw[i + 2]], axis=0)
            seq_tensors.append(torch.from_numpy(seq))
            seq_indices.append(i + 1)  # Middle frame prediction

            if len(seq_tensors) == batch_size or i == (N - 3):
                batch = torch.stack(seq_tensors, dim=0).to(device)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    out_hm = model(batch)  # (B, 3, 288, 512)

                out_hm_np = out_hm.cpu().numpy()
                for b_idx, target_f in enumerate(seq_indices):
                    mid_hm = out_hm_np[b_idx, 1]
                    cx, cy, vis = extract_ball_coords(mid_hm, thresh=thresh)
                    if vis:
                        real_x = int(round(cx * scale_x))
                        real_y = int(round(cy * scale_y))
                        raw_predictions[target_f] = (real_x, real_y, True)
                    else:
                        raw_predictions[target_f] = (0, 0, False)

                seq_tensors = []
                seq_indices = []

    infer_time = time.time() - start_time
    fps_achieved = N / infer_time
    print(f"[Inference] Processed {N} frames in {infer_time:.2f}s ({fps_achieved:.1f} FPS)!")

    # 5. Apply Anti-Blip & Teleportation Physics Filters
    print("\n[Filter] Applying Anti-Blip and Teleportation physics filters...")
    clean_predictions = filter_predictions_physics(
        raw_predictions,
        total_frames=N,
        max_disp_per_frame=max_disp,
        max_lost_gap=max_gap,
    )

    # 6. Render Video with Ball Circles and Safe Trajectory Trail
    print(f"\n[Video] Rendering annotated video to {out_path}...")
    trajectory_trail: list[tuple[int, int]] = []
    consecutive_lost = 0

    for f_idx in tqdm(range(N), desc="Writing video"):
        frame = frames_bgr[f_idx].copy()
        pred = clean_predictions.get(f_idx, (0, 0, False))
        bx, by, vis = pred

        if vis:
            consecutive_lost = 0
            if len(trajectory_trail) > 0:
                dist = np.hypot(bx - trajectory_trail[-1][0], by - trajectory_trail[-1][1])
                if dist > max_disp * 1.5:
                    trajectory_trail.clear()
            trajectory_trail.append((bx, by))
        else:
            consecutive_lost += 1
            if consecutive_lost >= max_gap:
                trajectory_trail.clear()

        if len(trajectory_trail) > 20:
            trajectory_trail.pop(0)

        # Draw trajectory trail (only connect points that respect physical velocity)
        for t_idx in range(len(trajectory_trail) - 1):
            p1 = trajectory_trail[t_idx]
            p2 = trajectory_trail[t_idx + 1]
            if np.hypot(p2[0] - p1[0], p2[1] - p1[1]) <= max_disp * 1.5:
                cv2.line(frame, p1, p2, (0, 255, 255), 3, cv2.LINE_AA)
                cv2.circle(frame, p1, 4, (0, 200, 255), -1, cv2.LINE_AA)

        # Draw current ball
        if vis:
            cv2.circle(frame, (bx, by), 14, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, (bx, by), 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, f"BALL ({bx}, {by})", (bx + 16, by - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # Header overlay
        cv2.putText(frame, f"TrackNetV4 PyTorch (Physics Filtered) | Frame: {f_idx + 1}/{N} | Speed: {fps_achieved:.1f} FPS", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        out_writer.write(frame)

    out_writer.release()
    print(f"\n[DONE] Annotated video saved successfully to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Predict football trajectory on MP4 video with physics filters")
    parser.add_argument("--video", required=True, help="Path to input video (.mp4)")
    parser.add_argument("--weights", default="models/football_torch/tracknetv4_best.pt", help="Path to .pt model weights")
    parser.add_argument("--thresh", type=float, default=0.5, help="Heatmap detection threshold (default: 0.5)")
    parser.add_argument("--max_disp", type=float, default=80.0, help="Max physical pixel displacement per frame (default: 80.0px)")
    parser.add_argument("--max_gap", type=int, default=3, help="Max lost frames before resetting trajectory trail (default: 3)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for GPU inference (default: 8)")
    parser.add_argument("--output_dir", default="output_predictions", help="Directory to save tracked video")

    args = parser.parse_args()
    predict_video(
        video_path=args.video,
        weights_path=args.weights,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        thresh=args.thresh,
        max_disp=args.max_disp,
        max_gap=args.max_gap,
    )


if __name__ == "__main__":
    main()
