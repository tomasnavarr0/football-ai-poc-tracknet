"""
Football Ball Label Checker & Corrector
Interactive desktop tool to visually review and correct ball annotations for TrackNetV4.

Controls:
  Mouse:
    - Left Click: Place/Correct ball center (x, y) -> visibility=1
    - Right Click: Remove ball / Mark occluded -> visibility=0, (0,0)
    - Mouse Move: Real-time 3x zoom magnifier loupe

  Navigation:
    - Right Arrow / D: Next frame
    - Left Arrow / A:  Previous frame
    - Down Arrow / S:  Next clip (auto-saves edits)
    - Up Arrow / W:    Previous clip (auto-saves edits)
    - Space:           Play/Pause continuous playback

  Correction Tools:
    - I: Interpolate trajectory between nearest labeled frames in this clip
    - X / Delete: Mark current frame as not visible
    - R: Reload / Discard unsaved changes for this clip
    - F: Jump to next suspicious/low-detection clip (< 10 visible frames)
    - Z: Toggle zoom magnifier loupe
    - H: Toggle help overlay
    - Ctrl+S: Save current clip labels
    - Q / Esc: Save and exit
"""

import argparse
import csv
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class LabelCorrector:
    def __init__(
        self,
        dataset_dir: str = "Dataset/football",
        video_filter: Optional[str] = None,
        clip_filter: Optional[str] = None,
        flagged_only: bool = False,
        display_width: int = 1440,
    ):
        self.dataset_dir = dataset_dir
        self.display_width = display_width
        self.show_loupe = True
        self.show_help = False
        self.show_trail = True
        self.playing = False
        self.play_fps = 25
        self.mouse_pos = (0, 0)
        self.is_modified = False

        # Discover all clips containing Label.csv
        self.clips = self._discover_clips(dataset_dir, video_filter, clip_filter)
        if not self.clips:
            print(f"Error: No clips with Label.csv found in {dataset_dir}!")
            sys.exit(1)

        self.clip_idx = 0
        self.frame_idx = 0

        # If flagged_only, find first flagged clip
        if flagged_only:
            self._jump_to_next_flagged(start_idx=0)

        # Load first clip
        self._load_current_clip()

    def _discover_clips(
        self,
        dataset_dir: str,
        video_filter: Optional[str],
        clip_filter: Optional[str],
    ) -> List[Tuple[str, str, str]]:
        """
        Returns list of tuples: (video_name, clip_name, clip_dir)
        sorted naturally.
        """
        all_labels = glob.glob(os.path.join(dataset_dir, "**", "Label.csv"), recursive=True)
        discovered = []

        for lpath in all_labels:
            clip_dir = os.path.dirname(lpath)
            rel = os.path.relpath(clip_dir, dataset_dir)
            parts = rel.split(os.sep)
            if len(parts) >= 2:
                video_name = parts[0]
                clip_name = parts[1]
            else:
                video_name = "default"
                clip_name = parts[0]

            if video_filter and video_filter.lower() not in video_name.lower():
                continue
            if clip_filter and clip_filter.lower() not in clip_name.lower():
                continue

            discovered.append((video_name, clip_name, clip_dir))

        def sort_key(item):
            v, c, _ = item
            num = ""
            for ch in c:
                if ch.isdigit():
                    num += ch
            c_num = int(num) if num else 0
            return (v, c_num, c)

        discovered.sort(key=sort_key)
        print(f"[Corrector] Found {len(discovered)} clip(s) available for review.")
        return discovered

    def _load_current_clip(self):
        """Loads images and Label.csv for the active clip."""
        self.video_name, self.clip_name, self.clip_dir = self.clips[self.clip_idx]
        self.label_path = os.path.join(self.clip_dir, "Label.csv")

        # Load Label.csv
        self.records: List[Dict] = []
        if os.path.exists(self.label_path):
            with open(self.label_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.records.append({
                        "file name": row["file name"],
                        "visibility": int(row.get("visibility", 0)),
                        "x-coordinate": int(float(row.get("x-coordinate", 0))),
                        "y-coordinate": int(float(row.get("y-coordinate", 0))),
                        "status": int(row.get("status", 0)),
                    })

        # Load frame image files
        self.frame_files = sorted(glob.glob(os.path.join(self.clip_dir, "*.jpg")))
        if not self.frame_files:
            self.frame_files = sorted(glob.glob(os.path.join(self.clip_dir, "*.png")))

        # If records count differs from images, reconcile
        if len(self.records) != len(self.frame_files):
            file_to_rec = {r["file name"]: r for r in self.records}
            reconciled = []
            for ff in self.frame_files:
                fn = os.path.basename(ff)
                if fn in file_to_rec:
                    reconciled.append(file_to_rec[fn])
                else:
                    reconciled.append({
                        "file name": fn,
                        "visibility": 0,
                        "x-coordinate": 0,
                        "y-coordinate": 0,
                        "status": 0,
                    })
            self.records = reconciled

        self.num_frames = len(self.frame_files)
        self.frame_idx = min(self.frame_idx, max(0, self.num_frames - 1))
        self.is_modified = False

        # Pre-cache image dimensions from frame 0
        if self.num_frames > 0:
            sample = cv2.imread(self.frame_files[0])
            if sample is not None:
                self.img_h, self.img_w = sample.shape[:2]
                self.scale = self.display_width / float(self.img_w)
                self.disp_w = int(self.img_w * self.scale)
                self.disp_h = int(self.img_h * self.scale)
            else:
                self.img_h, self.img_w = 1080, 1920
                self.scale = 1.0
                self.disp_w, self.disp_h = 1920, 1080

    def _save_current_clip(self):
        """Saves current annotations back to Label.csv."""
        if not self.records:
            return
        with open(self.label_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["file name", "visibility", "x-coordinate", "y-coordinate", "status"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                writer.writerow({
                    "file name": r["file name"],
                    "visibility": int(r["visibility"]),
                    "x-coordinate": int(r["x-coordinate"]),
                    "y-coordinate": int(r["y-coordinate"]),
                    "status": int(r["status"]),
                })
        self.is_modified = False
        print(f"[Saved] {self.clip_name} -> {self.label_path}")

    def _interpolate_clip_trajectory(self):
        """Linearly interpolates ball (x, y) between known visible frames."""
        visible_indices = [i for i, r in enumerate(self.records) if r["visibility"] == 1]
        if len(visible_indices) < 2:
            print("[Interpolate] Need at least 2 visible frames in this clip to interpolate.")
            return

        interpolated_count = 0
        for k in range(len(visible_indices) - 1):
            start_i = visible_indices[k]
            end_i = visible_indices[k + 1]
            gap = end_i - start_i

            if 1 < gap <= 30:  # Interpolate gaps up to 30 frames (~1s)
                x0, y0 = self.records[start_i]["x-coordinate"], self.records[start_i]["y-coordinate"]
                x1, y1 = self.records[end_i]["x-coordinate"], self.records[end_i]["y-coordinate"]

                for step in range(1, gap):
                    curr_i = start_i + step
                    alpha = step / float(gap)
                    interp_x = int(round(x0 + alpha * (x1 - x0)))
                    interp_y = int(round(y0 + alpha * (y1 - y0)))

                    self.records[curr_i]["x-coordinate"] = interp_x
                    self.records[curr_i]["y-coordinate"] = interp_y
                    self.records[curr_i]["visibility"] = 1
                    interpolated_count += 1

        self.is_modified = True
        print(f"[Interpolate] Interpolated {interpolated_count} occluded frame(s) in {self.clip_name}.")

    def _jump_to_next_flagged(self, start_idx: Optional[int] = None):
        """Jumps to the next clip with low visible detection (< 10 visible frames)."""
        idx = (self.clip_idx + 1) if start_idx is None else start_idx
        found = False

        for i in range(idx, len(self.clips)):
            _, _, cdir = self.clips[i]
            lpath = os.path.join(cdir, "Label.csv")
            if os.path.exists(lpath):
                with open(lpath, "r", encoding="utf-8") as f:
                    rdr = csv.DictReader(f)
                    vis = sum(1 for row in rdr if int(row.get("visibility", 0)) == 1)
                    if vis < 10:
                        if self.is_modified:
                            self._save_current_clip()
                        self.clip_idx = i
                        self._load_current_clip()
                        found = True
                        print(f"[Flagged] Jumped to clip with low detections ({vis}/60): {self.clip_name}")
                        break

        if not found:
            print("[Flagged] No further clips with < 10 detections found.")

    def _mouse_callback(self, event, x, y, flags, param):
        """Handles click and move interactions."""
        # Convert display (x, y) back to native resolution
        orig_x = int(round(x / self.scale))
        orig_y = int(round(y / self.scale))
        orig_x = max(0, min(self.img_w - 1, orig_x))
        orig_y = max(0, min(self.img_h - 1, orig_y))
        self.mouse_pos = (orig_x, orig_y)

        if event == cv2.EVENT_LBUTTONDOWN:
            # Set ball coordinates
            if self.frame_idx < len(self.records):
                self.records[self.frame_idx]["x-coordinate"] = orig_x
                self.records[self.frame_idx]["y-coordinate"] = orig_y
                self.records[self.frame_idx]["visibility"] = 1
                self.is_modified = True
                print(f"Frame {self.frame_idx:02d}: Ball set to ({orig_x}, {orig_y}) [Visible]")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Mark not visible / occluded
            if self.frame_idx < len(self.records):
                self.records[self.frame_idx]["x-coordinate"] = 0
                self.records[self.frame_idx]["y-coordinate"] = 0
                self.records[self.frame_idx]["visibility"] = 0
                self.is_modified = True
                print(f"Frame {self.frame_idx:02d}: Ball removed [Occluded / Not visible]")

    def _render_frame(self) -> np.ndarray:
        """Draws annotations, trajectory trails, magnifier loupe, and HUD."""
        fpath = self.frame_files[self.frame_idx]
        img = cv2.imread(fpath)
        if img is None:
            img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)

        annotated = img.copy()
        curr_rec = self.records[self.frame_idx] if self.frame_idx < len(self.records) else None

        # 1. Draw Trajectory Trail (Past 10 frames)
        if self.show_trail:
            trail_pts = []
            start_t = max(0, self.frame_idx - 10)
            for t in range(start_t, self.frame_idx):
                r = self.records[t]
                if r["visibility"] == 1:
                    trail_pts.append((r["x-coordinate"], r["y-coordinate"]))

            for idx in range(len(trail_pts) - 1):
                pt1 = trail_pts[idx]
                pt2 = trail_pts[idx + 1]
                # Gradual alpha / thickness
                cv2.line(annotated, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(annotated, pt1, 4, (0, 200, 255), -1, cv2.LINE_AA)

        # 2. Draw Current Ball Target
        ball_x, ball_y = 0, 0
        is_visible = False
        if curr_rec is not None and curr_rec["visibility"] == 1:
            ball_x = curr_rec["x-coordinate"]
            ball_y = curr_rec["y-coordinate"]
            is_visible = True

            # Double ring + crosshair on ball
            cv2.circle(annotated, (ball_x, ball_y), 12, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(annotated, (ball_x, ball_y), 3, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.line(annotated, (ball_x - 18, ball_y), (ball_x + 18, ball_y), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(annotated, (ball_x, ball_y - 18), (ball_x, ball_y + 18), (0, 255, 0), 1, cv2.LINE_AA)

            # Coordinate tag
            cv2.putText(
                annotated,
                f"({ball_x}, {ball_y})",
                (ball_x + 16, ball_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # Resize to display dimensions
        disp_img = cv2.resize(annotated, (self.disp_w, self.disp_h), interpolation=cv2.INTER_LINEAR)

        # 3. Render Real-time Magnifier Loupe (3x Zoom)
        if self.show_loupe:
            # Zoom center defaults to mouse cursor if within bounds, else ball pos
            target_cx, target_cy = self.mouse_pos
            if target_cx == 0 and target_cy == 0 and is_visible:
                target_cx, target_cy = ball_x, ball_y

            crop_radius = 40  # 80x80 crop
            x1 = max(0, target_cx - crop_radius)
            y1 = max(0, target_cy - crop_radius)
            x2 = min(self.img_w, target_cx + crop_radius)
            y2 = min(self.img_h, target_cy + crop_radius)

            crop = img[y1:y2, x1:x2]
            if crop.shape[0] > 0 and crop.shape[1] > 0:
                loupe_size = 200
                loupe = cv2.resize(crop, (loupe_size, loupe_size), interpolation=cv2.INTER_NEAREST)

                # Center reticle inside loupe
                lc = loupe_size // 2
                cv2.circle(loupe, (lc, lc), 5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(loupe, (lc - 12, lc), (lc + 12, lc), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.line(loupe, (lc, lc - 12), (lc, lc + 12), (0, 255, 0), 1, cv2.LINE_AA)

                # Place loupe in top-right corner with a stylish border
                margin = 15
                lx1 = self.disp_w - loupe_size - margin
                ly1 = margin + 50
                lx2 = lx1 + loupe_size
                ly2 = ly1 + loupe_size

                # Border shadow / frame
                cv2.rectangle(disp_img, (lx1 - 3, ly1 - 3), (lx2 + 3, ly2 + 3), (20, 20, 20), -1)
                cv2.rectangle(disp_img, (lx1 - 1, ly1 - 1), (lx2 + 1, ly2 + 1), (0, 200, 255), 2)
                disp_img[ly1:ly2, lx1:lx2] = loupe

                cv2.putText(
                    disp_img,
                    f"Zoom: ({target_cx}, {target_cy})",
                    (lx1, ly2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 220, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 4. Top HUD Header
        total_visible = sum(1 for r in self.records if r["visibility"] == 1)
        mod_tag = " [MODIFIED *]" if self.is_modified else " [SAVED]"
        status_color = (0, 255, 0) if is_visible else (0, 0, 255)
        status_text = "VISIBLE" if is_visible else "NOT VISIBLE / OCCLUDED"

        # Dark header banner
        overlay = disp_img.copy()
        cv2.rectangle(overlay, (0, 0), (self.disp_w, 42), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, disp_img, 0.25, 0, disp_img)

        # Clip info text
        clip_info = f"Clip {self.clip_idx + 1}/{len(self.clips)}: {self.video_name}/{self.clip_name}"
        frame_info = f"Frame: {self.frame_idx + 1}/{self.num_frames} ({curr_rec['file name'] if curr_rec else ''})"
        stats_info = f"Clip Visible: {total_visible}/{self.num_frames}"

        cv2.putText(disp_img, clip_info, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp_img, frame_info, (420, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 255, 200), 2, cv2.LINE_AA)
        cv2.putText(disp_img, stats_info, (760, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(disp_img, status_text, (980, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2, cv2.LINE_AA)

        save_col = (0, 165, 255) if self.is_modified else (100, 255, 100)
        cv2.putText(disp_img, mod_tag, (self.disp_w - 140, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, save_col, 2, cv2.LINE_AA)

        # Bottom Quick Help Bar
        cv2.rectangle(disp_img, (0, self.disp_h - 26), (self.disp_w, self.disp_h), (15, 15, 15), -1)
        help_bar = "L-Click: Set Ball | R-Click: Occluded | A/D: Prev/Next Frame | W/S: Prev/Next Clip | Space: Play | I: Interp | Ctrl+S: Save | H: Help"
        cv2.putText(disp_img, help_bar, (15, self.disp_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # Full Help Popup Overlay (Key 'H')
        if self.show_help:
            self._render_help_overlay(disp_img)

        return disp_img

    def _render_help_overlay(self, img: np.ndarray):
        """Renders comprehensive help window popup."""
        box_w, box_h = 620, 360
        bx = (self.disp_w - box_w) // 2
        by = (self.disp_h - box_h) // 2

        overlay = img.copy()
        cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.90, img, 0.10, 0, img)
        cv2.rectangle(img, (bx, by), (bx + box_w, by + box_h), (0, 200, 255), 2)

        lines = [
            "=== Football Label Corrector Shortcuts ===",
            "Left Click          : Set / Move Ball Center (sets visibility=1)",
            "Right Click / X     : Mark Ball as Not Visible (visibility=0, x=0, y=0)",
            "A / Left Arrow      : Previous Frame",
            "D / Right Arrow     : Next Frame",
            "W / Up Arrow        : Previous Clip (auto-saves edits)",
            "S / Down Arrow      : Next Clip (auto-saves edits)",
            "Space               : Play / Pause clip video stream",
            "I                   : Interpolate trajectory between labeled frames",
            "F                   : Jump to next clip with low detections (<10 frames)",
            "Z                   : Toggle 3x Zoom Magnifier Loupe",
            "T                   : Toggle trajectory trail",
            "R                   : Revert / Reload current clip from disk",
            "Ctrl+S              : Save current clip changes to Label.csv",
            "H                   : Toggle this Help screen",
            "Q / Esc             : Save and Exit",
        ]
        for idx, line in enumerate(lines):
            col = (0, 255, 255) if idx == 0 else (240, 240, 240)
            cv2.putText(img, line, (bx + 20, by + 30 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    def run(self):
        """Main interaction loop."""
        win_name = "Football Label Corrector (SAM 3.1 & TrackNetV4)"
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win_name, self._mouse_callback)

        print("\n" + "=" * 60)
        print("  Football Label Corrector Launched")
        print("  - Click anywhere on the ball to set its center.")
        print("  - Right click to mark occluded / missing.")
        print("  - Press 'H' for full keyboard controls.")
        print("=" * 60 + "\n")

        try:
            while True:
                display_frame = self._render_frame()
                cv2.imshow(win_name, display_frame)

                delay = int(1000 / self.play_fps) if self.playing else 30
                key = cv2.waitKey(delay) & 0xFF

                # Continuous playback
                if self.playing:
                    self.frame_idx = (self.frame_idx + 1) % max(1, self.num_frames)

                if key == 255:
                    continue

                # Q or Esc: Save and quit
                if key in (ord("q"), ord("Q"), 27):
                    if self.is_modified:
                        self._save_current_clip()
                    break

                # Space: Play/Pause
                elif key == 32:
                    self.playing = not self.playing

                # Next frame (D or Right Arrow)
                elif key in (ord("d"), ord("D"), 83):
                    self.playing = False
                    if self.frame_idx < self.num_frames - 1:
                        self.frame_idx += 1

                # Previous frame (A or Left Arrow)
                elif key in (ord("a"), ord("A"), 81):
                    self.playing = False
                    if self.frame_idx > 0:
                        self.frame_idx -= 1

                # Skip 5 frames forward (] or .)
                elif key in (ord("]"), ord(".")):
                    self.playing = False
                    self.frame_idx = min(self.num_frames - 1, self.frame_idx + 5)

                # Skip 5 frames backward ([ or ,)
                elif key in (ord("["), ord(",")):
                    self.playing = False
                    self.frame_idx = max(0, self.frame_idx - 5)

                # Next clip (S or Down Arrow or 'n')
                elif key in (ord("s"), ord("S"), ord("n"), ord("N"), 84):
                    if self.is_modified:
                        self._save_current_clip()
                    if self.clip_idx < len(self.clips) - 1:
                        self.clip_idx += 1
                        self.frame_idx = 0
                        self._load_current_clip()

                # Previous clip (W or Up Arrow or 'p')
                elif key in (ord("w"), ord("W"), ord("p"), ord("P"), 82):
                    if self.is_modified:
                        self._save_current_clip()
                    if self.clip_idx > 0:
                        self.clip_idx -= 1
                        self.frame_idx = 0
                        self._load_current_clip()

                # I: Linear interpolation across occlusions
                elif key in (ord("i"), ord("I")):
                    self._interpolate_clip_trajectory()

                # X or Delete: Remove ball from current frame
                elif key in (ord("x"), ord("X")):
                    if self.frame_idx < len(self.records):
                        self.records[self.frame_idx]["x-coordinate"] = 0
                        self.records[self.frame_idx]["y-coordinate"] = 0
                        self.records[self.frame_idx]["visibility"] = 0
                        self.is_modified = True
                        print(f"Frame {self.frame_idx:02d}: Ball removed [Occluded / Not visible]")

                # F: Jump to next flagged clip (<10 visible)
                elif key in (ord("f"), ord("F")):
                    self._jump_to_next_flagged()

                # Z: Toggle zoom magnifier loupe
                elif key in (ord("z"), ord("Z")):
                    self.show_loupe = not self.show_loupe

                # T: Toggle trajectory trail
                elif key in (ord("t"), ord("T")):
                    self.show_trail = not self.show_trail

                # R: Reload / discard unsaved changes
                elif key in (ord("r"), ord("R")):
                    self._load_current_clip()
                    print(f"[Revert] Reloaded {self.clip_name} from disk.")

                # Ctrl+S: Save current clip
                elif key == 19:  # ASCII for Ctrl+S
                    self._save_current_clip()

                # H: Toggle Help screen
                elif key in (ord("h"), ord("H")):
                    self.show_help = not self.show_help

        finally:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Football Ball Label Checker and Corrector")
    parser.add_argument("--dataset_dir", default="Dataset/football", help="Path to Dataset/football directory")
    parser.add_argument("--video", default=None, help="Filter to a specific video folder (e.g. 'clip' or 'clip (1)')")
    parser.add_argument("--clip", default=None, help="Start at a specific clip name (e.g. 'Clip33')")
    parser.add_argument("--flagged", action="store_true", help="Start directly by filtering clips with < 10 visible frames")
    parser.add_argument("--width", type=int, default=1440, help="Display window width (default: 1440)")

    args = parser.parse_args()

    corrector = LabelCorrector(
        dataset_dir=args.dataset_dir,
        video_filter=args.video,
        clip_filter=args.clip,
        flagged_only=args.flagged,
        display_width=args.width,
    )
    corrector.run()


if __name__ == "__main__":
    main()
