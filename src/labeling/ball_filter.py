"""
Physical filtering and trajectory smoothing for ball tracking.
"""

import numpy as np
import cv2


class BallTrajectoryFilter:
    """
    Validates detected ball candidates and filters trajectories using physical constraints.
    """
    def __init__(
        self,
        min_area: int = 15,
        max_area: int = 4000,
        min_circularity: float = 0.25,
        min_aspect_ratio: float = 0.40,
        max_speed_px: float = 220.0,
        max_gap_interpolation: int = 5,
        min_y_ratio: float = 0.05,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.min_aspect_ratio = min_aspect_ratio
        self.max_speed_px = max_speed_px
        self.max_gap_interpolation = max_gap_interpolation
        self.min_y_ratio = min_y_ratio

    def extract_centroid_from_mask(self, mask: np.ndarray):
        """
        Extract the centroid (x, y) and area of the largest valid contour in a binary mask.
        Returns: (x, y, area, is_valid)
        """
        if mask is None or not np.any(mask):
            return -1, -1, 0, False

        h_mask, w_mask = mask.shape[:2]
        mask_u8 = (mask * 255).astype(np.uint8) if mask.dtype != np.uint8 else mask
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return -1, -1, 0, False

        # Find contour with largest area
        best_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best_cnt)

        if area < self.min_area or area > self.max_area:
            return -1, -1, area, False

        # Check circularity and aspect ratio (tolerating motion blur elongation)
        perimeter = cv2.arcLength(best_cnt, True)
        if perimeter > 0:
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                _, _, w, h = cv2.boundingRect(best_cnt)
                ratio = min(w, h) / max(w, h)
                if ratio < self.min_aspect_ratio:
                    return -1, -1, area, False

        # Calculate moments for center
        M = cv2.moments(best_cnt)
        if M["m00"] > 0:
            cx = int(round(M["m10"] / M["m00"]))
            cy = int(round(M["m01"] / M["m00"]))

            # Reject detections in extreme top border (ceiling / stadium lights)
            if cy < int(h_mask * self.min_y_ratio):
                return -1, -1, area, False

            return cx, cy, area, True

        return -1, -1, area, False

    def smooth_trajectory(self, records: list) -> list:
        """
        Post-process a list of frame records.
        Each record is a dict with keys: 'frame_idx', 'file_name', 'visibility', 'x', 'y', 'status'
        Performs:
        1. Speed outlier rejection.
        2. Short gap interpolation (1 to max_gap_interpolation frames).
        """
        cleaned = [dict(r) for r in records]
        n = len(cleaned)
        if n == 0:
            return cleaned

        # 1. Outlier rejection based on impossible speed
        last_good_idx = -1
        for i in range(n):
            if cleaned[i]["visibility"] == 1:
                x, y = cleaned[i]["x"], cleaned[i]["y"]
                if last_good_idx != -1:
                    prev_x, prev_y = cleaned[last_good_idx]["x"], cleaned[last_good_idx]["y"]
                    dt = i - last_good_idx
                    dist = np.hypot(x - prev_x, y - prev_y)
                    # If ball jumped further than allowed speed * time, reject it
                    if dist > self.max_speed_px * dt:
                        cleaned[i]["visibility"] = 0
                        cleaned[i]["x"] = -1
                        cleaned[i]["y"] = -1
                        continue
                last_good_idx = i

        # 2. Interpolate brief occlusions (1 to max_gap_interpolation frames)
        i = 0
        while i < n:
            if cleaned[i]["visibility"] == 1:
                # Look for next visible
                next_idx = -1
                for j in range(i + 1, min(i + self.max_gap_interpolation + 2, n)):
                    if cleaned[j]["visibility"] == 1:
                        next_idx = j
                        break

                if next_idx != -1 and (next_idx - i) > 1:
                    gap = next_idx - i
                    x1, y1 = cleaned[i]["x"], cleaned[i]["y"]
                    x2, y2 = cleaned[next_idx]["x"], cleaned[next_idx]["y"]
                    # Interpolate
                    for k in range(1, gap):
                        t = k / gap
                        cleaned[i + k]["visibility"] = 1
                        cleaned[i + k]["x"] = int(round(x1 + t * (x2 - x1)))
                        cleaned[i + k]["y"] = int(round(y1 + t * (y2 - y1)))
                    i = next_idx
                    continue
            i += 1

        return cleaned
