# =============================================================================
# utils.py — Drawing helpers, FPS counter, colour palette
# =============================================================================

from __future__ import annotations

import colorsys
import os
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config


# ── Colour Palette ────────────────────────────────────────────────────────────

def _generate_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate *n* visually distinct BGR colours using the HSV wheel."""
    colors: List[Tuple[int, int, int]] = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR
    return colors


# Pre-generate 80 colours (enough for COCO classes)
_PALETTE = _generate_colors(80)


def class_color(class_id: int) -> Tuple[int, int, int]:
    """Return a consistent BGR colour for a given class ID."""
    return _PALETTE[class_id % len(_PALETTE)]


# ── FPS Counter ───────────────────────────────────────────────────────────────

class FPSCounter:
    """Rolling-average FPS counter using a deque of timestamps."""

    def __init__(self, window: int = 30):
        self._timestamps: deque = deque(maxlen=window)

    def tick(self) -> None:
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


# ── Overlay Helpers ───────────────────────────────────────────────────────────

def _overlay_rect(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    """Draw a semi-transparent filled rectangle on *frame* in-place."""
    sub = frame[y1:y2, x1:x2]
    if sub.size == 0:
        return
    rect = np.full_like(sub, color, dtype=np.uint8)
    cv2.addWeighted(rect, alpha, sub, 1 - alpha, 0, sub)
    frame[y1:y2, x1:x2] = sub


def draw_detection(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    class_name: str,
    confidence: float,
    track_id: Optional[int] = None,
    class_id: int = 0,
) -> None:
    """
    Draw a bounding box with a label on *frame*.

    Label format: "ClassName  #ID  conf%"
    """
    x1, y1, x2, y2 = bbox
    color = class_color(class_id)

    # ── Bounding box ──────────────────────────────────────────────────
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)

    # ── Build label text ──────────────────────────────────────────────
    label_parts = [class_name.capitalize()]
    if config.SHOW_TRACK_ID and track_id is not None:
        label_parts.append(f"#{track_id}")
    if config.SHOW_CONFIDENCE:
        label_parts.append(f"{confidence:.0%}")
    label = "  ".join(label_parts)

    # ── Label background ──────────────────────────────────────────────
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(
        label, font, config.FONT_SCALE, config.FONT_THICKNESS
    )
    lx1 = x1
    ly1 = max(y1 - th - baseline - 6, 0)
    lx2 = min(x1 + tw + 8, frame.shape[1])
    ly2 = max(y1, th + baseline + 6)

    _overlay_rect(frame, lx1, ly1, lx2, ly2, color, config.LABEL_ALPHA)

    # ── Label text ────────────────────────────────────────────────────
    cv2.putText(
        frame,
        label,
        (lx1 + 4, ly2 - baseline - 2),
        font,
        config.FONT_SCALE,
        (255, 255, 255),
        config.FONT_THICKNESS,
        cv2.LINE_AA,
    )


def draw_stats_panel(
    frame: np.ndarray,
    fps: float,
    object_count: int,
    confidence: float,
    model_name: str,
    paused: bool,
) -> None:
    """Draw a semi-transparent stats panel in the top-left corner."""
    lines = [
        f"  FPS: {fps:5.1f}",
        f"  Objects: {object_count}",
        f"  Conf: {confidence:.0%}",
        f"  Model: {model_name}",
    ]
    if paused:
        lines.insert(0, "  ⏸  PAUSED")

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = config.FONT_SCALE * 1.05
    thickness = config.FONT_THICKNESS
    line_height = 26
    padding = 8

    panel_w = 240
    panel_h = line_height * len(lines) + padding * 2

    _overlay_rect(frame, 10, 10, 10 + panel_w, 10 + panel_h, (20, 20, 20), config.STATS_ALPHA)

    for i, line in enumerate(lines):
        y = 10 + padding + line_height * i + 16
        cv2.putText(frame, line, (12, y), font, scale, (200, 255, 200), thickness, cv2.LINE_AA)


def draw_help_bar(frame: np.ndarray) -> None:
    """Draw a one-line keyboard shortcuts bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    help_text = "  Q: Quit   S: Screenshot   P: Pause   +/-: Confidence   C: Class Panel"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    bar_h = 24

    _overlay_rect(frame, 0, h - bar_h, w, h, (0, 0, 0), 0.65)
    cv2.putText(
        frame,
        help_text,
        (8, h - 7),
        font,
        scale,
        (200, 200, 200),
        thickness,
        cv2.LINE_AA,
    )


def draw_class_panel(
    frame: np.ndarray,
    detections: list,
) -> None:
    """
    Draw a panel on the right side showing live per-class detection counts.
    Each class is shown with its colour dot and count.
    """
    if not detections:
        return

    # Count detections per class
    from collections import Counter
    class_counts = Counter(det.class_name for det in detections)
    # Sort alphabetically
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[0])

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    line_h = 22
    padding = 8
    dot_r = 6

    panel_w = 180
    panel_h = line_h * len(sorted_classes) + padding * 2 + 20
    h, w = frame.shape[:2]

    px1 = w - panel_w - 10
    py1 = 10
    px2 = w - 10
    py2 = py1 + panel_h

    _overlay_rect(frame, px1, py1, px2, py2, (20, 20, 20), 0.55)

    # Header
    cv2.putText(
        frame,
        " Detected Classes",
        (px1 + 4, py1 + 16),
        font,
        0.42,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    for i, (class_name, count) in enumerate(sorted_classes):
        y = py1 + padding + 20 + line_h * i + 12
        # Look up class_id for colour — use the first matching detection
        class_id = next(
            (det.class_id for det in detections if det.class_name == class_name), 0
        )
        color = class_color(class_id)

        # Coloured dot
        cx = px1 + 12
        cv2.circle(frame, (cx, y - 3), dot_r, color, -1, cv2.LINE_AA)

        # Class name + count
        label = f" {class_name.capitalize()}: {count}"
        cv2.putText(
            frame,
            label,
            (cx + dot_r + 2, y),
            font,
            scale,
            (230, 230, 230),
            thickness,
            cv2.LINE_AA,
        )


# ── Screenshot ────────────────────────────────────────────────────────────────

def save_screenshot(frame: np.ndarray) -> str:
    """Save *frame* as a timestamped PNG in the screenshots directory."""
    os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(config.SCREENSHOT_DIR, f"detection_{ts}.png")
    cv2.imwrite(path, frame)
    return path


# ── Detection → Tracker ID lookup ────────────────────────────────────────────

def find_track_id(
    bbox: Tuple[int, int, int, int],
    tracked_bboxes: Dict[int, Tuple[int, int, int, int]],
    threshold: int = 10,
) -> Optional[int]:
    """
    Return the tracker ID whose bounding box matches *bbox* within *threshold* pixels.
    Returns None if no match is found.
    """
    x1, y1, x2, y2 = bbox
    for obj_id, (tx1, ty1, tx2, ty2) in tracked_bboxes.items():
        if (
            abs(tx1 - x1) <= threshold
            and abs(ty1 - y1) <= threshold
            and abs(tx2 - x2) <= threshold
            and abs(ty2 - y2) <= threshold
        ):
            return obj_id
    return None
