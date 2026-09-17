# =============================================================================
# main.py — Real-Time Object Detection System
#            Python · OpenCV · YOLOv8 · Centroid Tracker
# =============================================================================
# Usage:  python main.py
# Keys:   Q = quit  |  S = screenshot  |  P = pause/resume
#         + / = = raise confidence threshold by 5 %
#         - / _ = lower confidence threshold by 5 %
# =============================================================================

from __future__ import annotations

import sys
import time

import cv2

import config
from detector import YOLODetector
from tracker import CentroidTracker
from utils import (
    FPSCounter,
    draw_detection,
    draw_help_bar,
    draw_stats_panel,
    draw_class_panel,
    find_track_id,
    save_screenshot,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    """Open the webcam and apply resolution / FPS hints."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)   # CAP_DSHOW = faster on Windows
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)               # fallback (Linux / macOS)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {index}.")
        sys.exit(1)

    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)

    return cap


def print_banner(model_name: str) -> None:
    print("\n" + "=" * 60)
    print("  Real-Time Object Detection System")
    print(f"  Model : {model_name}")
    print(f"  Camera: index {config.CAMERA_INDEX}")
    print("  Keys  : Q=quit  S=screenshot  P=pause  +/-=confidence")
    print("=" * 60 + "\n")


# ── Main Loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Initialise ────────────────────────────────────────────────────
    print("[INFO] Loading YOLOv8 model …")
    detector = YOLODetector()
    tracker  = CentroidTracker()
    fps_ctr  = FPSCounter(window=30)

    print_banner(config.MODEL_NAME)

    cap = open_camera(
        config.CAMERA_INDEX,
        config.FRAME_WIDTH,
        config.FRAME_HEIGHT,
        config.TARGET_FPS,
    )

    cv2.namedWindow(config.WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_TITLE, config.FRAME_WIDTH or 1280, config.FRAME_HEIGHT or 720)

    paused = False
    show_class_panel = config.SHOW_CLASS_PANEL
    last_screenshot_msg: str = ""
    msg_until: float = 0.0

    # ── Loop ──────────────────────────────────────────────────────────
    while True:
        # --- Keyboard input (1 ms poll) ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:           # Q or Esc → quit
            break
        elif key == ord("p"):                       # P → pause/resume
            paused = not paused
        elif key == ord("c"):                       # C → toggle class panel
            show_class_panel = not show_class_panel
            print(f"[INFO] Class panel {'ON' if show_class_panel else 'OFF'}")
        elif key == ord("s"):                       # S → screenshot
            ret, snap = cap.read()
            if ret:
                path = save_screenshot(snap)
                last_screenshot_msg = f"Saved: {path}"
                msg_until = time.time() + 3.0
                print(f"[INFO] {last_screenshot_msg}")
        elif key in (ord("+"), ord("=")):           # + → raise confidence
            detector.set_confidence(detector.confidence + 0.05)
            print(f"[INFO] Confidence → {detector.confidence:.0%}")
        elif key in (ord("-"), ord("_")):           # - → lower confidence
            detector.set_confidence(detector.confidence - 0.05)
            print(f"[INFO] Confidence → {detector.confidence:.0%}")

        # --- Grab frame ---
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame grab failed — retrying …")
            time.sleep(0.05)
            continue

        if paused:
            # Still render the last frame with paused indicator
            if config.SHOW_STATS_PANEL:
                draw_stats_panel(frame, fps_ctr.fps, tracker.count, detector.confidence, config.MODEL_NAME, paused=True)
            if config.SHOW_HELP_BAR:
                draw_help_bar(frame)
            cv2.imshow(config.WINDOW_TITLE, frame)
            continue

        # --- Detect ---
        detections = detector.detect(frame)

        # --- Track ---
        tracked_bboxes = tracker.update(detections)

        # --- Draw detections ---
        for det in detections:
            track_id = find_track_id(det.bbox, tracked_bboxes)
            draw_detection(
                frame,
                bbox=det.bbox,
                class_name=det.class_name,
                confidence=det.confidence,
                track_id=track_id,
                class_id=det.class_id,
            )

        # --- Stats / help overlays ---
        fps_ctr.tick()

        if config.SHOW_STATS_PANEL:
            draw_stats_panel(
                frame,
                fps=fps_ctr.fps,
                object_count=tracker.count,
                confidence=detector.confidence,
                model_name=config.MODEL_NAME,
                paused=False,
            )

        if show_class_panel:
            draw_class_panel(frame, detections)

        if config.SHOW_HELP_BAR:
            draw_help_bar(frame)

        # --- Screenshot notification ---
        if last_screenshot_msg and time.time() < msg_until:
            cv2.putText(
                frame,
                last_screenshot_msg,
                (10, frame.shape[0] - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 120),
                1,
                cv2.LINE_AA,
            )

        # --- Show ---
        cv2.imshow(config.WINDOW_TITLE, frame)

    # ── Cleanup ───────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Bye!")


if __name__ == "__main__":
    main()
