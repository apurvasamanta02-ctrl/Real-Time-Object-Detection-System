# =============================================================================
# config.py — Central configuration for Real-Time Object Detection System
# =============================================================================

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "yolov8s.pt"       # Options: yolov8n (fast), yolov8s (balanced), yolov8m/l/x (accurate)
CONFIDENCE_THRESHOLD = 0.40     # Raised back to 0.40 to reduce false detections
IOU_THRESHOLD = 0.45            # Non-Maximum Suppression IoU threshold
DEVICE = ""                     # "" = auto (GPU if available, else CPU)

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0                # Webcam index (0 = default/built-in webcam)
FRAME_WIDTH = 1280              # Capture width  (pixels); 0 = use camera default
FRAME_HEIGHT = 720              # Capture height (pixels); 0 = use camera default
TARGET_FPS = 30                 # Target camera FPS (best-effort)

# ── Class Filter ──────────────────────────────────────────────────────────────
# Leave CLASS_FILTER empty [] to detect ALL 80 COCO classes.
# To detect only specific classes, list their names here (lowercase).
# Full list of all 80 classes is shown below for reference.
CLASS_FILTER = []               # e.g. ["person", "car", "dog"] — empty = detect ALL

# Classes to NEVER detect (blacklist). Add any class name you want to ignore.
# e.g. CLASS_EXCLUDE = ["person"] to stop detecting faces/people in the scene
CLASS_EXCLUDE = []              # e.g. ["person"] — empty = exclude nothing

# All 80 detectable COCO classes (for reference):
# person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
# traffic light, fire hydrant, stop sign, parking meter, bench,
# bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe,
# backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard,
# sports ball, kite, baseball bat, baseball glove, skateboard, surfboard,
# tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl,
# banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza,
# donut, cake, chair, couch, potted plant, bed, dining table, toilet,
# tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven,
# toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear,
# hair drier, toothbrush

# ── Display ───────────────────────────────────────────────────────────────────
WINDOW_TITLE = "Real-Time Object Detection  |  YOLOv8s  |  Press Q to quit"
SHOW_CONFIDENCE = True          # Show confidence score on bounding box label
SHOW_TRACK_ID = True            # Show tracker ID on bounding box label
SHOW_STATS_PANEL = True         # Show FPS + object count panel (top-left)
SHOW_HELP_BAR = True            # Show keyboard shortcuts bar (bottom)
SHOW_CLASS_PANEL = True         # Show per-class detection count panel (right side)

# ── Tracker ───────────────────────────────────────────────────────────────────
MAX_DISAPPEARED = 20            # Frames an object can vanish before ID is dropped
MAX_DISTANCE = 100              # Max centroid distance (px) to link detections
BBOX_SMOOTH_ALPHA = 0.35        # Bounding box smoothing (0=very smooth/laggy, 1=no smoothing/jittery)

# ── Appearance ────────────────────────────────────────────────────────────────
BOX_THICKNESS = 2               # Bounding box line thickness
FONT_SCALE = 0.55               # Label font scale
FONT_THICKNESS = 1              # Label font thickness
LABEL_ALPHA = 0.6               # Background rectangle transparency for labels
STATS_ALPHA = 0.5               # Background rectangle transparency for stats panel

# ── Screenshot ────────────────────────────────────────────────────────────────
SCREENSHOT_DIR = "screenshots"  # Directory to save screenshots (created if absent)
