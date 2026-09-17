# Real-Time Object Detection and Tracking System

An end-to-end computer vision application built with **Python**, **OpenCV**, and **YOLOv8** (Ultralytics) featuring a custom **Centroid Tracker** enhanced with **Exponential Moving Average (EMA) Bounding Box Smoothing** for real-time item detection and tracking from live webcam feeds.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & System Flow](#architecture--system-flow)
- [Technologies & Tools Used](#technologies--tools-used)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Usage & Keyboard Controls](#usage--keyboard-controls)
- [Configuration Reference](#configuration-reference)
- [Testing & Validation](#testing--validation)
- [Screenshots & Output](#screenshots--output)
- [License & Acknowledgments](#license--acknowledgments)

---

## 🔍 Overview
Detecting objects in static images is straightforward, but real-time video streams present challenges such as bounding box jitter, identity swapping, flickering detections, and false triggers.

This project addresses these challenges by integrating:
1. **YOLOv8 Neural Network**: High-speed, high-accuracy object detection supporting 80 COCO categories.
2. **Centroid Tracking Algorithm**: Maintains persistent unique numerical IDs (`#ID`) for each object across sequential frames.
3. **EMA Bounding Box Smoothing**: Mathematical temporal filtering (`BBOX_SMOOTH_ALPHA`) that eliminates box flickering and jitter.
4. **Dual Filtering System**: Allows whitelisting (`CLASS_FILTER`) or blacklisting (`CLASS_EXCLUDE`) to focus on relevant items (e.g. bottles, laptops) while ignoring unwanted categories.

---

## ⭐ Key Features
- 🚀 **Real-Time YOLOv8 Detection**: Supports YOLOv8 models (`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`) for balanced speed and accuracy.
- 🏷️ **Persistent Object Tracking**: Assigns unique integer IDs to objects as they move across the camera frame.
- 🎯 **Bounding Box Stabilization**: EMA smoothing suppresses frame-to-frame detection jitter.
- 🛡️ **Class Filtering & Exclusion**:
  - `CLASS_FILTER`: Detect only target categories (e.g., `["bottle", "cell phone"]`).
  - `CLASS_EXCLUDE`: Ignore specific categories (e.g., `["person"]` to eliminate face detection false positives).
- 📊 **Live On-Screen Analytics**:
  - Top-left HUD: Real-time FPS, total tracked object count, model name, active confidence threshold.
  - Right-side Class Breakdown Panel (`C` key): Live breakdown of detected items with color-coded indicators.
  - Bottom Shortcut Bar: Quick keyboard reference guide.
- ⌨️ **Interactive Controls**: Dynamic runtime threshold adjustments (`+`/`-`), instant screenshots (`S`), feed pausing (`P`), and class panel toggling (`C`).

---

## 🏗️ Architecture & System Flow

```
+------------------+     +-----------------------+     +-------------------------+
| Live Webcam Feed | --> | YOLOv8 Neural Network | --> | Raw Detections (bboxes) |
+------------------+     +-----------------------+     +-------------------------+
                                                                    |
                                                                    v
+------------------+     +-----------------------+     +-------------------------+
| Display & HUD    | <-- | EMA Box Smoothing     | <-- | Centroid Tracker        |
| Render Overlay   |     | (Jitter Suppression)  |     | (ID Assignment)         |
+------------------+     +-----------------------+     +-------------------------+
```

---

## 🛠️ Technologies & Tools Used
- **Programming Language**: Python 3.9+
- **Computer Vision Framework**: OpenCV (`opencv-python`)
- **Deep Learning Model**: Ultralytics YOLOv8 (`ultralytics`, PyTorch backend)
- **Scientific & Math Libraries**: NumPy, SciPy (Spatial Distance matching)
- **Testing Framework**: Python `unittest`

---

## 📁 Repository Structure
```
Real-Time Object Detection System/
├── main.py            # Main application entry point & video render loop
├── detector.py        # YOLODetector wrapper class with filter logic
├── tracker.py         # CentroidTracker with EMA bounding box smoothing
├── utils.py           # Drawing utilities, FPS counter, HUD overlays, screenshot saver
├── config.py          # Centralized configuration parameters
├── test_system.py     # Automated unit & validation test suite
├── statement.md       # Project statement, problem, scope & target users
├── requirements.txt   # Python dependency specifications
├── .gitignore         # Git ignore rules for cached models and outputs
└── README.md          # Comprehensive project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/real-time-object-detection-system.git
cd "real-time-object-detection-system"
```

### 2. Create & Activate Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🎮 Usage & Keyboard Controls

Run the main application:
```bash
python main.py
```

> **Note**: On the first run, Ultralytics will automatically download the YOLOv8 weights file (e.g., `yolov8s.pt`, ~22 MB).

### Interactive Keyboard Controls
| Key | Action |
|---|---|
| `Q` / `ESC` | Quit the application safely |
| `P` | Pause / Resume the video feed |
| `S` | Capture & save screenshot to `screenshots/` directory |
| `C` | Toggle right-side per-class breakdown panel on/off |
| `+` / `=` | Increase detection confidence threshold by 5% |
| `-` / `_` | Decrease detection confidence threshold by 5% |

---

## ⚙️ Configuration Reference (`config.py`)

All operational settings can be tuned in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `MODEL_NAME` | `"yolov8s.pt"` | YOLOv8 model size (`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`) |
| `CONFIDENCE_THRESHOLD` | `0.40` | Minimum detection confidence score (0.0 to 1.0) |
| `IOU_THRESHOLD` | `0.45` | Non-Maximum Suppression (NMS) IoU threshold |
| `CAMERA_INDEX` | `0` | Camera device index (0 = default webcam) |
| `CLASS_FILTER` | `[]` | Whitelist array of class names to detect (empty = detect all 80 COCO classes) |
| `CLASS_EXCLUDE` | `[]` | Blacklist array of class names to ignore (e.g., `["person"]`) |
| `BBOX_SMOOTH_ALPHA` | `0.35` | EMA smoothing factor (lower = smoother, higher = faster response) |
| `MAX_DISAPPEARED` | `20` | Frames an object can be missing before ID is deregistered |
| `MAX_DISTANCE` | `100` | Maximum centroid pixel distance for tracking frame-to-frame matching |

---

## 🧪 Testing & Validation

Run the automated test suite to verify all modules:
```bash
python test_system.py
```

Expected Output:
```text
......
----------------------------------------------------------------------
Ran 6 tests in 0.001s

OK
```

---

## 📸 Screenshots & Output
Screenshots captured using the `S` hotkey are saved automatically in the `screenshots/` folder with timestamped filenames (`detection_YYYYMMDD_HHMMSS.png`).

---

## 📜 License & Acknowledgments
- **Dataset / Weights**: Pre-trained on the COCO (Common Objects in Context) dataset.
- **YOLOv8**: Provided by Ultralytics LLC under the AGPL-3.0 License.
