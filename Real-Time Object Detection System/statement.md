# Project Statement — Real-Time Object Detection System

## 1. Problem Statement
In real-world computer vision applications—such as surveillance, industrial automation, autonomous robotics, and assistive technology—detecting objects in isolation is insufficient. Traditional object detection models process individual frames independently, leading to significant visual bounding box jitter, lost object identities across frames, and high false-positive rates when background objects interfere.

There is a critical need for an integrated, real-time video perception system that combines state-of-the-art neural object detection (YOLOv8) with lightweight temporal tracking (Centroid Tracking with Exponential Moving Average Bounding Box Smoothing) to deliver rock-solid, jitter-free object identification, persistent tracking IDs, and configurable filtering from a standard webcam feed.

---

## 2. Scope of the Project
The scope of this project encompasses:
- **Real-Time Video Capture**: Integrating webcam feeds using OpenCV (`cv2.VideoCapture`) with low-latency DirectShow bindings.
- **Deep Learning Object Detection**: Utilizing Ultralytics YOLOv8 (supporting 80 COCO classes) for fast, highly accurate multi-class object localization.
- **Temporal Multi-Object Tracking**: Implementation of a Euclidean distance-based Centroid Tracker to maintain persistent integer IDs (`#ID`) for each object across sequential video frames.
- **Bounding Box Jitter Suppression**: Applying Exponential Moving Average (EMA) filtering to smooth bounding box coordinates and eliminate frame-to-frame visual flickering.
- **Flexible Category Filtering**: Providing dynamic Whitelisting (`CLASS_FILTER`) and Blacklisting (`CLASS_EXCLUDE`) to target specific items (e.g. bottles, laptops, cell phones) while optionally ignoring unwanted background classes (e.g. person/face detection).
- **Interactive UI & Live Analytics**: Rendering on-screen performance overlays (FPS counter, tracked object count, model details, per-class breakdown panel) and live keyboard shortcuts (`Q`, `P`, `S`, `C`, `+`/`-`).
- **Comprehensive Verification & Testing**: Automated unit tests for configuration, tracking logic, smoothing math, and detection filtering.

---

## 3. Target Users
- **Surveillance & Security Personnel**: Operators requiring real-time tracking of individuals, vehicles, or unauthorized objects with stable bounding box visual overlays.
- **Industrial Automation Engineers**: Developers implementing conveyor belt item counting, inventory monitoring, or object sorting algorithms.
- **Robotics & IoT Developers**: Embedded engineers needing a modular, lightweight Python vision framework compatible with standard webcams and edge devices.
- **Academic Researchers & Students**: Students studying computer vision, object detection algorithms (YOLOv8), multi-object tracking, and digital signal smoothing.

---

## 4. High-Level Features
1. **YOLOv8 Real-Time Inference**: High-speed object recognition supporting 80 standard COCO categories (vehicles, electronics, personal items, animals, etc.).
2. **Centroid Tracking with Persistent IDs**: Assigns unique, continuous tracking IDs (`#1`, `#2`, ...) to detected objects even when they move across the webcam frame.
3. **EMA Bounding Box Stabilization**: Mathematical exponential smoothing (`BBOX_SMOOTH_ALPHA`) that stabilizes bounding boxes and eliminates visual flickering.
4. **Dual-Layer Class Filtering**:
   - `CLASS_FILTER`: Whitelist to detect only designated target items.
   - `CLASS_EXCLUDE`: Blacklist to ignore specific categories (e.g. excluding "person" to avoid false face/body triggers when inspecting items).
5. **Live HUD & Per-Class Analytics Panel**: Real-time FPS monitoring, object counter, current confidence threshold display, and an optional right-side per-class breakdown panel (`C` key toggle).
6. **Live Interactive Controls**:
   - `Q` / `ESC`: Safe program termination
   - `P`: Pause / Resume live feed
   - `S`: Instant high-resolution screenshot saving (`screenshots/` folder)
   - `+` / `-`: On-the-fly detection confidence threshold adjustment
   - `C`: Toggle live class breakdown panel
7. **Modular & Extensible Architecture**: Separated into dedicated modules (`config`, `detector`, `tracker`, `utils`, `main`, `test_system`) adhering to clean code standards.
