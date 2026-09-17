# =============================================================================
# detector.py — YOLOv8 detection wrapper
# =============================================================================

from __future__ import annotations

from collections import namedtuple
from typing import List

import numpy as np

import config

# A single detection result
Detection = namedtuple("Detection", ["bbox", "class_id", "class_name", "confidence"])
# bbox = (x1, y1, x2, y2) in pixel coordinates


class YOLODetector:
    """Wraps Ultralytics YOLOv8 and exposes a simple detect() interface."""

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        confidence: float = config.CONFIDENCE_THRESHOLD,
        iou: float = config.IOU_THRESHOLD,
        device: str = config.DEVICE,
    ):
        from ultralytics import YOLO  # imported here to keep startup fast

        self.model = YOLO(model_name)
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.class_names: List[str] = list(self.model.names.values())

        # Warm-up: run one dummy inference so the first real frame isn't slow
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self.model.predict(
            dummy,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLOv8 on *frame* (BGR numpy array).

        Returns a list of Detection namedtuples sorted by confidence (descending).
        Applies CLASS_FILTER from config if non-empty.
        """
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        # Build active filter and exclude sets (lowercase)
        filter_set = (
            {c.lower() for c in config.CLASS_FILTER}
            if config.CLASS_FILTER
            else set()
        )
        exclude_set = (
            {c.lower() for c in getattr(config, "CLASS_EXCLUDE", [])}
            if getattr(config, "CLASS_EXCLUDE", [])
            else set()
        )

        detections: List[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # Coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = self.class_names[class_id]

                c_name_lower = class_name.lower()

                # Apply class exclude
                if exclude_set and c_name_lower in exclude_set:
                    continue

                # Apply class filter
                if filter_set and c_name_lower not in filter_set:
                    continue

                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                    )
                )

        # Sort highest confidence first
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # ------------------------------------------------------------------
    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def set_confidence(self, value: float) -> None:
        """Update confidence threshold at runtime (clamped to [0.05, 0.95])."""
        self.confidence = max(0.05, min(0.95, value))
