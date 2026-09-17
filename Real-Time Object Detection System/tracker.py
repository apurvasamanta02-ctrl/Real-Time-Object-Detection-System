# =============================================================================
# tracker.py — Centroid-based multi-object tracker with EMA bbox smoothing
# =============================================================================

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import distance as dist

import config


class CentroidTracker:
    """
    Assigns a persistent integer ID to each detected object by matching
    centroids across consecutive frames using Euclidean distance.

    Bounding boxes are smoothed with Exponential Moving Average (EMA)
    to eliminate the jitter/flickering caused by frame-to-frame
    detection variation.

    Usage
    -----
    tracker = CentroidTracker()
    # Each frame:
    objects = tracker.update(detections)   # dict {id: (x1,y1,x2,y2)}
    """

    def __init__(
        self,
        max_disappeared: int = config.MAX_DISAPPEARED,
        max_distance: float = config.MAX_DISTANCE,
        smooth_alpha: float = config.BBOX_SMOOTH_ALPHA,
    ):
        self.next_object_id: int = 0
        self.objects: OrderedDict[int, np.ndarray] = OrderedDict()   # id → centroid
        self.bboxes: OrderedDict[int, np.ndarray] = OrderedDict()    # id → raw bbox (float array)
        self.smoothed: OrderedDict[int, np.ndarray] = OrderedDict()  # id → smoothed bbox (float array)
        self.disappeared: OrderedDict[int, int] = OrderedDict()

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.alpha = smooth_alpha   # EMA weight for new detection (lower = smoother)

    # ------------------------------------------------------------------
    def _bbox_to_array(self, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        return np.array(bbox, dtype=float)

    def _register(self, centroid: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        arr = self._bbox_to_array(bbox)
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = arr
        self.smoothed[self.next_object_id] = arr.copy()   # start smoothed = raw
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def _deregister(self, object_id: int) -> None:
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.smoothed[object_id]
        del self.disappeared[object_id]

    def _int_bbox(self, arr: np.ndarray) -> Tuple[int, int, int, int]:
        return tuple(arr.astype(int).tolist())  # type: ignore[return-value]

    # ------------------------------------------------------------------
    def update(
        self, detections: List  # List[detector.Detection]
    ) -> Dict[int, Tuple[int, int, int, int]]:
        """
        Update tracker with this frame's detections.

        Parameters
        ----------
        detections : list of Detection namedtuples

        Returns
        -------
        dict mapping object_id → (x1, y1, x2, y2)  — smoothed coordinates
        """
        # ── No detections ──────────────────────────────────────────────
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return {oid: self._int_bbox(self.smoothed[oid]) for oid in self.smoothed}

        # Compute input centroids
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        input_bboxes: List[Tuple[int, int, int, int]] = []

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            input_centroids[i] = (cx, cy)
            input_bboxes.append(det.bbox)

        # ── No existing objects — register all ─────────────────────────
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self._register(input_centroids[i], input_bboxes[i])
            return {oid: self._int_bbox(self.smoothed[oid]) for oid in self.smoothed}

        # ── Match existing objects to new detections ───────────────────
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = dist.cdist(np.array(object_centroids), input_centroids)

        # Rows = sorted by min value; cols likewise
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows: set = set()
        used_cols: set = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            obj_id = object_ids[row]
            new_bbox = self._bbox_to_array(input_bboxes[col])

            # ── EMA smoothing: blend new detection with history ────────
            old_smooth = self.smoothed[obj_id]
            self.smoothed[obj_id] = self.alpha * new_bbox + (1.0 - self.alpha) * old_smooth

            self.objects[obj_id] = input_centroids[col]
            self.bboxes[obj_id] = new_bbox
            self.disappeared[obj_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(D.shape[0])) - used_rows
        unused_cols = set(range(D.shape[1])) - used_cols

        # Increment disappeared count for unmatched existing objects
        if D.shape[0] >= D.shape[1]:
            for row in unused_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
        else:
            # Register new objects for unmatched detections
            for col in unused_cols:
                self._register(input_centroids[col], input_bboxes[col])

        return {oid: self._int_bbox(self.smoothed[oid]) for oid in self.smoothed}

    # ------------------------------------------------------------------
    @property
    def count(self) -> int:
        """Number of currently tracked objects."""
        return len(self.objects)
