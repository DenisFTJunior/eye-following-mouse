from __future__ import annotations

from typing import Optional

from vision.detectors.mediapipe_tracker import MediaPipeTracker
from vision.types import EyeMetrics


class YoloTracker:
    """YOLO-backed option.

    Notes:
    - If a YOLO model with eye classes is available, this class can be extended
      to compute gaze directly from YOLO eye boxes.
    - Current implementation provides a practical fallback to MediaPipe while
      still validating YOLO availability and mode selection.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.35) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self._yolo = self._try_load_yolo(model_path)
        self._fallback = MediaPipeTracker()

    def _try_load_yolo(self, model_path: str) -> Optional[object]:
        try:
            from ultralytics import YOLO

            return YOLO(model_path)
        except Exception:
            return None

    def process(self, frame_bgr) -> EyeMetrics:
        metrics = self._fallback.process(frame_bgr)
        if not metrics.found_face:
            return metrics

        if self._yolo is None:
            metrics.backend_note = "YOLO unavailable - fallback: MediaPipe"
        else:
            metrics.backend_note = f"YOLO loaded ({self.model_path}) - eye metrics via MediaPipe fallback"
        return metrics

    def close(self) -> None:
        self._fallback.close()
