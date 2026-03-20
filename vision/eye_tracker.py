from __future__ import annotations

from config import AppConfig
from vision.detectors.mediapipe_tracker import MediaPipeTracker
from vision.detectors.yolo_tracker import YoloTracker


def create_tracker(cfg: AppConfig):
    mode = cfg.detector.strip().lower()
    if mode == "yolo":
        return YoloTracker(cfg.yolo_model_path, cfg.yolo_conf_threshold, eye_source=cfg.eye_source)
    return MediaPipeTracker(
        eye_source=cfg.eye_source,
        mirrored_input=True,
        gaze_alpha=cfg.mediapipe_gaze_alpha,
        adaptive_min_span_x=cfg.mediapipe_adaptive_min_span_x,
        adaptive_min_span_y=cfg.mediapipe_adaptive_min_span_y,
    )
