from dataclasses import dataclass


@dataclass
class AppConfig:
    detector: str = "mediapipe"  # mediapipe | yolo
    eye_source: str = "both"  # left | right | both
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    blink_ear_threshold: float = 0.19
    blink_hold_seconds: float = 0.18
    click_cooldown_seconds: float = 0.8
    smoothing_alpha: float = 0.3
    gaze_gain_x: float = 2.0
    gaze_gain_y: float = 2.0
    direction_dead_zone: float = 0.04
    direction_affine_blend: float = 0.1
    direction_review_window: int = 90
    direction_mismatch_warn_rate: float = 0.28
    mouse_sensitivity: float = 0.1
    max_cursor_step: int = 70
    dead_zone_px: int = 8
    calibration_dwell_seconds: float = 1.0
    calibration_min_samples: int = 12
    grid_cols: int = 6
    grid_rows: int = 4
    # MediaPipe-only: expands compressed gaze ranges to 0..1.
    # Lower Y span => more vertical sensitivity (but potentially more jitter).
    mediapipe_adaptive_min_span_x: float = 0.1
    mediapipe_adaptive_min_span_y: float = 0.10
    mediapipe_gaze_alpha: float = 0.35
    yolo_model_path: str = "yolov8n-face.pt"
    yolo_conf_threshold: float = 0.35
