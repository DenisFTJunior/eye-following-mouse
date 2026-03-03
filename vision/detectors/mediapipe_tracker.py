from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from vision.types import EyeMetrics, Point


class MediaPipeTracker:
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

    LEFT_EYE_CORNERS = (33, 133)
    RIGHT_EYE_CORNERS = (362, 263)
    LEFT_EYE_LIDS = (159, 145)
    RIGHT_EYE_LIDS = (386, 374)

    def __init__(
        self,
        *,
        gaze_alpha: float = 0.35,
        adaptive_min_span_x: float = 0.18,
        adaptive_min_span_y: float = 0.16,
    ) -> None:
        self._landmarker: Optional[object] = None
        self._face_mesh = None

        self._frame_timestamp_ms = 0
        self._backend_note: Optional[str] = None

        model_path = self._find_task_model_path()
        if model_path is not None:
            try:
                options = vision.FaceLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=str(model_path)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=1,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                )
                self._landmarker = vision.FaceLandmarker.create_from_options(options)
                self._backend_note = f"MediaPipe Tasks ({model_path.name})"
            except Exception:
                self._landmarker = None

        if self._landmarker is None:
            face_mesh_cls = None
            try:
                face_mesh_cls = mp.solutions.face_mesh.FaceMesh
            except AttributeError:
                try:
                    from mediapipe.python.solutions.face_mesh import FaceMesh

                    face_mesh_cls = FaceMesh
                except Exception:
                    face_mesh_cls = None

            if face_mesh_cls is not None:
                self._face_mesh = face_mesh_cls(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            if self._backend_note is None:
                self._backend_note = (
                    "MediaPipe FaceMesh fallback"
                    if self._face_mesh is not None
                    else "No face backend: provide face_landmarker.task"
                )

        # Simple temporal smoothing for gaze to reduce jitter before calibration.
        self._gaze_smooth = np.array([0.5, 0.5], dtype=np.float32)
        self._gaze_alpha = float(np.clip(gaze_alpha, 0.02, 0.95))
        # Adaptive normalization bounds to expand limited gaze ranges to near 0..1.
        self._adaptive_min = np.array([0.35, 0.35], dtype=np.float32)
        self._adaptive_max = np.array([0.65, 0.65], dtype=np.float32)
        self._adaptive_expand_alpha = 0.22
        self._adaptive_shrink_alpha = 0.006
        self._adaptive_min_span = np.array(
            [
                float(np.clip(adaptive_min_span_x, 0.04, 0.50)),
                float(np.clip(adaptive_min_span_y, 0.04, 0.50)),
            ],
            dtype=np.float32,
        )

    def _find_task_model_path(self) -> Optional[Path]:
        env_path = os.getenv("MEDIAPIPE_FACE_LANDMARKER_PATH")
        candidates = []
        if env_path:
            candidates.append(Path(env_path))

        repo_root = Path(__file__).resolve().parents[2]
        candidates.extend(
            [
                repo_root / "face_landmarker.task",
                repo_root / "models" / "face_landmarker.task",
                repo_root / "assets" / "face_landmarker.task",
                Path.cwd() / "face_landmarker.task",
            ]
        )

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def _next_timestamp_ms(self) -> int:
        now_ms = int(time.monotonic() * 1000)
        self._frame_timestamp_ms = max(self._frame_timestamp_ms + 1, now_ms)
        return self._frame_timestamp_ms

    def _to_px(self, lm, width: int, height: int) -> Point:
        # convert landmark normalized coordinates [0,1] to pixel coordinates
        return int(lm.x * width), int(lm.y * height)

    def _to_px_float(self, lm, width: int, height: int) -> Tuple[float, float]:
        return float(lm.x * width), float(lm.y * height)

    def _dist(self, p1: Point, p2: Point) -> float:
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def _ear(self, points: List[Point]) -> float:
        # Eye Aspect Ratio (EAR) calculation based on the 6 eye landmarks.
        # can say when blinking by checking if EAR is below a threshold.
        p1, p2, p3, p4, p5, p6 = points
        denom = 2.0 * self._dist(p1, p4)
        if denom == 0:
            return 1.0
        return (self._dist(p2, p6) + self._dist(p3, p5)) / denom

    def _ratio(self, value: float, min_v: float, max_v: float) -> float:
        if max_v == min_v:
            return 0.5
        out = (value - min_v) / (max_v - min_v)
        return float(np.clip(out, 0.0, 1.0))

    def _focus_direction(self, gaze_x: float, gaze_y: float, deadband: float = 0.08) -> str:
        horizontal = "center"
        vertical = "center"

        if gaze_x < 0.5 - deadband:
            horizontal = "left"
        elif gaze_x > 0.5 + deadband:
            horizontal = "right"

        if gaze_y < 0.5 - deadband:
            vertical = "up"
        elif gaze_y > 0.5 + deadband:
            vertical = "down"

        if horizontal == "center" and vertical == "center":
            return "center"
        if horizontal == "center":
            return vertical
        if vertical == "center":
            return horizontal
        return f"{vertical}-{horizontal}"

    def _adaptive_normalize(self, gaze_x: float, gaze_y: float) -> tuple[float, float]:
        value = np.array([gaze_x, gaze_y], dtype=np.float32)

        below = value < self._adaptive_min
        above = value > self._adaptive_max

        self._adaptive_min = np.where(
            below,
            self._adaptive_min + self._adaptive_expand_alpha * (value - self._adaptive_min),
            self._adaptive_min + self._adaptive_shrink_alpha * (value - self._adaptive_min),
        )
        self._adaptive_max = np.where(
            above,
            self._adaptive_max + self._adaptive_expand_alpha * (value - self._adaptive_max),
            self._adaptive_max + self._adaptive_shrink_alpha * (value - self._adaptive_max),
        )

        center = 0.5 * (self._adaptive_min + self._adaptive_max)
        span = np.maximum(self._adaptive_max - self._adaptive_min, self._adaptive_min_span)
        self._adaptive_min = center - 0.5 * span
        self._adaptive_max = center + 0.5 * span

        norm = (value - self._adaptive_min) / np.maximum(self._adaptive_max - self._adaptive_min, 1e-6)
        norm = np.clip(norm, 0.0, 1.0)
        return float(norm[0]), float(norm[1])

    def process(self, frame_bgr) -> EyeMetrics:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb)

        lms = None
        if self._landmarker is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._landmarker.detect_for_video(mp_image, self._next_timestamp_ms())
            if result.face_landmarks:
                lms = result.face_landmarks[0]
        elif self._face_mesh is not None:
            result = self._face_mesh.process(frame_rgb)
            if result.multi_face_landmarks:
                lms = result.multi_face_landmarks[0].landmark

        if lms is None:
            return EyeMetrics(found_face=False, backend_note=self._backend_note)

        to_px = lambda idx: self._to_px(lms[idx], w, h)
        to_px_float = lambda idx: self._to_px_float(lms[idx], w, h)

        li_float = [to_px_float(i) for i in self.LEFT_IRIS]
        ri_float = [to_px_float(i) for i in self.RIGHT_IRIS]
        li = [to_px(i) for i in self.LEFT_IRIS]
        ri = [to_px(i) for i in self.RIGHT_IRIS]

        li_center_float = tuple(np.mean(np.array(li_float, dtype=np.float32), axis=0).tolist())
        ri_center_float = tuple(np.mean(np.array(ri_float, dtype=np.float32), axis=0).tolist())
        li_center = (int(li_center_float[0]), int(li_center_float[1]))
        ri_center = (int(ri_center_float[0]), int(ri_center_float[1]))

        l_outer_f, l_inner_f = to_px_float(self.LEFT_EYE_CORNERS[0]), to_px_float(self.LEFT_EYE_CORNERS[1])
        r_inner_f, r_outer_f = to_px_float(self.RIGHT_EYE_CORNERS[0]), to_px_float(self.RIGHT_EYE_CORNERS[1])

        l_upper_f, l_lower_f = to_px_float(self.LEFT_EYE_LIDS[0]), to_px_float(self.LEFT_EYE_LIDS[1])
        r_upper_f, r_lower_f = to_px_float(self.RIGHT_EYE_LIDS[0]), to_px_float(self.RIGHT_EYE_LIDS[1])

        # Use a cross-eye span to improve stability and avoid per-eye stuck-at-center cases.
        # Expand spans slightly to avoid overly narrow ranges that clamp movement.
        face_left = min(l_outer_f[0], r_outer_f[0])
        face_right = max(l_outer_f[0], r_outer_f[0])
        face_top = min(l_upper_f[1], r_upper_f[1])
        face_bottom = max(l_lower_f[1], r_lower_f[1])
        margin_x = 0.08 * max(6.0, face_right - face_left)
        margin_y = 0.08 * max(6.0, face_bottom - face_top)
        face_left -= margin_x
        face_right += margin_x
        face_top -= margin_y
        face_bottom += margin_y

        # Ensure minimum span to avoid division squeeze.
        if face_right - face_left < 8.0:
            cx = 0.5 * (face_left + face_right)
            face_left, face_right = cx - 4.0, cx + 4.0
        if face_bottom - face_top < 8.0:
            cy = 0.5 * (face_top + face_bottom)
            face_top, face_bottom = cy - 4.0, cy + 4.0

        both_iris_x = 0.5 * (li_center_float[0] + ri_center_float[0])
        both_iris_y = 0.5 * (li_center_float[1] + ri_center_float[1])

        gaze_x = self._ratio(both_iris_x, face_left, face_right)
        gaze_y = self._ratio(both_iris_y, face_top, face_bottom)

        # Fallback to per-eye ratios if the span collapses (extreme cases only).
        if not np.isfinite(gaze_x) or face_right - face_left < 2.0:
            left_x = self._ratio(li_center_float[0], min(l_outer_f[0], l_inner_f[0]), max(l_outer_f[0], l_inner_f[0]))
            right_x = self._ratio(ri_center_float[0], min(r_outer_f[0], r_inner_f[0]), max(r_outer_f[0], r_inner_f[0]))
            gaze_x = (left_x + right_x) / 2.0

        if not np.isfinite(gaze_y) or face_bottom - face_top < 2.0:
            left_y = self._ratio(li_center_float[1], min(l_upper_f[1], l_lower_f[1]), max(l_upper_f[1], l_lower_f[1]))
            right_y = self._ratio(ri_center_float[1], min(r_upper_f[1], r_lower_f[1]), max(r_upper_f[1], r_lower_f[1]))
            gaze_y = (left_y + right_y) / 2.0

        # Expand dynamic range to use full 0..1 even when raw values are compressed.
        gaze_x, gaze_y = self._adaptive_normalize(gaze_x, gaze_y)

        # Temporal smoothing to stabilize cursor before mapper smoothing.
        g_now = np.array([gaze_x, gaze_y], dtype=np.float32)
        self._gaze_smooth = self._gaze_smooth * (1.0 - self._gaze_alpha) + g_now * self._gaze_alpha
        gaze_x, gaze_y = float(self._gaze_smooth[0]), float(self._gaze_smooth[1])

        left_ear = self._ear([to_px(i) for i in self.LEFT_EYE_EAR])
        right_ear = self._ear([to_px(i) for i in self.RIGHT_EYE_EAR])

        ear_balance = 1.0 - np.clip(abs(left_ear - right_ear) / 0.15, 0.0, 1.0)
        openness = np.clip((left_ear + right_ear) / 0.55, 0.0, 1.0)
        focus_confidence = float(np.clip(ear_balance * openness, 0.0, 1.0))
        focus_direction = self._focus_direction(gaze_x, gaze_y)

        landmarks: Dict[str, List[Point]] = {
            "left_iris": li,
            "right_iris": ri,
            "left_eye": [to_px(i) for i in self.LEFT_EYE_EAR],
            "right_eye": [to_px(i) for i in self.RIGHT_EYE_EAR],
            "pupil_centers": [li_center, ri_center],
        }

        return EyeMetrics(
            found_face=True,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            focus_confidence=focus_confidence,
            focus_direction=focus_direction,
            left_ear=left_ear,
            right_ear=right_ear,
            landmarks=landmarks,
            backend_note=self._backend_note,
        )

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
        if self._face_mesh is not None:
            self._face_mesh.close()
   
