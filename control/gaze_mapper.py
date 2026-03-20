from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from logger_util import log_and_print


@dataclass
class CalibrationPoint:
    gaze: Tuple[float, float]
    screen: Tuple[int, int]


class GazeMapper:
    CALIBRATION_ORDER = [
        "top_left",
        "top_center",
        "top_right",
        "mid_left",
        "center",
        "mid_right",
        "bottom_left",
        "bottom_center",
        "bottom_right",
    ]

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        smoothing_alpha: float = 0.2,
        screen_x: int = 0,
        screen_y: int = 0,
        gain_x: float = 1.35,
        gain_y: float = 1.45,
        adaptive_smoothing: bool = True,
        direction_dead_zone: float = 0.04,
        direction_affine_blend: float = 0.65,
        direction_review_window: int = 90,
        direction_mismatch_warn_rate: float = 0.28,
    ) -> None:
        self.screen_x = int(screen_x)
        self.screen_y = int(screen_y)
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.alpha = float(np.clip(smoothing_alpha, 0.01, 0.95))
        self.gain_x = float(np.clip(gain_x, 0.5, 8.0))
        self.gain_y = float(np.clip(gain_y, 0.5, 8.0))
        self.adaptive_smoothing = adaptive_smoothing
        self._min_calib_span_x = 0.12
        self._min_calib_span_y = 0.10
        self._smoothed = np.array(
            [self.screen_x + screen_w // 2, self.screen_y + screen_h // 2],
            dtype=np.float32,
        )
        self._smoothed_direction = np.array([0.0, 0.0], dtype=np.float32)
        self._neutral_gaze = np.array([0.5, 0.5], dtype=np.float32)
        self._neutral_alpha = 0.02
        self._direction_dead_zone = float(np.clip(direction_dead_zone, 0.0, 0.35))
        self._direction_affine_blend = float(np.clip(direction_affine_blend, 0.0, 1.0))
        self._direction_mismatch_warn_rate = float(np.clip(direction_mismatch_warn_rate, 0.0, 1.0))
        self._calibration: Dict[str, CalibrationPoint] = {}
        self._affine_transform: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._affine_residual_norm = 1.0
        self._direction_review = deque(maxlen=max(16, int(direction_review_window)))

    def add_calibration(self, key: str, gaze_x: float, gaze_y: float, sx: int, sy: int) -> None:
        self._calibration[key] = CalibrationPoint(gaze=(gaze_x, gaze_y), screen=(sx, sy))
        self._update_affine_transform()

    @property
    def is_calibrated(self) -> bool:
        required = set(self.CALIBRATION_ORDER)
        return required.issubset(set(self._calibration.keys()))

    def reset_calibration(self) -> None:
        self._calibration.clear()
        self._affine_transform = None
        self._affine_residual_norm = 1.0
        self._direction_review.clear()
        self._neutral_gaze = np.array([0.5, 0.5], dtype=np.float32)

    def get_target_index(self) -> int:
        for idx, key in enumerate(self.CALIBRATION_ORDER):
            if key not in self._calibration:
                return idx
        return len(self.CALIBRATION_ORDER)

    def _map_axis(self, v: float, min_v: float, max_v: float, out_max: int) -> float:
        if max_v == min_v:
            return out_max / 2
        n = np.clip((v - min_v) / (max_v - min_v), 0.0, 1.0)
        return float(n * out_max)

    def _apply_gain(self, value_0_1: float, gain: float) -> float:
        centered = (float(value_0_1) - 0.5) * gain + 0.5
        return float(np.clip(centered, 0.0, 1.0))

    def _axis_bounds(self) -> tuple[float, float, float, float, bool, bool]:
        if not self.is_calibrated:
            return 0.0, 1.0, 0.0, 1.0, False, False

        left_points = ["top_left", "mid_left", "bottom_left"]
        right_points = ["top_right", "mid_right", "bottom_right"]
        top_points = ["top_left", "top_center", "top_right"]
        bottom_points = ["bottom_left", "bottom_center", "bottom_right"]

        left_vals = np.array([self._calibration[key].gaze[0] for key in left_points], dtype=np.float32)
        right_vals = np.array([self._calibration[key].gaze[0] for key in right_points], dtype=np.float32)
        top_vals = np.array([self._calibration[key].gaze[1] for key in top_points], dtype=np.float32)
        bottom_vals = np.array([self._calibration[key].gaze[1] for key in bottom_points], dtype=np.float32)

        left_med = float(np.median(left_vals))
        right_med = float(np.median(right_vals))
        top_med = float(np.median(top_vals))
        bottom_med = float(np.median(bottom_vals))

        invert_x = left_med > right_med
        invert_y = top_med > bottom_med

        min_x = min(left_med, right_med)
        max_x = max(left_med, right_med)
        min_y = min(top_med, bottom_med)
        max_y = max(top_med, bottom_med)

        cx = (min_x + max_x) * 0.5
        cy = (min_y + max_y) * 0.5
        span_x = max(max_x - min_x, self._min_calib_span_x)
        span_y = max(max_y - min_y, self._min_calib_span_y)

        min_x = cx - span_x * 0.5
        max_x = cx + span_x * 0.5
        min_y = cy - span_y * 0.5
        max_y = cy + span_y * 0.5

        min_x = float(np.clip(min_x, 0.0, 1.0))
        max_x = float(np.clip(max_x, 0.0, 1.0))
        min_y = float(np.clip(min_y, 0.0, 1.0))
        max_y = float(np.clip(max_y, 0.0, 1.0))
        return min_x, max_x, min_y, max_y, invert_x, invert_y

    def _update_affine_transform(self) -> None:
        # Fit a simple affine transform from normalized gaze (x, y, 1) -> screen px.
        # Needs at least 3 points to solve; more points are handled via least squares.
        if len(self._calibration) < 3:
            self._affine_transform = None
            return

        A = []
        bx = []
        by = []
        for point in self._calibration.values():
            gx, gy = point.gaze
            sx, sy = point.screen
            A.append([gx, gy, 1.0])
            bx.append(sx)
            by.append(sy)

        A_arr = np.asarray(A, dtype=np.float32)
        bx_arr = np.asarray(bx, dtype=np.float32)
        by_arr = np.asarray(by, dtype=np.float32)

        try:
            sol_x, *_ = np.linalg.lstsq(A_arr, bx_arr, rcond=None)
            sol_y, *_ = np.linalg.lstsq(A_arr, by_arr, rcond=None)
            self._affine_transform = (sol_x.astype(np.float32), sol_y.astype(np.float32))
            pred_x = A_arr @ sol_x
            pred_y = A_arr @ sol_y
            err_x = float(np.sqrt(np.mean((pred_x - bx_arr) ** 2)))
            err_y = float(np.sqrt(np.mean((pred_y - by_arr) ** 2)))
            diag = max(1.0, float(np.hypot(self.screen_w, self.screen_h)))
            self._affine_residual_norm = float(np.clip((err_x + err_y) / diag, 0.0, 1.0))
        except Exception:
            self._affine_transform = None
            self._affine_residual_norm = 1.0

    def _direction_from_affine(self, gaze_x: float, gaze_y: float) -> Optional[tuple[float, float]]:
        affine_xy = self._apply_affine(gaze_x, gaze_y)
        if affine_xy is None:
            return None
        raw_x, raw_y = affine_xy
        cx = self.screen_x + 0.5 * (self.screen_w - 1)
        cy = self.screen_y + 0.5 * (self.screen_h - 1)
        half_w = max(1.0, 0.5 * (self.screen_w - 1))
        half_h = max(1.0, 0.5 * (self.screen_h - 1))
        dir_x = float(np.clip((raw_x - cx) / half_w, -1.0, 1.0))
        dir_y = float(np.clip((raw_y - cy) / half_h, -1.0, 1.0))
        return dir_x, dir_y

    def _record_direction_review(self, axis_dir: tuple[float, float], final_dir: tuple[float, float]) -> None:
        ax, ay = axis_dir
        fx, fy = final_dir
        if abs(ax) > 0.08 and abs(fx) > 0.08:
            self._direction_review.append(int(np.sign(ax) != np.sign(fx)))
        if abs(ay) > 0.08 and abs(fy) > 0.08:
            self._direction_review.append(int(np.sign(ay) != np.sign(fy)))

    def _apply_affine(self, gaze_x: float, gaze_y: float) -> Optional[tuple[float, float]]:
        if self._affine_transform is None:
            return None

        ax, ay = self._affine_transform
        raw_x = float(ax[0] * gaze_x + ax[1] * gaze_y + ax[2])
        raw_y = float(ay[0] * gaze_x + ay[1] * gaze_y + ay[2])
        if not np.isfinite(raw_x) or not np.isfinite(raw_y):
            return None
        return raw_x, raw_y

    def map_to_screen(self, gaze_x: float, gaze_y: float) -> Tuple[int, int]:
        affine_xy = self._apply_affine(gaze_x, gaze_y)
        log_and_print(
            f"Affine transform applied: {affine_xy}" if affine_xy is not None else "No affine transform available"
        )
        log_and_print(
            f"calibration points: {self.is_calibrated}, {len(self._calibration)}, affine_transform={'set' if self._affine_transform else 'not set'}"
        )
        if affine_xy is not None:
            raw_x, raw_y = affine_xy
        elif self.is_calibrated:
            min_x, max_x, min_y, max_y, invert_x, invert_y = self._axis_bounds()

            center_gaze_x, center_gaze_y = self._calibration["center"].gaze
            center_expected_x = (min_x + max_x) * 0.5
            center_expected_y = (min_y + max_y) * 0.5

            bias_corrected_x = gaze_x - (center_gaze_x - center_expected_x)
            bias_corrected_y = gaze_y - (center_gaze_y - center_expected_y)

            norm_x = self._ratio_to_unit(bias_corrected_x, min_x, max_x)
            norm_y = self._ratio_to_unit(bias_corrected_y, min_y, max_y)

            if invert_x:
                norm_x = 1.0 - norm_x
            if invert_y:
                norm_y = 1.0 - norm_y

            boosted_x = self._apply_gain(norm_x, self.gain_x)
            boosted_y = self._apply_gain(norm_y, self.gain_y)
            
            log_and_print(
                f"bosted gaze: ({boosted_x}, {boosted_y}), norm gaze: ({norm_x}, {norm_y}), bias corrected gaze: ({bias_corrected_x}, {bias_corrected_y})"
            )

            raw_x = float(self.screen_x + boosted_x * (self.screen_w - 1))
            raw_y = float(self.screen_y + boosted_y * (self.screen_h - 1))
        else:
            log_and_print(f"GazeMapper: not calibrated, using direct mapping with gain {self.gain_x}, {self.gain_y} and gaze ({gaze_x}, {gaze_y})")
            raw_x = float(self.screen_x + np.clip(gaze_x, 0.0, 1.0) * (self.screen_w - 1))
            raw_y = float(self.screen_y + np.clip(gaze_y, 0.0, 1.0) * (self.screen_h - 1))
            log_and_print(f"Direct mapped gaze to screen: ({raw_x}, {raw_y})")
            
        target = np.array([raw_x, raw_y], dtype=np.float32)
        alpha = self.alpha
        if self.adaptive_smoothing:
            movement = float(np.linalg.norm(target - self._smoothed))
            fast_alpha = min(0.85, self.alpha + 0.45)
            slow_alpha = max(0.04, self.alpha * 0.55)
            blend = np.clip(movement / 140.0, 0.0, 1.0)
            alpha = float(slow_alpha + (fast_alpha - slow_alpha) * blend)

        self._smoothed = self._smoothed * (1.0 - alpha) + target * alpha

        x = int(np.clip(self._smoothed[0], self.screen_x, self.screen_x + self.screen_w - 1))
        y = int(np.clip(self._smoothed[1], self.screen_y, self.screen_y + self.screen_h - 1))
        return x, y

    def gaze_to_direction(self, gaze_x: float, gaze_y: float) -> Tuple[float, float]:
        axis_dir: tuple[float, float]
        if self.is_calibrated:
            min_x, max_x, min_y, max_y, invert_x, invert_y = self._axis_bounds()

            center_gaze_x, center_gaze_y = self._calibration["center"].gaze
            dir_x = self._centered_axis_direction(gaze_x, min_x, max_x, center_gaze_x, invert_x)
            dir_y = self._centered_axis_direction(gaze_y, min_y, max_y, center_gaze_y, invert_y)
            axis_dir = (dir_x, dir_y)

            affine_dir = self._direction_from_affine(gaze_x, gaze_y)
            if affine_dir is not None:
                residual_quality = float(np.clip(1.0 - (self._affine_residual_norm * 3.0), 0.0, 1.0))
                blend = self._direction_affine_blend * residual_quality
                dir_x = (1.0 - blend) * dir_x + blend * affine_dir[0]
                dir_y = (1.0 - blend) * dir_y + blend * affine_dir[1]
        else:
            affine_dir = self._direction_from_affine(gaze_x, gaze_y)
            if affine_dir is not None:
                dir_x, dir_y = affine_dir
            else:
                norm_x = float(np.clip(gaze_x, 0.0, 1.0))
                norm_y = float(np.clip(gaze_y, 0.0, 1.0))

                # Learn neutral gaze before calibration so center bias does not force wrong direction.
                sample = np.array([norm_x, norm_y], dtype=np.float32)
                delta = np.abs(sample - self._neutral_gaze)
                if float(delta[0]) <= 0.18 and float(delta[1]) <= 0.18:
                    self._neutral_gaze = self._neutral_gaze * (1.0 - self._neutral_alpha) + sample * self._neutral_alpha

                dir_x = (norm_x - float(self._neutral_gaze[0])) * 2.0
                dir_y = (norm_y - float(self._neutral_gaze[1])) * 2.0
            axis_dir = (dir_x, dir_y)

        dir_x = float(np.clip(dir_x * self.gain_x, -1.0, 1.0))
        dir_y = float(np.clip(dir_y * self.gain_y, -1.0, 1.0))

        vec = np.array([dir_x, dir_y], dtype=np.float32)
        mag = float(np.linalg.norm(vec))

        if mag <= self._direction_dead_zone:
            vec = np.array([0.0, 0.0], dtype=np.float32)
        elif mag > 1e-6:
            scaled_mag = (mag - self._direction_dead_zone) / (1.0 - self._direction_dead_zone)
            vec = vec / mag * float(np.clip(scaled_mag, 0.0, 1.0))

        alpha = self.alpha
        if self.adaptive_smoothing:
            delta = float(np.linalg.norm(vec - self._smoothed_direction))
            fast_alpha = min(0.9, self.alpha + 0.5)
            slow_alpha = max(0.06, self.alpha * 0.6)
            blend = float(np.clip(delta / 0.35, 0.0, 1.0))
            alpha = float(slow_alpha + (fast_alpha - slow_alpha) * blend)

        self._smoothed_direction = self._smoothed_direction * (1.0 - alpha) + vec * alpha
        out_x = float(self._smoothed_direction[0])
        out_y = float(self._smoothed_direction[1])
        self._record_direction_review(axis_dir, (out_x, out_y))
        return out_x, out_y

    def get_direction_review(self) -> tuple[float, bool]:
        if not self._direction_review:
            return 0.0, False
        mismatch_rate = float(np.mean(np.asarray(self._direction_review, dtype=np.float32)))
        return mismatch_rate, mismatch_rate >= self._direction_mismatch_warn_rate

    def _centered_axis_direction(
        self,
        value: float,
        min_v: float,
        max_v: float,
        center_v: float,
        invert: bool,
    ) -> float:
        value_u = self._ratio_to_unit(value, min_v, max_v)
        center_u = self._ratio_to_unit(center_v, min_v, max_v)

        if invert:
            value_u = 1.0 - value_u
            center_u = 1.0 - center_u

        if value_u >= center_u:
            denom = max(1e-6, 1.0 - center_u)
            out = (value_u - center_u) / denom
        else:
            denom = max(1e-6, center_u)
            out = (value_u - center_u) / denom

        return float(np.clip(out, -1.0, 1.0))

    def get_calibration_hint(self) -> str:
        missing = [k for k in self.CALIBRATION_ORDER if k not in self._calibration]
        if not missing:
            return "Calibration complete"
        return f"Capture: {missing[0]}"

    def _ratio_to_unit(self, value: float, min_v: float, max_v: float) -> float:
        if max_v == min_v:
            return 0.5
        return float(np.clip((value - min_v) / (max_v - min_v), 0.0, 1.0))
