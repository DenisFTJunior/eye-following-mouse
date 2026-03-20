from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import pyautogui

from config import AppConfig
from control.blink_click import BlinkClickDetector
from control.gaze_mapper import GazeMapper
from control.mouse_controller import MouseController
from logger_util import log_and_print
from ui.debug_overlay import draw_calibration_target, draw_focus_point, draw_gaze_vector, draw_grid, draw_landmarks, put_hud
from vision.camera import CameraStream
from vision.eye_tracker import create_tracker


def get_virtual_desktop_bounds() -> tuple[int, int, int, int]:
    try:
        import ctypes

        user32 = ctypes.windll.user32
        x = int(user32.GetSystemMetrics(76))
        y = int(user32.GetSystemMetrics(77))
        w = int(user32.GetSystemMetrics(78))
        h = int(user32.GetSystemMetrics(79))
        if w > 0 and h > 0:
            return x, y, w, h
    except Exception:
        pass

    w, h = pyautogui.size()
    return 0, 0, int(w), int(h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eye-gaze mouse control prototype")
    parser.add_argument("--detector", choices=["mediapipe", "yolo"], default="mediapipe")
    parser.add_argument("--camera-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AppConfig(detector=args.detector, camera_index=args.camera_index)

    screen_x, screen_y, screen_w, screen_h = get_virtual_desktop_bounds()

    camera = CameraStream(cfg.camera_index, cfg.frame_width, cfg.frame_height)
    tracker = create_tracker(cfg)
    log_and_print(f"Virtual desktop bounds: x={screen_x}, y={screen_y}, w={screen_w}, h={screen_h}")
    mapper = GazeMapper(
        screen_w,
        screen_h,
        cfg.smoothing_alpha,
        screen_x=screen_x,
        screen_y=screen_y,
        gain_x=cfg.gaze_gain_x,
        gain_y=cfg.gaze_gain_y,
        adaptive_smoothing=True,
        direction_dead_zone=cfg.direction_dead_zone,
    )
    mouse = MouseController(cfg.max_cursor_step, cfg.dead_zone_px, sensitivity=cfg.mouse_sensitivity)
    blink = BlinkClickDetector(cfg.blink_ear_threshold, cfg.blink_hold_seconds, cfg.click_cooldown_seconds)

    paused = False
    last_t = time.time()
    blink_event_until = 0.0
    last_gaze_log_t = 0.0

    calibration_targets = [
        ("top_left", "top-left", 0.10, 0.10),
        ("top_center", "top-center", 0.50, 0.10),
        ("top_right", "top-right", 0.90, 0.10),
        ("mid_left", "left", 0.10, 0.50),
        ("center", "center", 0.50, 0.50),
        ("mid_right", "right", 0.90, 0.50),
        ("bottom_left", "bottom-left", 0.10, 0.90),
        ("bottom_center", "bottom-center", 0.50, 0.90),
        ("bottom_right", "bottom-right", 0.90, 0.90),
    ]
    calibration_index = 0
    dwell_start_t: float | None = None
    dwell_samples: list[tuple[float, float]] = []

    log_and_print("Controls:")
    log_and_print("  q: quit")
    log_and_print("  p: pause/unpause cursor control")
    log_and_print("  r: restart 9-point calibration")
    log_and_print("Startup calibration: look at each target until it auto-captures")

    try:
        while True:
            frame = camera.read()
            frame = cv2.flip(frame, 1)

            metrics = tracker.process(frame)
            cursor_xy = pyautogui.position()
            focus_xy = None
            gaze_vector = (0.0, 0.0)

            calibration_complete = mapper.is_calibrated and calibration_index >= len(calibration_targets)

            if metrics.found_face:
                dir_x, dir_y = mapper.gaze_to_direction(metrics.gaze_x, metrics.gaze_y)
                gaze_vector = (dir_x, dir_y)
                now_log_t = time.time()
                if now_log_t - last_gaze_log_t >= 0.25:
                    log_and_print(f"Gaze direction vector: ({dir_x:.3f}, {dir_y:.3f})")
                    last_gaze_log_t = now_log_t

                if calibration_complete and not paused:
                    mouse.move_by_direction(dir_x, dir_y)

                cursor_xy = pyautogui.position()
                focus_xy = cursor_xy
                metrics.focus_x = cursor_xy[0]
                metrics.focus_y = cursor_xy[1]

                if calibration_complete and blink.update(metrics.left_ear, metrics.right_ear) and not paused:
                    mouse.click()
                    blink_event_until = time.time() + 0.8

                draw_landmarks(frame, metrics)

                if not calibration_complete and calibration_index < len(calibration_targets):
                    key, label, tx_norm, ty_norm = calibration_targets[calibration_index]
                    now_t = time.time()

                    if dwell_start_t is None:
                        dwell_start_t = now_t
                        dwell_samples.clear()

                    dwell_samples.append((metrics.gaze_x, metrics.gaze_y))
                    elapsed = now_t - dwell_start_t
                    seconds_left = cfg.calibration_dwell_seconds - elapsed

                    if elapsed >= cfg.calibration_dwell_seconds and len(dwell_samples) >= cfg.calibration_min_samples:
                        sample_arr = np.array(dwell_samples, dtype=np.float32)
                        std_x = float(np.std(sample_arr[:, 0]))
                        std_y = float(np.std(sample_arr[:, 1]))
                        stable_capture = std_x <= 0.035 and std_y <= 0.040

                        if not stable_capture:
                            dwell_start_t = now_t
                            dwell_samples.clear()
                            log_and_print(
                                f"Calibration point {key} unstable (std x={std_x:.3f}, y={std_y:.3f}) - hold steadier"
                            )
                            continue

                        med_gaze = np.median(sample_arr, axis=0)
                        avg_gaze = (
                            float(med_gaze[0]),
                            float(med_gaze[1]),
                        )
                        sx = screen_x + int(tx_norm * (screen_w - 1))
                        sy = screen_y + int(ty_norm * (screen_h - 1))
                        mapper.add_calibration(key, avg_gaze[0], avg_gaze[1], sx, sy)
                        calibration_index += 1
                        dwell_start_t = None
                        dwell_samples.clear()
                        log_and_print(f"Captured calibration {calibration_index}/{len(calibration_targets)}: {key}")

                    progress = f"{calibration_index}/{len(calibration_targets)}"
                    draw_calibration_target(frame, (tx_norm, ty_norm), label, progress, seconds_left)
            else:
                dwell_start_t = None
                dwell_samples.clear()

            active_col = int(max(0.0, min(0.999, metrics.gaze_x)) * cfg.grid_cols)
            active_row = int(max(0.0, min(0.999, metrics.gaze_y)) * cfg.grid_rows)
            draw_grid(frame, cfg.grid_cols, cfg.grid_rows, (active_col, active_row))

            now = time.time()
            fps = 1.0 / max(1e-6, now - last_t)
            last_t = now
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, frame.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 230, 230),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Cursor XY: ({cursor_xy[0]}, {cursor_xy[1]})",
                (10, frame.shape[0] - 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (80, 255, 80),
                2,
                cv2.LINE_AA,
            )

            if time.time() < blink_event_until:
                cv2.putText(
                    frame,
                    "BLINK CLICK!",
                    (frame.shape[1] - 220, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            draw_focus_point(frame, focus_xy)
            draw_gaze_vector(frame, gaze_vector)

            if not calibration_complete:
                note = "Calibration in progress"
            else:
                note = metrics.backend_note or ("PAUSED" if paused else None)

            put_hud(
                frame,
                cfg.detector,
                (metrics.gaze_x, metrics.gaze_y),
                cursor_xy,
                focus_xy,
                gaze_vector,
                metrics.focus_direction,
                metrics.focus_confidence,
                metrics.left_ear,
                metrics.right_ear,
                blink.is_closed,
                mapper.get_calibration_hint(),
                note,
            )

            cv2.imshow("Gaze Mouse Debug", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("p"):
                paused = not paused
                continue
            if key == ord("r"):
                mapper.reset_calibration()
                calibration_index = 0
                dwell_start_t = None
                dwell_samples.clear()
                log_and_print("Calibration reset")

    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
