from __future__ import annotations

from typing import Tuple

import cv2


def draw_grid(frame, cols: int, rows: int, active_cell: Tuple[int, int] | None = None):
    h, w = frame.shape[:2]
    cols = max(1, cols)
    rows = max(1, rows)
    cell_w = w // cols
    cell_h = h // rows

    for c in range(1, cols):
        x = c * cell_w
        cv2.line(frame, (x, 0), (x, h), (80, 80, 80), 1)
    for r in range(1, rows):
        y = r * cell_h
        cv2.line(frame, (0, y), (w, y), (80, 80, 80), 1)

    if active_cell is not None:
        c, r = active_cell
        c = max(0, min(cols - 1, c))
        r = max(0, min(rows - 1, r))
        x1, y1 = c * cell_w, r * cell_h
        x2, y2 = x1 + cell_w, y1 + cell_h
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)


def draw_landmarks(frame, metrics):
    for p in metrics.landmarks.get("left_iris", []):
        cv2.circle(frame, p, 2, (255, 255, 0), -1)
    for p in metrics.landmarks.get("right_iris", []):
        cv2.circle(frame, p, 2, (0, 255, 255), -1)
    for p in metrics.landmarks.get("pupil_centers", []):
        cv2.circle(frame, p, 4, (0, 255, 0), -1)


def draw_focus_point(frame, focus_xy: Tuple[int, int] | None):
    if focus_xy is None:
        return
    x, y = int(focus_xy[0]), int(focus_xy[1])
    h, w = frame.shape[:2]
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    cv2.circle(frame, (x, y), 10, (255, 180, 0), 2)
    cv2.circle(frame, (x, y), 2, (255, 220, 0), -1)


def draw_gaze_vector(frame, gaze_vector: Tuple[float, float], length_px: int = 110):
    vx = float(max(-1.0, min(1.0, gaze_vector[0])))
    vy = float(max(-1.0, min(1.0, gaze_vector[1])))
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Screen-space vector: +x is right, +y is down.
    tip_x = int(cx + vx * length_px)
    tip_y = int(cy + vy * length_px)
    cv2.arrowedLine(frame, (cx, cy), (tip_x, tip_y), (0, 220, 255), 3, cv2.LINE_AA, tipLength=0.2)
    cv2.circle(frame, (cx, cy), 5, (0, 220, 255), -1)
    cv2.putText(
        frame,
        f"gaze vec: ({vx:+.2f}, {vy:+.2f})",
        (cx - 120, max(22, cy - 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )


def draw_calibration_target(
    frame,
    target_norm_xy: Tuple[float, float] | None,
    label: str,
    progress_text: str,
    seconds_left: float,
):
    h, w = frame.shape[:2]
    panel_h = 82
    cv2.rectangle(frame, (0, 0), (w, panel_h), (20, 20, 20), -1)
    cv2.putText(frame, "Calibration tutorial", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (80, 230, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Look at {label} and hold gaze", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{progress_text} | capture in {max(0.0, seconds_left):.1f}s", (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 255, 120), 2, cv2.LINE_AA)

    if target_norm_xy is None:
        return

    tx = int(max(0.05, min(0.95, target_norm_xy[0])) * (w - 1))
    ty = int(max(0.15, min(0.95, target_norm_xy[1])) * (h - 1))
    cv2.circle(frame, (tx, ty), 18, (0, 215, 255), 2)
    cv2.circle(frame, (tx, ty), 4, (0, 215, 255), -1)


def put_hud(
    frame,
    detector_mode: str,
    gaze_xy: Tuple[float, float],
    cursor_xy: Tuple[int, int],
    focus_xy: Tuple[int, int] | None,
    gaze_vector: Tuple[float, float],
    focus_direction: str,
    focus_confidence: float,
    left_ear: float,
    right_ear: float,
    blink_closed: bool,
    calibration_hint: str,
    note: str | None,
):
    lines = [
        f"mode: {detector_mode}",
        f"gaze norm: ({gaze_xy[0]:.3f}, {gaze_xy[1]:.3f})",
        f"gaze dir vec(center): ({gaze_vector[0]:+.3f}, {gaze_vector[1]:+.3f})",
        f"cursor: ({cursor_xy[0]}, {cursor_xy[1]})",
        f"focus: {focus_xy if focus_xy is not None else 'n/a'} | dir: {focus_direction} | conf: {focus_confidence:.2f}",
        f"EAR L/R: {left_ear:.3f} / {right_ear:.3f}",
        f"blink state: {'closed' if blink_closed else 'open'}",
        f"calibration: {calibration_hint}",
    ]
    if note:
        lines.append(note)

    y = 24
    for text in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (40, 240, 40), 2, cv2.LINE_AA)
        y += 22
