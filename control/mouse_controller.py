from __future__ import annotations

import math

import pyautogui


class MouseController:
    def __init__(self, max_step: int = 70, dead_zone_px: int = 8, sensitivity: float = 1.0) -> None:
        pyautogui.FAILSAFE = False
        self.max_step = max(1, int(max_step))
        self.dead_zone_px = max(0, int(dead_zone_px))
        self.sensitivity = max(0.05, float(sensitivity))
        self._carry_dx = 0.0
        self._carry_dy = 0.0

    def move_to(self, x: int, y: int) -> None:
        cx, cy = pyautogui.position()
        dx = x - cx
        dy = y - cy
        dist = math.hypot(dx, dy)

        if dist <= self.dead_zone_px:
            return

        if dist > self.max_step:
            scale = self.max_step / dist
            x = int(cx + dx * scale)
            y = int(cy + dy * scale)

        pyautogui.moveTo(x, y, _pause=False)

    def move_by_direction(self, dir_x: float, dir_y: float) -> None:
        magnitude = math.hypot(dir_x, dir_y)
        if magnitude <= 1e-6:
            return

        magnitude = min(1.0, magnitude)
        speed_scale = magnitude ** 1.25
        min_step = 1.2
        step = (min_step + (self.max_step - min_step) * speed_scale) * self.sensitivity

        unit_x = dir_x / magnitude
        unit_y = dir_y / magnitude

        move_x = unit_x * step + self._carry_dx
        move_y = unit_y * step + self._carry_dy

        px_move_x = int(round(move_x))
        px_move_y = int(round(move_y))

        self._carry_dx = move_x - px_move_x
        self._carry_dy = move_y - px_move_y

        if px_move_x == 0 and px_move_y == 0:
            return

        cx, cy = pyautogui.position()
        nx = cx + px_move_x
        ny = cy + px_move_y
        pyautogui.moveTo(nx, ny, _pause=False)

    def click(self) -> None:
        pyautogui.click()
