from __future__ import annotations

import time


class BlinkClickDetector:
    def __init__(self, threshold: float, hold_seconds: float, cooldown_seconds: float) -> None:
        self.threshold = threshold
        self.open_threshold = threshold + 0.015
        self.hold_seconds = hold_seconds
        self.cooldown_seconds = cooldown_seconds

        self._closed_since = None
        self._last_click = 0.0
        self._armed = True

    def update(self, left_ear: float, right_ear: float) -> bool:
        now = time.time()
        avg_ear = (left_ear + right_ear) * 0.5
        closed = (left_ear < self.threshold and right_ear < self.threshold) or avg_ear < (self.threshold - 0.01)

        if closed:
            if self._closed_since is None:
                self._closed_since = now

            held = now - self._closed_since
            if (
                self._armed
                and held >= self.hold_seconds
                and (now - self._last_click) >= self.cooldown_seconds
            ):
                self._last_click = now
                self._armed = False
                return True
        else:
            self._closed_since = None
            if avg_ear > self.open_threshold:
                self._armed = True

        return False

    @property
    def is_closed(self) -> bool:
        return self._closed_since is not None
