from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Point = Tuple[int, int]


@dataclass
class EyeMetrics:
    found_face: bool
    gaze_x: float = 0.5  # normalized 0..1
    gaze_y: float = 0.5  # normalized 0..1
    focus_x: Optional[int] = None
    focus_y: Optional[int] = None
    focus_confidence: float = 0.0
    focus_direction: str = "center"
    left_ear: float = 1.0
    right_ear: float = 1.0
    landmarks: Dict[str, List[Point]] = field(default_factory=dict)
    boxes: Dict[str, Tuple[int, int, int, int]] = field(default_factory=dict)
    backend_note: Optional[str] = None
