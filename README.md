# Gaze-Controlled Mouse Prototype

Real-time desktop prototype that tracks eye gaze via webcam to move the cursor and triggers mouse click on a deliberate blink hold. Built in Python with MediaPipe/YOLO detector modes, startup 9-point calibration, and live debug overlays.

---

## Scope

- **Webcam acquisition**: OpenCV-based camera stream with diagnostics.
- **Eye tracking**: MediaPipe Face Mesh by default; optional YOLO mode (currently falls back to MediaPipe metrics, ready for eye-class model integration).
- **Gaze-to-cursor control**: Eye gaze is converted to a direction vector; cursor follows that direction with adaptive smoothing.
- **Blink-to-click**: Detects deliberate sustained closure with cooldown and re-arm logic to prevent repeated clicks while eyes stay closed.
- **Debug UI**: Real-time overlay showing landmarks, EAR values, active grid cell, calibration status, and FPS.
- **Safety**: Pause hotkey and failsafe cursor limits.

---

## Features

- Switchable detector backends (`--detector mediapipe` or `--detector yolo`).
- Blink click with hold threshold + cooldown + reopen re-arming.
- Real-time grid/squares overlay highlighting current gaze cell.
- Startup 9-point calibration tutorial with auto-capture (dwell) for per-user accuracy.
- Live focus point estimate (screen XY), direction, and confidence in the HUD.
- Smooth cursor motion with configurable dead-zone and max-step limits.
- HUD with gaze coordinates, EAR values, blink state, and calibration hints.
- Multi-monitor (virtual desktop) cursor bounds on Windows.
- Emergency pause (`p`) and quit (`q`) hotkeys.

---

## Installation

```bash
# Clone or download the project
cd "c:\projects\study\ai\visual following"

# Install dependencies
pip install -r requirements.txt
```

Recommended: use a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- Webcam
- Windows (uses `pyautogui` for mouse control)

---

## Usage

Run the app with MediaPipe (default):

```bash
python main.py
```

Or explicitly select a detector:

```bash
python main.py --detector mediapipe
python main.py --detector yolo
```

You must complete calibration before cursor control starts.

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `p` | Pause/unpause cursor control |
| `r` | Restart 9-point calibration |

---

## How It Works

1. **Camera**: Opens the default webcam and streams frames.
2. **Detection**:
   - MediaPipe: Face Mesh landmarks → iris centers and eye aspect ratios (EAR).
   - YOLO: Loads YOLO model; currently uses MediaPipe metrics as fallback (ready for eye-class model integration).
3. **Gaze estimation**: Uses iris centers within eye corners/lids to produce high-resolution 0–1 gaze coordinates.
4. **Calibration**: Before control starts, a 9-point tutorial auto-captures gaze anchors by dwell-time.
5. **Direction extraction**: Calibrated gaze is converted into a signed movement vector with per-axis center-aware mapping, gain, dead-zone, and adaptive smoothing.
6. **Blink detection**: EAR threshold + hold triggers a click, then requires eye reopen before another click is allowed.
7. **Cursor control**: Cursor follows the direction vector every frame using dead-zone and max-step constraints.
8. **Debug overlay**: Grid squares, calibration target, landmarks, and HUD show internal state.

---

## Project Structure

```
.
├── main.py                     # Application entrypoint and main loop
├── config.py                   # Tunable parameters
├── requirements.txt            # Python dependencies
├── vision/
│   ├── camera.py               # Webcam acquisition
│   ├── types.py                # Shared data structures
│   ├── eye_tracker.py          # Detector factory
│   └── detectors/
│       ├── mediapipe_tracker.py # MediaPipe face/eye tracking
│       └── yolo_tracker.py     # YOLO mode (fallback to MediaPipe)
├── control/
│   ├── gaze_mapper.py          # Gaze → screen mapping + calibration
│   ├── mouse_controller.py     # Cursor movement and click
│   └── blink_click.py          # Blink-hold click logic
└── ui/
    └── debug_overlay.py        # Grid, landmarks, and HUD rendering
```

---

## Configuration

Edit `config.py` to tune:

- `detector`: `"mediapipe"` or `"yolo"`
- `camera_index`: Webcam device index
- `frame_width` / `frame_height`: Capture resolution
- `blink_ear_threshold`: EAR cutoff for blink detection
- `blink_hold_seconds`: Hold duration to trigger click
- `click_cooldown_seconds`: Minimum interval between clicks
- `smoothing_alpha`: EMA smoothing factor (0–1)
- `gaze_gain_x` / `gaze_gain_y`: Sensitivity gain after calibration
- `direction_dead_zone`: Minimum direction magnitude before movement
- `max_cursor_step`: Max cursor movement per frame
- `mouse_sensitivity`: Overall cursor speed scaling
- `dead_zone_px`: Dead-zone radius around current cursor
- `calibration_dwell_seconds`: Per-target hold duration for auto-capture
- `calibration_min_samples`: Minimum samples before a target is accepted
- `grid_cols` / `grid_rows`: Debug overlay grid size
- `mediapipe_gaze_alpha`: Temporal smoothing in MediaPipe gaze stream
- `mediapipe_adaptive_min_span_x` / `mediapipe_adaptive_min_span_y`: Adaptive normalization span floor (lower Y span = higher Y sensitivity)
- `yolo_model_path`: Path to YOLO model file
- `yolo_conf_threshold`: YOLO confidence threshold

### Practical tuning

- If vertical movement is too weak, increase `gaze_gain_y`.
- If downward/upward range feels compressed, lower `mediapipe_adaptive_min_span_y` slightly.
- If movement is jittery after increasing sensitivity, raise `mediapipe_adaptive_min_span_y` or lower `gaze_gain_y`.
- After changing mapping/sensitivity, restart and recalibrate (`r`).

---

## Troubleshooting

- **Cursor does not move down enough**
    - Re-run full 9-point calibration.
    - Ensure you hold steady on bottom-row targets (especially `bottom_center` and `bottom_right`).
    - Increase `gaze_gain_y` and/or reduce `mediapipe_adaptive_min_span_y`.

- **Blink click unreliable**
    - Improve lighting and keep face centered.
    - Tune `blink_ear_threshold` to your eye geometry.
    - Reduce `blink_hold_seconds` slightly if clicks require too long of a closure.
    - If accidental multi-clicks occur, increase `click_cooldown_seconds`.

---

## Extending YOLO Mode

The current `yolo_tracker` loads Ultralytics YOLO and falls back to MediaPipe metrics. To use a true YOLO eye model:

1. Provide a YOLO model trained on eye classes.
2. Update `YoloTracker.process` to extract eye boxes and compute gaze from box centers/corners.
3. Optionally replace EAR computation with eye-box height ratios.

---

## Safety Notes

- The app moves the cursor; keep your hand ready on the mouse or use the `p` pause key.
- `pyautogui.FAILSAFE` is disabled in code; use the `p` pause key or `q` quit key to stop movement.
- Use in a well-lit environment for stable tracking.

---

## License

Prototype project for educational/research use. Dependencies retain their original licenses.
