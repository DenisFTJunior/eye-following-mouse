# Gaze Direction Validation Scenarios

Use this checklist after calibration to validate eye-direction correctness and runtime robustness.

## Preconditions
- Stable lighting and centered face.
- Fresh 9-point calibration completed.
- HUD visible (`dir vec` and `dir review mismatch`).

## Must-pass Scenarios
1. **Horizontal correctness**
   - Action: Fixate center, then look left and right 3 times each.
   - Expectation: `dir_x` sign matches gaze side every time; cursor never mirrors.

2. **Vertical correctness**
   - Action: Fixate center, then look up and down 3 times each.
   - Expectation: `dir_y` sign matches gaze direction every time, including lower row targets.

3. **Diagonal consistency**
   - Action: Look at top-left, top-right, bottom-left, bottom-right.
   - Expectation: Both axis signs correspond to the quadrant (no axis sign flip).

4. **Center stability**
   - Action: Hold center fixation for 5-10 seconds.
   - Expectation: Low cursor drift and low direction oscillation; mismatch line stays low.

5. **Head-shift resilience**
   - Action: Repeat left/right/up/down while slightly translating and rotating the head.
   - Expectation: No sudden sign inversions during mild pose changes.

6. **Dropout recovery**
   - Action: Briefly occlude/reveal eyes or move out/in of frame.
   - Expectation: Direction recovers smoothly, without large jump spikes.

7. **Blink during motion**
   - Action: Hold off-center gaze and perform deliberate blink click.
   - Expectation: Blink click triggers as expected and direction state remains stable afterwards.

8. **Recalibration regression**
   - Action: Press `r`, recalibrate, and repeat scenarios 1-4.
   - Expectation: Direction correctness is equal or better than before recalibration.
