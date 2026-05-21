"""
Interactive gaze calibration.

Guides the user through 5 fixation points (left, right, up, down, centre)
and records average gaze readings to compute per-axis offsets and thresholds.
"""

import time
import json
import os
import numpy as np

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "calibration_data.json")

# Duration to collect samples per fixation point (seconds)
SAMPLE_DURATION = 2.5

CALIBRATION_POINTS = [
    ("CENTRE",     (0.5, 0.5)),
    ("LEFT",       (0.1, 0.5)),
    ("RIGHT",      (0.9, 0.5)),
    ("UP",         (0.5, 0.1)),
    ("DOWN",       (0.5, 0.9)),
]


class CalibrationData:
    def __init__(self):
        self.gaze_x_offset: float = 0.0
        self.gaze_y_offset: float = 0.0
        self.gaze_x_scale: float = 1.0
        self.gaze_y_scale: float = 1.0

    def save(self):
        data = {
            "gaze_x_offset": self.gaze_x_offset,
            "gaze_y_offset": self.gaze_y_offset,
            "gaze_x_scale":  self.gaze_x_scale,
            "gaze_y_scale":  self.gaze_y_scale,
        }
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls) -> "CalibrationData":
        cd = cls()
        if os.path.exists(CALIBRATION_FILE):
            with open(CALIBRATION_FILE) as f:
                data = json.load(f)
            cd.gaze_x_offset = data.get("gaze_x_offset", 0.0)
            cd.gaze_y_offset = data.get("gaze_y_offset", 0.0)
            cd.gaze_x_scale  = data.get("gaze_x_scale",  1.0)
            cd.gaze_y_scale  = data.get("gaze_y_scale",  1.0)
        return cd


def run_calibration(gaze_estimator, cap) -> CalibrationData:
    """
    Interactive calibration routine.  Runs inside an OpenCV window.
    Returns CalibrationData with computed offsets/scales.
    """
    if not CV2_OK:
        raise ImportError("opencv-python is required for calibration.")

    readings = {}

    for label, (nx, ny) in CALIBRATION_POINTS:
        print(f"[Calibration] Look at: {label}")
        samples_x, samples_y = [], []
        deadline = time.time() + SAMPLE_DURATION
        countdown_shown = False

        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)  # mirror so left/right feels natural
            h, w = frame.shape[:2]

            # Draw fixation target
            cx, cy = int(nx * w), int(ny * h)
            cv2.circle(frame, (cx, cy), 20, (0, 255, 255), 3)
            cv2.circle(frame, (cx, cy), 5,  (0, 255, 255), -1)

            remaining = deadline - time.time()
            cv2.putText(
                frame,
                f"Look {label}  ({remaining:.1f}s)",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2,
            )

            gx, gy = gaze_estimator.estimate(frame)
            if gx is not None:
                samples_x.append(gx)
                samples_y.append(gy)

            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC aborts
                cv2.destroyWindow("Calibration")
                return CalibrationData()

        readings[label] = (
            float(np.mean(samples_x)) if samples_x else 0.0,
            float(np.mean(samples_y)) if samples_y else 0.0,
        )
        print(f"  → gaze mean: x={readings[label][0]:.3f}  y={readings[label][1]:.3f}")

    cv2.destroyWindow("Calibration")

    cd = CalibrationData()

    # Centre offset
    centre_x, centre_y = readings.get("CENTRE", (0.0, 0.0))
    cd.gaze_x_offset = centre_x
    cd.gaze_y_offset = centre_y

    # Scale: how far the gaze moves between left and right fixation
    left_x  = readings.get("LEFT",  (0.0, 0.0))[0]
    right_x = readings.get("RIGHT", (0.0, 0.0))[0]
    up_y    = readings.get("UP",    (0.0, 0.0))[1]
    down_y  = readings.get("DOWN",  (0.0, 0.0))[1]

    span_x = abs(right_x - left_x)
    span_y = abs(down_y  - up_y)

    cd.gaze_x_scale = 2.0 / span_x  if span_x > 0.01 else 1.0
    cd.gaze_y_scale = 2.0 / span_y  if span_y > 0.01 else 1.0

    cd.save()
    print("[Calibration] Done. Data saved.")
    return cd
