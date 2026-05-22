"""
9-point gaze calibration.

Maps raw estimator output → normalized [-1, +1] screen coordinates
using polynomial regression (degree 2, 6 features).

Saves/loads from calibration.json so re-calibration is optional.
"""

import json
import numpy as np
from pathlib import Path

CALIB_FILE = "calibration.json"

# 9 calibration targets as (screen_x, screen_y) in [0, 1]
# top-left → top-right → middle row → bottom row
CALIB_POINTS = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
]


def _features(rx: np.ndarray, ry: np.ndarray) -> np.ndarray:
    """Build polynomial feature matrix [1, rx, ry, rx², ry², rx·ry]."""
    return np.column_stack([
        np.ones_like(rx), rx, ry, rx ** 2, ry ** 2, rx * ry,
    ])


class GazeCalibration:
    """
    Polynomial calibration mapping:
      raw (gx, gy)  →  calibrated (cx, cy) in [-1, +1]

    Usage:
      cal = GazeCalibration()
      cal.load()  # optional: load previous calibration

      # during calibration:
      cal.add_sample(raw_x, raw_y, target_x, target_y)
      cal.fit()
      cal.save()

      # during tracking:
      cx, cy = cal.transform(raw_x, raw_y)
    """

    def __init__(self):
        self._cx_coef: np.ndarray | None = None
        self._cy_coef: np.ndarray | None = None
        self._samples: list[tuple] = []

    def reset(self):
        self._cx_coef = None
        self._cy_coef = None
        self._samples = []

    @property
    def is_fitted(self) -> bool:
        return self._cx_coef is not None

    def add_sample(self, raw_x: float, raw_y: float,
                   target_x: float, target_y: float):
        self._samples.append((raw_x, raw_y, target_x, target_y))

    def fit(self) -> bool:
        if len(self._samples) < len(CALIB_POINTS):
            return False
        d = np.array(self._samples)
        X = _features(d[:, 0], d[:, 1])
        self._cx_coef = np.linalg.lstsq(X, d[:, 2], rcond=None)[0]
        self._cy_coef = np.linalg.lstsq(X, d[:, 3], rcond=None)[0]
        return True

    def transform(self, gx: float, gy: float) -> tuple[float, float]:
        if not self.is_fitted:
            return gx, gy
        X = _features(np.array([gx]), np.array([gy]))
        cx = float(X @ self._cx_coef)
        cy = float(X @ self._cy_coef)
        return float(np.clip(cx, -1.2, 1.2)), float(np.clip(cy, -1.2, 1.2))

    def save(self, path: str = CALIB_FILE):
        if not self.is_fitted:
            return
        Path(path).write_text(json.dumps({
            "cx": self._cx_coef.tolist(),
            "cy": self._cy_coef.tolist(),
            "samples": [list(s) for s in self._samples],
        }))

    def load(self, path: str = CALIB_FILE) -> bool:
        try:
            d = json.loads(Path(path).read_text())
            self._cx_coef = np.array(d["cx"])
            self._cy_coef = np.array(d["cy"])
            self._samples = [tuple(s) for s in d["samples"]]
            return True
        except Exception:
            return False
