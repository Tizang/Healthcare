"""
Gaze estimator — uses L2CS-Net if available, falls back to MediaPipe iris tracking.

L2CS-Net setup (optional, higher accuracy):
  pip install torch torchvision l2cs
  Download weights from:
    https://drive.google.com/file/d/1eUr4OALR9K9P1WVuKTqBnXMa65JEm0Gh/
  Place at: Healthcare Code/models/L2CSNet_gaze360.pkl
"""

import time
import numpy as np
import cv2

from gaze.filters import KalmanFilter1D  # noqa: F401  (re-exported for back-compat)

# ── Try L2CS-Net ──────────────────────────────────────────────────────────────
_l2cs_available = False
try:
    import torch
    from l2cs import Pipeline as L2CSPipeline
    _l2cs_available = True
except ImportError:
    pass

import os as _os
_BASE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
L2CS_WEIGHTS = _os.path.join(_BASE, "models", "L2CSNet_gaze360.pkl")
_MP_MODEL_DEFAULT = _os.path.join(_BASE, "face_landmarker.task")

# ── MediaPipe landmarks ───────────────────────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

_L_INNER,  _L_OUTER  = 133, 33
_L_TOP,    _L_BOT    = 159, 145
_R_INNER,  _R_OUTER  = 362, 263
_R_TOP,    _R_BOT    = 386, 374
_L_IRIS,   _R_IRIS   = 468, 473



class GazeEstimator:
    """
    Estimates gaze direction as (gx, gy) in raw sensor units.
    Priority: tobii-research → Cursor/pyautogui → L2CS-Net → MediaPipe.

    After calibration, pass output through GazeCalibration.transform().
    """

    def __init__(self, screen_w: int = 1920, screen_h: int = 1080,
                 mediapipe_model: str = _MP_MODEL_DEFAULT):
        self.mode = "mediapipe"
        self._kx = KalmanFilter1D()
        self._ky = KalmanFilter1D()
        self._delegate = None

        # 1. tobii-research SDK (Python ≤ 3.10)
        try:
            from gaze.tobii_estimator import TobiiEstimator
            self._delegate = TobiiEstimator()
            self.mode = "tobii"
            return
        except Exception as e:
            print(f"[Gaze] tobii-research nicht verfügbar ({e})")

        # 2. Cursor-Modus via pyautogui (Tobii Experience steuert Cursor)
        try:
            from gaze.cursor_estimator import CursorEstimator
            self._delegate = CursorEstimator(screen_w, screen_h)
            self.mode = "cursor"
            return
        except Exception as e:
            print(f"[Gaze] Cursor-Modus nicht verfügbar ({e}), nutze Kamera")

        # 3. L2CS-Net
        if _l2cs_available:
            import os
            if os.path.exists(L2CS_WEIGHTS):
                try:
                    self._pipeline = L2CSPipeline(
                        weights=L2CS_WEIGHTS,
                        arch="ResNet50",
                        device=torch.device("cpu"),
                        include_detector=True,
                    )
                    self.mode = "l2cs"
                    print("[Gaze] L2CS-Net aktiv")
                except Exception as e:
                    print(f"[Gaze] L2CS-Net Fehler ({e}), nutze MediaPipe")
            else:
                print(f"[Gaze] L2CS-Net Gewichte nicht gefunden unter: {L2CS_WEIGHTS}")
                print("[Gaze] Download: https://drive.google.com/file/d/1eUr4OALR9K9P1WVuKTqBnXMa65JEm0Gh/")
                print("[Gaze] Datei ablegen unter: Healthcare Code/models/L2CSNet_gaze360.pkl")
                print("[Gaze] Nutze MediaPipe als Fallback")

        if self.mode == "mediapipe":
            opts = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=mediapipe_model),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(opts)
            self._t0 = time.time()
            print("[Gaze] MediaPipe aktiv")

    def estimate(self, bgr_frame: np.ndarray | None = None):
        """
        Returns smoothed (gx, gy), or (None, None) if no face/gaze detected.
        gx: negative=links, positive=rechts
        gy: negative=unten, positive=oben
        """
        if self._delegate is not None:
            return self._delegate.estimate(bgr_frame)

        if self.mode == "l2cs":
            raw = self._raw_l2cs(bgr_frame)
        else:
            raw = self._raw_mediapipe(bgr_frame)

        if raw[0] is None:
            return None, None

        gx = self._kx.update(raw[0])
        gy = self._ky.update(raw[1])
        return gx, gy

    def reset_filter(self):
        if self._delegate is not None:
            self._delegate.reset_filter()
            return
        self._kx.reset()
        self._ky.reset()

    # ── L2CS-Net ──────────────────────────────────────────────────────────────

    def _raw_l2cs(self, frame):
        results = self._pipeline.step(frame)
        if results is None or len(results.yaw) == 0:
            return None, None
        # Radians → normalized; ±0.52 rad (30°) ≈ ±1.0
        yaw   =  float(results.yaw[0])   / 0.52
        pitch = -float(results.pitch[0]) / 0.35   # up = positive
        return float(np.clip(yaw, -2, 2)), float(np.clip(pitch, -2, 2))

    # ── MediaPipe ─────────────────────────────────────────────────────────────

    def _raw_mediapipe(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts  = int((time.time() - self._t0) * 1000)
        res = self._landmarker.detect_for_video(img, ts)

        if not res.face_landmarks:
            return None, None

        lm = res.face_landmarks[0]

        def _eye(inner, outer, top, bot, iris):
            cx = (lm[inner].x + lm[outer].x) / 2
            cy = (lm[inner].y + lm[outer].y) / 2
            ew = abs(lm[outer].x - lm[inner].x)
            eh = abs(lm[bot].y   - lm[top].y)
            if ew < 0.001 or eh < 0.001:
                return 0.0, 0.0
            return (lm[iris].x - cx) / (ew / 2), (lm[iris].y - cy) / (eh / 2)

        lx, ly = _eye(_L_INNER, _L_OUTER, _L_TOP, _L_BOT, _L_IRIS)
        rx, ry = _eye(_R_INNER, _R_OUTER, _R_TOP, _R_BOT, _R_IRIS)
        return float((lx + rx) / 2), float((ly + ry) / 2)
