"""
Tobii 4C Gaze Estimator via tobii_research SDK.

Voraussetzungen (Windows):
  1. Tobii Experience Software installieren (liefert den Stream Engine Treiber)
  2. pip install tobii-research
  3. Tobii 4C per USB anschließen und in der Tobii-Software kalibrieren

Liefert (gx, gy) in [-1, +1]:
  gx: links=-1, rechts=+1
  gy: unten=-1, oben=+1
  Mitte des Bildschirms = (0, 0)
"""

import time
import threading
import numpy as np

from gaze.filters import KalmanFilter1D


class TobiiEstimator:
    def __init__(self):
        import tobii_research as tr

        trackers = tr.find_all_eyetrackers()
        if not trackers:
            raise RuntimeError("Kein Tobii Eye Tracker gefunden — USB angeschlossen?")

        self._tracker = trackers[0]
        print(f"[Tobii] Gerät gefunden: {self._tracker.model}  |  {self._tracker.serial_number}")

        self._gx:   float | None = None
        self._gy:   float | None = None
        self._last: float        = 0.0
        self._lock  = threading.Lock()
        self._kx    = KalmanFilter1D(process_var=1e-4, measure_var=0.02)
        self._ky    = KalmanFilter1D(process_var=1e-4, measure_var=0.02)

        self._tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, self._on_gaze,
                                   as_dictionary=True)

    def _on_gaze(self, data: dict):
        lv = data.get("left_gaze_point_validity",  0)
        rv = data.get("right_gaze_point_validity", 0)
        lp = data.get("left_gaze_point_on_display_area",  (float("nan"), float("nan")))
        rp = data.get("right_gaze_point_on_display_area", (float("nan"), float("nan")))

        pts = []
        if lv and not any(np.isnan(lp)):
            pts.append(lp)
        if rv and not any(np.isnan(rp)):
            pts.append(rp)

        if not pts:
            return

        ax = float(np.mean([p[0] for p in pts]))
        ay = float(np.mean([p[1] for p in pts]))

        # Tobii liefert [0,1] Bildschirmkoordinaten → in [-1,+1] umrechnen
        # Y invertiert: Tobii-Oben (y=0) → arm-oben (+1)
        raw_gx =  (ax - 0.5) * 2.0
        raw_gy = -(ay - 0.5) * 2.0

        gx = self._kx.update(raw_gx)
        gy = self._ky.update(raw_gy)

        with self._lock:
            self._gx   = float(np.clip(gx, -1.5, 1.5))
            self._gy   = float(np.clip(gy, -1.5, 1.5))
            self._last = time.time()

    def estimate(self, frame=None):
        """frame wird ignoriert — Tobii braucht kein Kamerabild."""
        with self._lock:
            if self._gx is None or (time.time() - self._last) > 0.4:
                return None, None
            return self._gx, self._gy

    def reset_filter(self):
        self._kx.reset()
        self._ky.reset()

    def disconnect(self):
        try:
            import tobii_research as tr
            self._tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self._on_gaze)
        except Exception:
            pass
