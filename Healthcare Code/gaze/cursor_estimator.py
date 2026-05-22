"""
Cursor-basierter Gaze Estimator.

Funktioniert mit jedem Eye Tracker der den Windows-Cursor steuert,
z.B. Tobii 4C über Tobii Experience Software.

Setup:
  1. Tobii Experience installieren und Tobii 4C kalibrieren
  2. In den Tobii-Einstellungen: "Cursor-Steuerung" aktivieren
  3. pip install pyautogui
  → Python liest die Cursor-Position als Gaze-Signal
"""

import time
import threading
import numpy as np

from gaze.estimator import KalmanFilter1D


class CursorEstimator:
    """
    Liest Gaze-Daten aus der System-Cursor-Position (60 Hz Polling).
    Erwartet dass der Eye Tracker den Cursor steuert.
    """

    def __init__(self, screen_w: int, screen_h: int):
        import pyautogui
        pyautogui.FAILSAFE = False   # kein Abbruch wenn Cursor in Ecke
        self._pa = pyautogui
        self._sw = screen_w
        self._sh = screen_h

        self._gx:   float = 0.0
        self._gy:   float = 0.0
        self._last: float = 0.0
        self._lock  = threading.Lock()
        self._kx    = KalmanFilter1D(process_var=3e-4, measure_var=0.03)
        self._ky    = KalmanFilter1D(process_var=3e-4, measure_var=0.03)
        self._running = True

        threading.Thread(target=self._poll, daemon=True).start()
        print("[Gaze] Cursor-Modus aktiv — Tobii muss Cursor-Steuerung haben")

    def _poll(self):
        while self._running:
            try:
                x, y = self._pa.position()
                # Normalisiere [0, screen] → [-1, +1], Y invertiert (oben = +1)
                raw_gx =  (x / self._sw - 0.5) * 2.0
                raw_gy = -(y / self._sh - 0.5) * 2.0
                gx = self._kx.update(raw_gx)
                gy = self._ky.update(raw_gy)
                with self._lock:
                    self._gx   = float(np.clip(gx, -1.5, 1.5))
                    self._gy   = float(np.clip(gy, -1.5, 1.5))
                    self._last = time.time()
            except Exception:
                pass
            time.sleep(1 / 60)

    def estimate(self, frame=None):
        with self._lock:
            if time.time() - self._last > 0.5:
                return None, None
            return self._gx, self._gy

    def reset_filter(self):
        self._kx.reset()
        self._ky.reset()

    def disconnect(self):
        self._running = False
