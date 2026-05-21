"""
9-Punkt Gitter-Kalibrierung mit affiner Transformation.

Verbesserungen gegenüber der alten 5-Punkt Version:
  - 9 Punkte (3×3 Gitter) → bessere Abdeckung der Randbereiche
  - Affine Transformation (6 Parameter) statt nur Offset+Scale
    → korrigiert auch Schräg-Fehler und Achsen-Übersprechen
  - Stabilitätserkennung: verwirft Samples bei Augenzittern
  - Live-Gaze-Dot zeigt wo das System gerade hinschaut
  - Ausreißer-Filterung per Median statt Mean
"""

import json
import os
import time
import numpy as np

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "calibration_data.json")

SAMPLE_DURATION   = 2.5    # Sekunden pro Fixationspunkt
STABILITY_MAX_STD = 0.04   # Max erlaubte Standardabweichung (norm. Einheiten)

# 9 Punkte als normalisierte (nx, ny) Bildschirmkoordinaten
CALIBRATION_POINTS = [
    ("MITTE",        (0.50, 0.50)),
    ("LINKS",        (0.10, 0.50)),
    ("RECHTS",       (0.90, 0.50)),
    ("OBEN",         (0.50, 0.10)),
    ("UNTEN",        (0.50, 0.90)),
    ("OBEN-LINKS",   (0.12, 0.15)),
    ("OBEN-RECHTS",  (0.88, 0.15)),
    ("UNTEN-LINKS",  (0.12, 0.85)),
    ("UNTEN-RECHTS", (0.88, 0.85)),
]

# Ziel-Gaze-Werte für jeden Punkt (normalisierter [-1,+1] Raum)
_TARGET_GAZE = {
    "MITTE":        ( 0.00,  0.00),
    "LINKS":        (-1.00,  0.00),
    "RECHTS":       ( 1.00,  0.00),
    "OBEN":         ( 0.00, -1.00),
    "UNTEN":        ( 0.00,  1.00),
    "OBEN-LINKS":   (-1.00, -1.00),
    "OBEN-RECHTS":  ( 1.00, -1.00),
    "UNTEN-LINKS":  (-1.00,  1.00),
    "UNTEN-RECHTS": ( 1.00,  1.00),
}


class CalibrationData:
    """
    Affine Transformation:
      [gx_korr]   [a  b] [gx_roh]   [tx]
      [gy_korr] = [c  d] [gy_roh] + [ty]
    """

    def __init__(self):
        self.matrix = np.eye(2)
        self.offset = np.zeros(2)

    def apply(self, gaze_x: float, gaze_y: float):
        raw = np.array([gaze_x, gaze_y])
        out = self.matrix @ raw + self.offset
        return float(out[0]), float(out[1])

    def save(self):
        with open(CALIBRATION_FILE, "w") as f:
            json.dump({
                "matrix": self.matrix.tolist(),
                "offset": self.offset.tolist(),
            }, f, indent=2)

    @classmethod
    def load(cls) -> "CalibrationData":
        cd = cls()
        if os.path.exists(CALIBRATION_FILE):
            try:
                with open(CALIBRATION_FILE) as f:
                    data = json.load(f)
                cd.matrix = np.array(data["matrix"])
                cd.offset = np.array(data["offset"])
            except Exception:
                pass
        return cd


def _robust_mean(samples: list) -> tuple:
    """Median-basierter Mittelwert — filtert Ausreißer raus."""
    if not samples:
        return 0.0, 0.0
    xs = np.array([s[0] for s in samples])
    ys = np.array([s[1] for s in samples])

    def clean(arr):
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        mask = np.abs(arr - med) < max(2.5 * mad, 0.001)
        return float(np.mean(arr[mask])) if mask.any() else float(med)

    return clean(xs), clean(ys)


def _is_stable(recent: list) -> bool:
    if len(recent) < 6:
        return False
    xs = [s[0] for s in recent]
    ys = [s[1] for s in recent]
    return np.std(xs) < STABILITY_MAX_STD and np.std(ys) < STABILITY_MAX_STD


def run_calibration(gaze_estimator, cap) -> CalibrationData:
    """
    Interaktive 9-Punkt Kalibrierung.
    Gibt CalibrationData mit affiner Transformation zurück.
    """
    if not CV2_OK:
        raise ImportError("opencv-python ist erforderlich.")

    raw_readings = {}
    WIN = "Kalibrierung  (ESC = Abbrechen)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for idx, (label, (nx, ny)) in enumerate(CALIBRATION_POINTS):
        samples = []
        recent  = []
        deadline = time.time() + SAMPLE_DURATION

        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # Abgedunkelter Hintergrund
            dark = frame.copy()
            cv2.rectangle(dark, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(dark, 0.45, frame, 0.55, 0, frame)

            cx, cy    = int(nx * w), int(ny * h)
            remaining = deadline - time.time()
            progress  = 1.0 - remaining / SAMPLE_DURATION
            stable    = _is_stable(recent)

            # Countdown-Ring
            for a in range(0, int(360 * progress), 3):
                rad = np.radians(a - 90)
                px  = int(cx + 26 * np.cos(rad))
                py  = int(cy + 26 * np.sin(rad))
                cv2.circle(frame, (px, py), 2, (0, 200, 255), -1)

            # Fixationspunkt
            dot_col = (0, 255, 100) if stable else (0, 200, 255)
            cv2.circle(frame, (cx, cy), 18, (40, 40, 60), -1)
            cv2.circle(frame, (cx, cy), 10, dot_col, -1)
            cv2.circle(frame, (cx, cy),  3, (255, 255, 255), -1)

            # Fortschritts-Header
            bar_w = int(w * progress)
            cv2.rectangle(frame, (0, 0), (bar_w, 6), (0, 200, 255), -1)
            cv2.putText(frame, f"Schau auf:  {label}  ({idx+1}/9)",
                        (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"{remaining:.1f}s",
                        (w - 80, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 200, 255), 2)

            # Gaze schätzen + Live-Dot
            gx, gy = gaze_estimator.estimate(frame)
            if gx is not None:
                recent.append((gx, gy))
                if len(recent) > 15:
                    recent.pop(0)

                # Live-Blickpunkt auf Kamerabild
                dot_x = int(np.clip(w / 2 + gx * w / 3, 8, w - 8))
                dot_y = int(np.clip(h / 2 + gy * h / 3, 8, h - 8))
                cv2.circle(frame, (dot_x, dot_y), 9, (200, 60, 255), -1)
                cv2.circle(frame, (dot_x, dot_y), 4, (255, 200, 255), -1)

                if stable:
                    samples.append((gx, gy))
                    cv2.putText(frame, f"Stabil  {len(samples)} samples",
                                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 100), 1)
                else:
                    cv2.putText(frame, "Halte den Blick ruhig ...",
                                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 180, 255), 1)

            cv2.imshow("Kalibrierung  (ESC = Abbrechen)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow("Kalibrierung  (ESC = Abbrechen)")
                return CalibrationData()

        mx, my = _robust_mean(samples)
        raw_readings[label] = (mx, my)
        print(f"  [{label:<14}]  roh=({mx:+.3f}, {my:+.3f})  "
              f"stabile Samples: {len(samples)}")

    cv2.destroyWindow("Kalibrierung  (ESC = Abbrechen)")

    # ── Affine Transformation per Least-Squares ──────────────────────────
    labels = [l for l in raw_readings if l in _TARGET_GAZE]
    if len(labels) < 4:
        print("[Kalibrierung] Zu wenige Punkte — Identität wird verwendet")
        return CalibrationData()

    raw_pts    = np.array([raw_readings[l] for l in labels])
    target_pts = np.array([_TARGET_GAZE[l]  for l in labels])

    # Löse: target = [raw | 1] @ params  (least-squares)
    A = np.hstack([raw_pts, np.ones((len(raw_pts), 1))])
    params, _, _, _ = np.linalg.lstsq(A, target_pts, rcond=None)  # (3,2)

    cd = CalibrationData()
    cd.matrix = params[:2].T   # 2×2
    cd.offset = params[2]      # (2,)
    cd.save()

    predicted = raw_pts @ cd.matrix.T + cd.offset
    residuals = np.linalg.norm(predicted - target_pts, axis=1)
    print(f"\n[Kalibrierung] Abgeschlossen.")
    print(f"  Mittlerer Fehler: {residuals.mean():.3f}  "
          f"Max: {residuals.max():.3f}  "
          f"({'gut' if residuals.mean() < 0.15 else 'akzeptabel' if residuals.mean() < 0.3 else 'wiederholen?'})")

    return cd
