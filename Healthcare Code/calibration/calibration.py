"""
9-Punkt Gitter-Kalibrierung mit Polynomial-Regression (Grad 2).

Verbesserungen gegenüber affiner Transformation:
  - Polynomial Grad 2: korrigiert auch nichtlineare Verzerrungen an den Rändern
    Basis: [1, gx, gy, gx², gx·gy, gy²]  → 6 Parameter pro Achse
  - Mit 9 Messpunkten überbestimmtes System → robuster gegen Messrauschen
  - Stabilitätserkennung: verwirft Samples bei Augenzittern
  - Live-Gaze-Dot zeigt aktuellen Messwert
  - Ausreißer-Filterung per Median/MAD
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
STABILITY_MAX_STD = 0.03   # Max erlaubte Standardabweichung

# 9 Punkte als normalisierte (nx, ny) Bildschirmkoordinaten
# + dazugehörige Ziel-Gaze-Werte im [-1,+1]-Raum
CALIBRATION_POINTS = [
    ("MITTE",        (0.50, 0.50), ( 0.00,  0.00)),
    ("LINKS",        (0.08, 0.50), (-1.00,  0.00)),
    ("RECHTS",       (0.92, 0.50), ( 1.00,  0.00)),
    ("OBEN",         (0.50, 0.08), ( 0.00, -1.00)),
    ("UNTEN",        (0.50, 0.92), ( 0.00,  1.00)),
    ("OBEN-LINKS",   (0.08, 0.08), (-1.00, -1.00)),
    ("OBEN-RECHTS",  (0.92, 0.08), ( 1.00, -1.00)),
    ("UNTEN-LINKS",  (0.08, 0.92), (-1.00,  1.00)),
    ("UNTEN-RECHTS", (0.92, 0.92), ( 1.00,  1.00)),
]


def _poly_features(gx: float, gy: float) -> np.ndarray:
    """Polynomial basis: [1, gx, gy, gx², gx·gy, gy²]"""
    return np.array([1.0, gx, gy, gx * gx, gx * gy, gy * gy])


class CalibrationData:
    """
    Polynomial (degree-2) gaze correction.

    Maps raw (gx, gy) → corrected (gx, gy) via:
        corrected = poly_coeffs @ [1, gx, gy, gx², gx·gy, gy²]
    where poly_coeffs is a (2, 6) matrix.

    Identity transform (no calibration):
        poly_coeffs = [[0, 1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0]]
    """

    def __init__(self):
        # Identity: pass raw values through unchanged
        self.poly_coeffs = np.array([
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ])

    def apply(self, gaze_x: float, gaze_y: float):
        phi = _poly_features(gaze_x, gaze_y)
        out = self.poly_coeffs @ phi
        return float(out[0]), float(out[1])

    def save(self):
        with open(CALIBRATION_FILE, "w") as f:
            json.dump({"poly_coeffs": self.poly_coeffs.tolist()}, f, indent=2)

    @classmethod
    def load(cls) -> "CalibrationData":
        cd = cls()
        if not os.path.exists(CALIBRATION_FILE):
            return cd
        try:
            with open(CALIBRATION_FILE) as f:
                data = json.load(f)
            if "poly_coeffs" in data:
                cd.poly_coeffs = np.array(data["poly_coeffs"])
            elif "matrix" in data and "offset" in data:
                # Migrate old affine format
                M = np.array(data["matrix"])    # (2,2)
                b = np.array(data["offset"])    # (2,)
                # affine: out = M @ raw + b
                # poly basis [1, gx, gy, gx², gx·gy, gy²]
                # affine uses only [1, gx, gy] → embed into poly
                cd.poly_coeffs = np.array([
                    [b[0], M[0, 0], M[0, 1], 0.0, 0.0, 0.0],
                    [b[1], M[1, 0], M[1, 1], 0.0, 0.0, 0.0],
                ])
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
    Interaktive 9-Punkt Kalibrierung mit Polynomial-Regression.
    Gibt CalibrationData zurück.
    """
    if not CV2_OK:
        raise ImportError("opencv-python ist erforderlich.")

    raw_readings  = {}
    target_lookup = {}
    WIN = "Kalibrierung  (ESC = Abbrechen)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for idx, (label, (nx, ny), target_gaze) in enumerate(CALIBRATION_POINTS):
        samples  = []
        recent   = []
        deadline = time.time() + SAMPLE_DURATION

        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

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
            cv2.circle(frame, (cx, cy), 20, (40, 40, 60), -1)
            cv2.circle(frame, (cx, cy), 12, dot_col, -1)
            cv2.circle(frame, (cx, cy),  4, (255, 255, 255), -1)

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

                dot_x = int(np.clip(w / 2 + gx * w / 2.5, 8, w - 8))
                dot_y = int(np.clip(h / 2 + gy * h / 2.5, 8, h - 8))
                cv2.circle(frame, (dot_x, dot_y), 9, (200, 60, 255), -1)
                cv2.circle(frame, (dot_x, dot_y), 4, (255, 200, 255), -1)

                if stable:
                    samples.append((gx, gy))
                    cv2.putText(frame, f"Stabil  {len(samples)} Samples",
                                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 100), 1)
                else:
                    cv2.putText(frame, "Halte den Blick ruhig ...",
                                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 180, 255), 1)

            cv2.imshow(WIN, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow(WIN)
                return CalibrationData()

        mx, my = _robust_mean(samples)
        raw_readings[label]  = (mx, my)
        target_lookup[label] = target_gaze
        print(f"  [{label:<14}]  roh=({mx:+.4f}, {my:+.4f})  "
              f"ziel=({target_gaze[0]:+.2f}, {target_gaze[1]:+.2f})  "
              f"stabile Samples: {len(samples)}")

    cv2.destroyWindow(WIN)

    # ── Polynomial Regression Grad 2 (Least-Squares) ──────────────────────
    labels = [l for l, _, _ in CALIBRATION_POINTS if l in raw_readings]
    if len(labels) < 6:
        print("[Kalibrierung] Zu wenige Punkte — Identität wird verwendet")
        return CalibrationData()

    raw_pts    = np.array([raw_readings[l]  for l in labels])    # (N, 2)
    target_pts = np.array([target_lookup[l] for l in labels])    # (N, 2)

    # Build polynomial feature matrix: each row = [1, gx, gy, gx², gx·gy, gy²]
    A = np.column_stack([_poly_features(gx, gy) for gx, gy in raw_pts]).T  # (N, 6)

    # Solve: A @ C.T = target_pts  →  C.T = lstsq solution  (6, 2)
    sol, _, _, _ = np.linalg.lstsq(A, target_pts, rcond=None)   # (6, 2)

    cd = CalibrationData()
    cd.poly_coeffs = sol.T   # (2, 6)
    cd.save()

    # Quality report
    predicted = A @ sol                                              # (N, 2)
    residuals = np.linalg.norm(predicted - target_pts, axis=1)
    mean_err  = residuals.mean()
    quality   = "gut" if mean_err < 0.12 else ("akzeptabel" if mean_err < 0.25 else "wiederholen?")
    print(f"\n[Kalibrierung] Abgeschlossen — Polynomial Grad 2")
    print(f"  Mittlerer Fehler: {mean_err:.3f}  Max: {residuals.max():.3f}  ({quality})")
    for label, err in zip(labels, residuals):
        mark = "✓" if err < 0.15 else ("~" if err < 0.30 else "✗")
        print(f"    {mark} {label:<14} Fehler={err:.3f}")

    return cd
