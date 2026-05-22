"""
SOLOASSIST II — Eye Tracking Controller
========================================
Maximale Genauigkeit durch:
  • L2CS-Net Gaze Estimation (neuronales Netz, optional)
  • 9-Punkt-Kalibrierung (individuelle Anpassung)
  • Kalman-Filter (optimale Glättung)

Ablauf:
  1. Beim ersten Start → Kalibrierung (9 Punkte anschauen)
  2. Danach → direktes Starten mit gespeicherter Kalibrierung

Tasten:
  ESC / Q  →  Beenden
  SPACE    →  Pause / Weiter
  C        →  Neu kalibrieren
"""

import sys
import time
import subprocess
import threading
import argparse

import cv2
import os
import numpy as np

from gaze.estimator import GazeEstimator
from gaze.calibration import GazeCalibration, CALIB_POINTS
from controller.arm_controller import SoloAssistController

# ── Argumente ─────────────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser()
_ap.add_argument("--simulate",         action="store_true")
_ap.add_argument("--ip",               default="127.0.0.1")
_ap.add_argument("--port",             default=5522, type=int)
_ap.add_argument("--skip-calibration", action="store_true",
                 help="Gespeicherte Kalibrierung nutzen ohne Neukalibrierung")
_args = _ap.parse_args()

# ── Einstellungen ─────────────────────────────────────────────────────────────
ARM_IP    = _args.ip
ARM_PORT  = _args.port
SIMULATE  = _args.simulate

WEBCAM       = 0
MAX_SPEED    = 100
DEADZONE     = 0.08
FACE_TIMEOUT = 2.0      # nur relevant bei Webcam-Modus

# Kalibrierungs-Timing
_SETTLE  = 0.7   # Sekunden warten bevor Messung startet (Auge anpassen)
_COLLECT = 1.5   # Sekunden Messwerte sammeln


# ── Bildschirmgrösse ──────────────────────────────────────────────────────────
def _screen_size():
    if sys.platform == "win32":
        try:
            import ctypes
            u = ctypes.windll.user32
            return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
        except Exception:
            pass
    else:
        try:
            out = subprocess.check_output(
                ["osascript", "-e",
                 'tell application "Finder" to get bounds of window of desktop'],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
            p = [int(x.strip()) for x in out.split(",")]
            return p[2], p[3]
        except Exception:
            pass
    return 1920, 1080

SCREEN_W, SCREEN_H = _screen_size()


# ── Gaze + Kalibrierung ───────────────────────────────────────────────────────
estimator   = GazeEstimator()
calibration = GazeCalibration()

_calib_loaded = calibration.load()
if _calib_loaded:
    print("[Kalibrierung] Gespeicherte Kalibrierung geladen")


# ── Arm ───────────────────────────────────────────────────────────────────────
class _SimArm:
    is_connected = True
    def connect(self):         return True
    def disconnect(self):      pass
    def stop(self):            pass
    def move_polar(self, lr, ud, io):
        if lr or ud:
            print(f"[SIM]  LR={lr:+4d}  UD={ud:+4d}")


arm = _SimArm() if SIMULATE else SoloAssistController(ARM_IP, ARM_PORT)
if not arm.connect():
    print(f"FEHLER: Kein Arm bei {ARM_IP}:{ARM_PORT}")
    sys.exit(1)


# ── Arm-Schleife (20 Hz) ──────────────────────────────────────────────────────
_cmd  = [0, 0, 0]
_lock = threading.Lock()
_run  = [True]

def _arm_loop():
    while _run[0]:
        time.sleep(0.05)
        with _lock:
            lr, ud, io = _cmd
        arm.move_polar(lr, ud, io)

threading.Thread(target=_arm_loop, daemon=True).start()


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────
def _to_speed(v: float) -> int:
    if abs(v) < DEADZONE:
        return 0
    sign = 1 if v > 0 else -1
    return int(sign * (abs(v) - DEADZONE) / (1.0 - DEADZONE) * MAX_SPEED)


# ── Kamera (optional — wird bei Tobii nur für das Display genutzt) ────────────
cap = cv2.VideoCapture(WEBCAM)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)
_has_camera = cap.isOpened()
if not _has_camera:
    print(f"[Kamera] Keine Webcam — Display zeigt schwarzes Bild")
_black_frame = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

WIN = "SOLOASSIST Eye Tracking"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, SCREEN_W, SCREEN_H)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# ── Zustand ───────────────────────────────────────────────────────────────────
STATE = "running" if (_calib_loaded and _args.skip_calibration) or _calib_loaded else "calibrating"

_ci       = 0        # aktueller Kalibrierungs-Punkt Index
_cbuf     = []       # Messwerte für aktuellen Punkt
_ct0      = 0.0      # Zeitpunkt Punktanzeige-Start
_sx       = 0.0      # geglätteter Gaze X (nur im running-Modus)
_sy       = 0.0      # geglätteter Gaze Y

last_face = time.time()
paused    = False
fps_buf   = []


def _start_calibration():
    global STATE, _ci, _cbuf, _ct0, _sx, _sy
    calibration.reset()
    estimator.reset_filter()
    STATE = "calibrating"
    _ci   = 0
    _cbuf = []
    _ct0  = time.time()
    _sx   = 0.0
    _sy   = 0.0

if STATE == "calibrating":
    _ct0 = time.time()

print(f"Gestartet. {SCREEN_W}×{SCREEN_H}  |  ESC=Beenden  SPACE=Pause  C=Kalibrieren")


# ── Hauptschleife ─────────────────────────────────────────────────────────────
while True:
    now = time.time()
    fps_buf = [t for t in fps_buf + [now] if now - t < 1.0]
    fps     = len(fps_buf)

    # Kamerabild (nur für Display)
    if _has_camera:
        ok, frame = cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            disp = cv2.resize(frame, (SCREEN_W, SCREEN_H), interpolation=cv2.INTER_LINEAR)
        else:
            disp = _black_frame.copy()
    else:
        disp = _black_frame.copy()

    # Gaze schätzen (Tobii: kein Frame nötig; Webcam: Frame wird genutzt)
    raw_x, raw_y = estimator.estimate(frame if _has_camera else None)
    if raw_x is not None:
        last_face = now
    face_ok = (now - last_face) < FACE_TIMEOUT or estimator.mode == "tobii"

    # ── KALIBRIERUNG ──────────────────────────────────────────────────────
    if STATE == "calibrating":
        pt      = CALIB_POINTS[_ci]
        px      = int(pt[0] * SCREEN_W)
        py      = int(pt[1] * SCREEN_H)
        elapsed = now - _ct0

        # Hintergrund abdunkeln
        dark = disp.copy()
        cv2.rectangle(dark, (0, 0), (SCREEN_W, SCREEN_H), (0, 0, 0), -1)
        cv2.addWeighted(dark, 0.55, disp, 0.45, 0, disp)

        collecting = elapsed > _SETTLE
        progress   = np.clip((elapsed - _SETTLE) / _COLLECT, 0.0, 1.0)

        # Kreise zeichnen
        ring = (0, 230, 80) if collecting else (80, 80, 220)
        cv2.circle(disp, (px, py), 28, (255, 255, 255), -1)
        cv2.circle(disp, (px, py), 28, (0, 0, 0),       3)
        cv2.circle(disp, (px, py), 10, (0, 0, 0),       -1)

        # Fortschritts-Ring
        if collecting:
            angle = int(360 * progress)
            cv2.ellipse(disp, (px, py), (40, 40), -90, 0, angle, ring, 5)

        # Text
        msg = f"Schau auf den Punkt  ({_ci + 1} / {len(CALIB_POINTS)})"
        tw  = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
        cv2.putText(disp, msg,
                    ((SCREEN_W - tw) // 2, SCREEN_H - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        if not face_ok:
            cv2.putText(disp, "Kein Gesicht erkannt",
                        (SCREEN_W // 2 - 160, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2, cv2.LINE_AA)

        # Messwerte sammeln
        if collecting and raw_x is not None:
            _cbuf.append((raw_x, raw_y))

        # Punkt abgeschlossen
        if elapsed >= _SETTLE + _COLLECT:
            if _cbuf:
                ax = float(np.mean([s[0] for s in _cbuf]))
                ay = float(np.mean([s[1] for s in _cbuf]))
                # Ziel in [-1, +1]: links=-1, rechts=+1, oben=+1, unten=-1
                tx = (pt[0] - 0.5) * 2.0
                ty = -(pt[1] - 0.5) * 2.0
                calibration.add_sample(ax, ay, tx, ty)

            _ci  += 1
            _cbuf = []
            _ct0  = now

            if _ci >= len(CALIB_POINTS):
                if calibration.fit():
                    calibration.save()
                    print("[Kalibrierung] Abgeschlossen und gespeichert")
                else:
                    print("[Kalibrierung] Fehlgeschlagen, starte neu")
                    _start_calibration()
                    continue
                STATE = "running"
                estimator.reset_filter()

        with _lock:
            _cmd[:] = [0, 0, 0]

    # ── TRACKING ──────────────────────────────────────────────────────────
    else:
        if raw_x is not None:
            cal_x, cal_y = calibration.transform(raw_x, raw_y)
            # Sanfter Tiefpassfilter nach Kalman (für letzte Stabilität)
            _sx = 0.55 * cal_x + 0.45 * _sx
            _sy = 0.55 * cal_y + 0.45 * _sy

        gx = float(np.clip(_sx, -1.0, 1.0))
        gy = float(np.clip(_sy, -1.0, 1.0))

        lr = _to_speed(gx)
        ud = _to_speed(gy)

        if face_ok and not paused and raw_x is not None:
            with _lock:
                _cmd[:] = [lr, ud, 0]
        else:
            with _lock:
                _cmd[:] = [0, 0, 0]

        # Cursor
        cx      = int(np.clip((gx + 1) / 2, 0, 1) * SCREEN_W)
        cy      = int(np.clip((-gy + 1) / 2, 0, 1) * SCREEN_H)
        in_dz   = abs(gx) < DEADZONE and abs(gy) < DEADZONE
        dot_col = (100, 100, 255) if in_dz else (0, 220, 50)
        cv2.circle(disp, (cx, cy), 18, (0, 0, 0),     -1)
        cv2.circle(disp, (cx, cy), 14, dot_col,        -1)
        cv2.circle(disp, (cx, cy), 14, (255, 255, 255), 2)

        # HUD
        status     = "PAUSE" if paused else ("KEIN GESICHT" if not face_ok else
                     f"LR={lr:+4d}  UD={ud:+4d}")
        status_col = (0, 80, 255) if (not face_ok or paused) else (255, 255, 255)
        cv2.rectangle(disp, (0, 0), (SCREEN_W, 55), (20, 20, 20), -1)
        cv2.putText(disp, f"FPS: {fps:2d}   {status}",
                    (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_col, 2, cv2.LINE_AA)
        cv2.putText(disp, "ESC/Q: Beenden    SPACE: Pause    C: Kalibrieren",
                    (14, SCREEN_H - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (100, 100, 100), 1, cv2.LINE_AA)

    cv2.imshow(WIN, disp)

    # ── Tasten ────────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord("q"), ord("Q")):
        break
    elif key == ord(" "):
        paused = not paused
        if paused:
            arm.stop()
    elif key in (ord("c"), ord("C")):
        _start_calibration()


# ── Aufräumen ─────────────────────────────────────────────────────────────────
_run[0] = False
arm.stop()
arm.disconnect()
cap.release()
cv2.destroyAllWindows()
print("Beendet.")
