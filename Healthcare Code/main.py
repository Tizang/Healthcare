"""
SOLOASSIST II — Eye Tracking Controller
========================================
Basiert auf: github.com/soumyagautam/Eye-Mouse-Tracking

Wie es funktioniert:
  MediaPipe erkennt die Iris-Position im Kamerabild (Punkte 468–477).
  Diese Position wird direkt auf den Bildschirm und auf den Arm gemappt:
    Iris links  im Bild  →  Arm fährt links
    Iris rechts im Bild  →  Arm fährt rechts
    Iris oben   im Bild  →  Arm fährt hoch
    Iris unten  im Bild  →  Arm fährt runter

Tasten:
  ESC / Q   →  Beenden
  SPACE     →  Pause / Weiter
"""

import sys
import time
import subprocess
import threading
import argparse

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from controller.arm_controller import SoloAssistController

# ── Kommandozeilen-Argumente ──────────────────────────────────────────────────
_ap = argparse.ArgumentParser()
_ap.add_argument("--simulate", action="store_true", help="Kein echter Arm, nur Simulation")
_ap.add_argument("--ip",   default="192.168.1.100")
_ap.add_argument("--port", default=5000, type=int)
_args = _ap.parse_args()

# ── Einstellungen ─────────────────────────────────────────────────────────────

ARM_IP       = _args.ip
ARM_PORT     = _args.port
SIMULATE     = _args.simulate        # --simulate Flag oder unten auf True setzen

WEBCAM       = 0                 # Webcam-Index
MAX_SPEED    = 150               # Maximale Arm-Geschwindigkeit
DEADZONE     = 0.15              # Totzone in der Mitte (0.0–1.0)
SMOOTH       = 0.5               # Glättung: 0=eingefroren, 1=roh/direkt
GAZE_SCALE   = 3.5               # Verstärkung: roher Iris-Offset ≈ ±0.3 → ±1.0
Y_OFFSET     = 0.35              # Iris sitzt natürlicherweise über Augenmitte → korrigieren
FACE_TIMEOUT = 2.0               # Sekunden ohne Gesicht → Arm stoppt

MODEL = "face_landmarker.task"   # MediaPipe Modell (im selben Ordner)


# ── Bildschirmgrösse erkennen (macOS) ─────────────────────────────────────────

def _screen_size():
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
    return 1440, 900

SCREEN_W, SCREEN_H = _screen_size()


# ── MediaPipe FaceLandmarker ──────────────────────────────────────────────────

_face_opts = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
_landmarker = vision.FaceLandmarker.create_from_options(_face_opts)
_t0 = time.time()

# Augen-Eckpunkte zum Berechnen des Iris-Offsets
LEFT_INNER,  LEFT_OUTER  = 133, 33
LEFT_TOP,    LEFT_BOT    = 159, 145
RIGHT_INNER, RIGHT_OUTER = 362, 263
RIGHT_TOP,   RIGHT_BOT   = 386, 374
LEFT_IRIS_C, RIGHT_IRIS_C = 468, 473


def get_gaze(bgr_frame: np.ndarray):
    """
    Gibt (gaze_x, gaze_y) zurück — Iris-Offset vom Augenmittelpunkt.
    gx normalisiert auf Augenbreite, gy normalisiert auf Augenhöhe.
    Typischer Bereich: ±0.3. Gibt (None, None) zurück wenn kein Gesicht.
    """
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts  = int((time.time() - _t0) * 1000)
    res = _landmarker.detect_for_video(img, ts)

    if not res.face_landmarks:
        return None, None

    lm = res.face_landmarks[0]

    def eye_gaze(inner, outer, top, bot, iris_c):
        cx = (lm[inner].x + lm[outer].x) / 2
        cy = (lm[inner].y + lm[outer].y) / 2
        ew = abs(lm[outer].x - lm[inner].x)   # Breite für X
        eh = abs(lm[bot].y   - lm[top].y)     # Höhe für Y  ← war vorher ew!
        if ew < 0.001 or eh < 0.001:
            return 0.0, 0.0
        gx = (lm[iris_c].x - cx) / (ew / 2)
        gy = (lm[iris_c].y - cy) / (eh / 2)
        return gx, gy

    lx, ly = eye_gaze(LEFT_INNER,  LEFT_OUTER,  LEFT_TOP,  LEFT_BOT,  LEFT_IRIS_C)
    rx, ry = eye_gaze(RIGHT_INNER, RIGHT_OUTER, RIGHT_TOP, RIGHT_BOT, RIGHT_IRIS_C)

    return float((lx + rx) / 2), float((ly + ry) / 2)


# ── Arm ───────────────────────────────────────────────────────────────────────

class _SimArm:
    """Simulierter Arm für Tests ohne Hardware."""
    is_connected = True
    def connect(self):    return True
    def disconnect(self): pass
    def stop(self):       pass
    def move_polar(self, lr, ud, io):
        if lr or ud:
            print(f"[SIM]  LR={lr:+4d}  UD={ud:+4d}")


arm = _SimArm() if SIMULATE else SoloAssistController(ARM_IP, ARM_PORT)
if not arm.connect():
    print(f"FEHLER: Kein Arm bei {ARM_IP}:{ARM_PORT}")
    sys.exit(1)


# ── Arm-Befehlsschleife (20 Hz, Hintergrundthread) ───────────────────────────

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
    """Totzone + lineare Skalierung auf MAX_SPEED."""
    if abs(v) < DEADZONE:
        return 0
    sign = 1 if v > 0 else -1
    return int(sign * (abs(v) - DEADZONE) / (1.0 - DEADZONE) * MAX_SPEED)


# ── Hauptschleife ─────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(WEBCAM)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)

if not cap.isOpened():
    print(f"FEHLER: Webcam {WEBCAM} konnte nicht geöffnet werden")
    arm.disconnect()
    sys.exit(1)

WIN = "SOLOASSIST Eye Tracking"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, SCREEN_W, SCREEN_H)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Geglätteter Gaze-Wert (startet in der Mitte = kein Ausschlag)
sx, sy    = 0.0, 0.0
last_face = time.time()
paused    = False
fps_buf   = []

print(f"Gestartet. Bildschirm: {SCREEN_W}×{SCREEN_H}  |  ESC=Beenden  SPACE=Pause")

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    frame = cv2.flip(frame, 1)

    now = time.time()
    fps_buf = [t for t in fps_buf + [now] if now - t < 1.0]
    fps = len(fps_buf)

    # ── Gaze schätzen ─────────────────────────────────────────────────────
    raw_x, raw_y = get_gaze(frame)

    if raw_x is not None:
        last_face = now
        sx = SMOOTH * raw_x               + (1.0 - SMOOTH) * sx
        sy = SMOOTH * (raw_y + Y_OFFSET)  + (1.0 - SMOOTH) * sy

    face_ok = (now - last_face) < FACE_TIMEOUT

    # ── Gaze → Arm-Geschwindigkeit ────────────────────────────────────────
    # Roher Offset ≈ ±0.3 → mit GAZE_SCALE auf ±1 strecken
    gx =  np.clip(sx * GAZE_SCALE, -1.0, 1.0)
    gy = -np.clip(sy * GAZE_SCALE, -1.0, 1.0)   # Y invertiert: Iris oben = hoch

    lr = _to_speed(gx)
    ud = _to_speed(gy)

    if face_ok and not paused and raw_x is not None:
        with _lock:
            _cmd[:] = [lr, ud, 0]
    else:
        with _lock:
            _cmd[:] = [0, 0, 0]

    # ── Anzeige ───────────────────────────────────────────────────────────
    disp = cv2.resize(frame, (SCREEN_W, SCREEN_H), interpolation=cv2.INTER_LINEAR)

    # Cursor-Punkt (direkte Iris-Position, wie im Repo: screen = screen_w * iris_x)
    # Cursor-Position: gx/gy in [-1,+1] → Bildschirmpixel
    cx = int(np.clip((gx + 1) / 2, 0, 1) * SCREEN_W)
    cy = int(np.clip((-gy + 1) / 2, 0, 1) * SCREEN_H)   # gy ist schon invertiert
    in_dz   = abs(gx) < DEADZONE and abs(gy) < DEADZONE
    dot_col = (100, 100, 255) if in_dz else (0, 220, 50)   # blau=Totzone, grün=aktiv
    cv2.circle(disp, (cx, cy), 18, (0, 0, 0),   -1)
    cv2.circle(disp, (cx, cy), 14, dot_col,      -1)
    cv2.circle(disp, (cx, cy), 14, (255,255,255), 2)

    # HUD oben links
    status = "PAUSE" if paused else ("KEIN GESICHT" if not face_ok else
             f"LR={lr:+4d}  UD={ud:+4d}")
    status_col = (0, 80, 255) if (not face_ok or paused) else (255, 255, 255)
    cv2.rectangle(disp, (0, 0), (SCREEN_W, 55), (20, 20, 20), -1)
    cv2.putText(disp, f"FPS: {fps:2d}   {status}",
                (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_col, 2, cv2.LINE_AA)

    # Hinweis unten
    cv2.putText(disp, "ESC / Q : Beenden     SPACE : Pause",
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

# ── Aufräumen ─────────────────────────────────────────────────────────────────
_run[0] = False
arm.stop()
arm.disconnect()
cap.release()
cv2.destroyAllWindows()
print("Beendet.")
