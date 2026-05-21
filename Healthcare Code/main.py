"""
SOLOASSIST II — Eye & Head Tracking Control
============================================
Steuert den chirurgischen Roboterarm über Blickrichtung und Kopfneigung
einer normalen Webcam.

Steuerung:
  Blick links/rechts      → Arm links/rechts
  Blick oben/unten        → Arm oben/unten
  Diagonal                → Diagonale Bewegung
  Kopf heben              → Arm fährt heraus (extend)
  Kopf senken             → Arm fährt hinein (retract)

Tastenkürzel:
  ESC / Q   → Sofortiger Stopp + Beenden
  SPACE     → Bewegung pausieren / fortsetzen
  C         → Kalibrierung starten
  R         → Kalibrierung zurücksetzen
  H         → Kopf-Neutral kalibrieren (jetzt gerade schauen!)

Konfiguration:
  ARM_IP   / ARM_PORT  — IP und Port des RemoteHost
  Setze ARM_SIMULATE = True um ohne physischen Arm zu testen
"""

import sys
import time
import logging
import threading
import argparse

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ARM_IP        = "192.168.1.100"   # ← IP-Adresse des RemoteHost-PCs anpassen
ARM_PORT      = 5000              # ← Port des RemoteHost anpassen
ARM_SIMULATE  = False             # True = kein echter Arm, nur Debug-Output
WEBCAM_INDEX  = 0                 # Webcam-Index (0 = erste Kamera)
SEND_INTERVAL = 0.05              # Sekunden zwischen Arm-Befehlen (20 Hz)
FACE_TIMEOUT  = 2.0               # Stopp nach X Sekunden ohne Gesicht

# ---------------------------------------------------------------------------
# Imports (mit hilfreichen Fehlermeldungen falls Pakete fehlen)
# ---------------------------------------------------------------------------
try:
    import cv2
except ImportError:
    sys.exit("FEHLER: opencv-python nicht installiert.\n"
             "  pip install opencv-python mediapipe numpy")

try:
    import mediapipe as mp
except ImportError:
    sys.exit("FEHLER: mediapipe nicht installiert.\n"
             "  pip install mediapipe")

from eye_tracking.gaze     import GazeEstimator
from eye_tracking.head_pose import HeadPoseEstimator
from eye_tracking.smoothing import Vec2Smoother, ExponentialSmoother
from controller.mapper      import GazeToArmMapper, MapperConfig
from controller.arm_controller import SoloAssistController
from calibration.calibration   import run_calibration, CalibrationData


# ---------------------------------------------------------------------------
# Simulated arm (for testing without hardware)
# ---------------------------------------------------------------------------
class SimulatedArm:
    """Drops-in for SoloAssistController when ARM_SIMULATE=True."""

    is_connected = True

    def connect(self):          return True
    def disconnect(self):       pass
    def stop(self):             log.info("[SIM] STOP")

    def move_polar(self, lr: int, ud: int, io: int):
        if lr != 0 or ud != 0 or io != 0:
            log.debug("[SIM] move_polar  LR=%+4d  UD=%+4d  IO=%+4d", lr, ud, io)


# ---------------------------------------------------------------------------
# Debug overlay helpers
# ---------------------------------------------------------------------------
FONT  = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 220,  50)
RED   = (0,  50, 220)
CYAN  = (220, 220, 0)
WHITE = (255, 255, 255)
GRAY  = (130, 130, 130)


def draw_overlay(
    frame: np.ndarray,
    gaze_x, gaze_y,
    pitch,
    speed_lr, speed_ud, speed_io,
    direction: str,
    fps: float,
    paused: bool,
    face_detected: bool,
    deadzone: float = 0.18,
):
    h, w = frame.shape[:2]

    # Semi-transparent dark bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 105), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    def txt(msg, x, y, color=WHITE, scale=0.55, thick=1):
        cv2.putText(frame, msg, (x, y), FONT, scale, color, thick, cv2.LINE_AA)

    txt(f"FPS: {fps:5.1f}", 10, 22, CYAN, 0.6, 2)

    if not face_detected:
        txt("NO FACE DETECTED — ARM STOPPED", 10, 55, RED, 0.7, 2)
    elif paused:
        txt("PAUSED  (SPACE to resume)", 10, 55, (0, 180, 255), 0.7, 2)
    else:
        gx_str = f"{gaze_x:+.2f}" if gaze_x is not None else "---"
        gy_str = f"{gaze_y:+.2f}" if gaze_y is not None else "---"
        p_str  = f"{pitch:+.1f}°"  if pitch  is not None else "---"
        txt(f"Gaze  X:{gx_str}  Y:{gy_str}   Pitch:{p_str}", 10, 45, WHITE, 0.55)
        txt(f"Arm   LR:{speed_lr:+4d}  UD:{speed_ud:+4d}  IO:{speed_io:+4d}", 10, 70, WHITE, 0.55)
        txt(f"Direction: {direction}", 10, 95, GREEN, 0.6, 2)

    # Gaze crosshair + deadzone visualisation (bottom-right)
    cx, cy = w - 85, h - 85
    r = 65
    cv2.circle(frame, (cx, cy), r, GRAY, 1)
    cv2.line(frame, (cx - r, cy), (cx + r, cy), GRAY, 1)
    cv2.line(frame, (cx, cy - r), (cx, cy + r), GRAY, 1)

    # Deadzone circle (filled, semi-transparent red)
    dz_r = int(deadzone * r)
    dz_overlay = frame.copy()
    cv2.circle(dz_overlay, (cx, cy), dz_r, (0, 50, 180), -1)
    cv2.addWeighted(dz_overlay, 0.25, frame, 0.75, 0, frame)
    cv2.circle(frame, (cx, cy), dz_r, (0, 80, 255), 1)
    txt(f"DZ {int(deadzone*100)}%", cx - dz_r - 28, cy - dz_r + 5, (0, 120, 255), 0.38)

    if gaze_x is not None and gaze_y is not None:
        gx_px = int(cx + np.clip(gaze_x, -1, 1) * r)
        gy_px = int(cy + np.clip(gaze_y, -1, 1) * r)
        # Dot colour: red inside deadzone, green outside
        in_dz = (gaze_x**2 + gaze_y**2) < deadzone**2
        dot_col = (0, 80, 255) if in_dz else GREEN
        cv2.circle(frame, (gx_px, gy_px), 8, dot_col, -1)
        cv2.circle(frame, (gx_px, gy_px), 8, WHITE, 1)

    # ESC hint
    txt("ESC/Q: quit   SPACE: pause   C: calibrate   H: head-neutral   +/-: deadzone",
        8, h - 10, GRAY, 0.40)

    return frame


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SOLOASSIST Eye Tracking Controller")
    parser.add_argument("--ip",       default=ARM_IP,   help="RemoteHost IP")
    parser.add_argument("--port",     default=ARM_PORT,  type=int)
    parser.add_argument("--simulate", action="store_true", default=ARM_SIMULATE)
    parser.add_argument("--cam",      default=WEBCAM_INDEX, type=int)
    parser.add_argument("--calibrate", action="store_true", help="Run calibration on start")
    args = parser.parse_args()

    # ---- Arm connection ----
    if args.simulate:
        arm = SimulatedArm()
        log.info("SIMULATION MODE — no real arm commands sent")
    else:
        arm = SoloAssistController(args.ip, args.port)
        if not arm.connect():
            log.error("Cannot connect to arm at %s:%d — exiting", args.ip, args.port)
            sys.exit(1)

    # ---- Webcam ----
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    if not cap.isOpened():
        log.error("Cannot open webcam %d", args.cam)
        arm.disconnect()
        sys.exit(1)

    # ---- Tracking & mapping objects ----
    gaze_est   = GazeEstimator(min_detection_confidence=0.6)
    head_est   = HeadPoseEstimator()
    gaze_smoother  = Vec2Smoother(alpha=0.22)
    pitch_smoother = ExponentialSmoother(alpha=0.18)

    # ---- Load calibration (affine transform) ----
    cal_data = CalibrationData.load()
    mapper = GazeToArmMapper(MapperConfig(), calibration=cal_data)

    # ---- Optional startup calibration ----
    if args.calibrate:
        log.info("Starting calibration…")
        cal_data = run_calibration(gaze_est, cap)
        mapper._cal = cal_data

    # ---- State ----
    paused          = False
    last_face_time  = time.time()
    last_send_time  = time.time()
    fps_times       = []
    speed_lr = speed_ud = speed_io = 0
    direction = "CENTRE"

    # ---- Arm send thread ----
    _send_lock   = threading.Lock()
    _latest_cmd  = [0, 0, 0]
    _arm_running = [True]

    def arm_send_loop():
        while _arm_running[0]:
            time.sleep(SEND_INTERVAL)
            with _send_lock:
                lr, ud, io = _latest_cmd
            if arm.is_connected:
                arm.move_polar(lr, ud, io)

    arm_thread = threading.Thread(target=arm_send_loop, daemon=True)
    arm_thread.start()

    log.info("Eye tracking controller started. Press ESC or Q to quit.")

    # ---- Main loop ----
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Webcam frame missed")
                continue

            frame = cv2.flip(frame, 1)   # mirror for natural feel

            # FPS
            now = time.time()
            fps_times.append(now)
            fps_times = [t for t in fps_times if now - t < 1.0]
            fps = len(fps_times)

            # ---- Gaze estimation ----
            raw_gx, raw_gy = gaze_est.estimate(frame)
            face_detected = raw_gx is not None

            if face_detected:
                last_face_time = now
                gaze_x, gaze_y = gaze_smoother.smooth(raw_gx, raw_gy)
            else:
                gaze_x = gaze_y = None

            # ---- Head pose estimation ----
            pitch = None
            if gaze_est._transform_matrix is not None:
                p, _, _ = head_est.estimate(gaze_est._transform_matrix)
                if p is not None:
                    pitch = pitch_smoother.smooth(p)

            # ---- Safety: face timeout ----
            face_lost = (now - last_face_time) > FACE_TIMEOUT
            if face_lost and not face_detected:
                with _send_lock:
                    _latest_cmd[:] = [0, 0, 0]
                arm.stop()
                gaze_smoother.reset()
                pitch_smoother.reset()

            # ---- Mapping ----
            if face_detected and not paused and not face_lost:
                speed_lr, speed_ud, speed_io = mapper.map(gaze_x, gaze_y, pitch)
                direction = mapper.get_direction_label(speed_lr, speed_ud, speed_io)
                with _send_lock:
                    _latest_cmd[:] = [speed_lr, speed_ud, speed_io]
            else:
                speed_lr = speed_ud = speed_io = 0
                direction = "PAUSED" if paused else "NO FACE"
                with _send_lock:
                    _latest_cmd[:] = [0, 0, 0]

            # ---- Debug drawing ----
            gaze_est.draw_debug(frame)
            frame = draw_overlay(
                frame, gaze_x, gaze_y, pitch,
                speed_lr, speed_ud, speed_io, direction,
                fps, paused, face_detected,
                deadzone=mapper.config.gaze_deadzone,
            )

            cv2.imshow("SOLOASSIST Eye Tracking", frame)

            # ---- Key handling ----
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):        # ESC / Q → quit
                break
            elif key == ord(" "):                       # SPACE → pause
                paused = not paused
                if paused:
                    arm.stop()
                    with _send_lock:
                        _latest_cmd[:] = [0, 0, 0]
                log.info("Tracking %s", "PAUSED" if paused else "RESUMED")
            elif key in (ord("c"), ord("C")):           # C → calibrate
                log.info("Starting calibration…")
                arm.stop()
                with _send_lock:
                    _latest_cmd[:] = [0, 0, 0]
                cal_data = run_calibration(gaze_est, cap)
                mapper._cal = cal_data
            elif key in (ord("r"), ord("R")):           # R → reset calibration
                mapper._cal = CalibrationData()         # Identität
                head_est.neutral_pitch = 0.0
                gaze_smoother.reset()
                pitch_smoother.reset()
                log.info("Calibration reset")
            elif key in (ord("h"), ord("H")):           # H → head neutral
                if gaze_est._transform_matrix is not None:
                    head_est.calibrate_neutral(gaze_est._transform_matrix)
                    log.info("Head neutral set (pitch offset = %.1f°)", head_est.neutral_pitch)
            elif key in (ord("+"), ord("="), ord(".")):  # + → deadzone größer
                mapper.config.gaze_deadzone = min(0.60, mapper.config.gaze_deadzone + 0.02)
                log.info("Deadzone: %.0f%%", mapper.config.gaze_deadzone * 100)
            elif key in (ord("-"), ord(","), ord("_")):  # - → deadzone kleiner
                mapper.config.gaze_deadzone = max(0.02, mapper.config.gaze_deadzone - 0.02)
                log.info("Deadzone: %.0f%%", mapper.config.gaze_deadzone * 100)

    except KeyboardInterrupt:
        log.info("Interrupted by user")

    finally:
        log.info("Shutting down…")
        _arm_running[0] = False
        arm.stop()
        arm.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        log.info("Done.")


if __name__ == "__main__":
    main()
