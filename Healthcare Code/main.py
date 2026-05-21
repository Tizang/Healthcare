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
import subprocess

import numpy as np

def _get_screen_size():
    # macOS: ask the system directly via osascript
    try:
        out = subprocess.check_output(
            ["osascript", "-e",
             "tell application \"Finder\" to get bounds of window of desktop"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        parts = [int(x.strip()) for x in out.split(",")]
        return parts[2], parts[3]   # "0, 0, W, H"
    except Exception:
        pass
    # Fallback: parse system_profiler
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            text=True, stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            if "Resolution" in line and "x" in line:
                nums = [int(s) for s in line.split() if s.isdigit()]
                if len(nums) >= 2:
                    return nums[0], nums[1]
    except Exception:
        pass
    return 1440, 900   # safe default

SCREEN_W, SCREEN_H = _get_screen_size()

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

from eye_tracking.gaze          import GazeEstimator
from eye_tracking.head_pose     import HeadPoseEstimator
from eye_tracking.smoothing     import AdaptiveVec2Smoother, ExponentialSmoother
from eye_tracking.apple_pointer import ApplePointerGaze
from controller.mapper          import GazeToArmMapper, MapperConfig
from controller.arm_controller  import SoloAssistController
from calibration.calibration    import run_calibration, CalibrationData


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
    gaze_y_offset: float = -0.45,
    gaze_scale: float = 2.8,
):
    h, w = frame.shape[:2]
    sf = w / 640  # scale factor for UI elements relative to 640px base

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, int(110 * sf)), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    def txt(msg, x, y, color=WHITE, scale=0.55, thick=1):
        cv2.putText(frame, msg, (x, y), FONT,
                    scale * sf, color, max(1, int(thick * sf)), cv2.LINE_AA)

    txt(f"FPS: {fps:5.1f}", 10, int(25 * sf), CYAN, 0.7, 2)

    if not face_detected:
        txt("NO FACE DETECTED — ARM STOPPED", 10, int(60 * sf), RED, 0.8, 2)
    elif paused:
        txt("PAUSED  (SPACE to resume)", 10, int(60 * sf), (0, 180, 255), 0.8, 2)
    else:
        gx_str = f"{gaze_x:+.2f}" if gaze_x is not None else "---"
        gy_str = f"{gaze_y:+.2f}" if gaze_y is not None else "---"
        p_str  = f"{pitch:+.1f}°"  if pitch  is not None else "---"
        txt(f"Gaze  X:{gx_str}  Y:{gy_str}   Pitch:{p_str}", 10, int(50 * sf), WHITE, 0.6)
        txt(f"Arm   LR:{speed_lr:+4d}  UD:{speed_ud:+4d}  IO:{speed_io:+4d}", 10, int(75 * sf), WHITE, 0.6)
        txt(f"Direction: {direction}", 10, int(100 * sf), GREEN, 0.7, 2)

    # Deadzone mini-map (bottom-right corner)
    r   = int(70 * sf)
    cx  = w - r - int(20 * sf)
    cy  = h - r - int(20 * sf)
    cv2.circle(frame, (cx, cy), r, GRAY, 1)
    cv2.line(frame, (cx - r, cy), (cx + r, cy), GRAY, 1)
    cv2.line(frame, (cx, cy - r), (cx, cy + r), GRAY, 1)
    dz_r = int(deadzone * r)
    dz_ov = frame.copy()
    cv2.circle(dz_ov, (cx, cy), dz_r, (0, 50, 180), -1)
    cv2.addWeighted(dz_ov, 0.25, frame, 0.75, 0, frame)
    cv2.circle(frame, (cx, cy), dz_r, (0, 80, 255), 1)
    txt(f"DZ {int(deadzone*100)}%", cx - dz_r - int(30*sf), cy - dz_r + int(5*sf),
        (0, 120, 255), 0.42)

    if gaze_x is not None and gaze_y is not None:
        # Corrected gaze (Y offset so neutral gaze = 0)
        corrected_y = gaze_y - gaze_y_offset
        in_dz = (gaze_x**2 + corrected_y**2) < deadzone**2

        # Mini-map dot (raw gaze, no scale)
        mmx = int(cx + np.clip(gaze_x,      -1, 1) * r)
        mmy = int(cy + np.clip(corrected_y, -1, 1) * r)
        mm_col = (0, 80, 255) if in_dz else GREEN
        cv2.circle(frame, (mmx, mmy), int(7*sf), mm_col, -1)
        cv2.circle(frame, (mmx, mmy), int(7*sf), WHITE, 1)

        # ── Hauptcursor: skaliert auf vollen Screen ───────────────────────
        # gaze_scale streckt den Gaze-Bereich auf den gesamten Frame:
        # raw gaze ±(1/gaze_scale) entspricht dann einer Bildschirmecke
        sx = np.clip(gaze_x      * gaze_scale, -1, 1)
        sy = np.clip(corrected_y * gaze_scale, -1, 1)

        dot_x = int((sx + 1) / 2 * w)
        dot_y = int((sy + 1) / 2 * h)
        dot_x = max(14, min(w - 14, dot_x))
        dot_y = max(14, min(h - 14, dot_y))

        dot_r = int(14 * sf)
        dot_col = (0, 80, 255) if in_dz else GREEN
        cv2.circle(frame, (dot_x, dot_y), dot_r + 2, (0, 0, 0), -1)   # shadow
        cv2.circle(frame, (dot_x, dot_y), dot_r,     dot_col,   -1)
        cv2.circle(frame, (dot_x, dot_y), dot_r,     WHITE,      2)

    txt("ESC/Q: quit   SPACE: pause   C: calibrate   H: head-neutral   +/-: deadzone",
        8, h - int(12 * sf), GRAY, 0.42)

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
    parser.add_argument("--pointer-mode", action="store_true",
                        help="Use Apple Head Pointer / Eye Tracking cursor instead of MediaPipe gaze")
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
    apple_gaze = None
    if args.pointer_mode:
        if not ApplePointerGaze.available():
            log.error("--pointer-mode requires pyobjc-framework-Quartz: pip install pyobjc-framework-Quartz")
            sys.exit(1)
        apple_gaze = ApplePointerGaze(screen_w=SCREEN_W, screen_h=SCREEN_H)
        log.info("POINTER MODE — using Apple Head Pointer / Eye Tracking cursor")

    gaze_est   = GazeEstimator(min_detection_confidence=0.5)
    head_est   = HeadPoseEstimator()
    # Adaptive: light smoothing during fast saccades, heavy during fixation
    gaze_smoother  = AdaptiveVec2Smoother(alpha_still=0.15, alpha_moving=0.65,
                                          velocity_threshold=0.03)
    pitch_smoother = ExponentialSmoother(alpha=0.45)

    # ---- Load calibration (affine transform) ----
    cal_data = CalibrationData.load()
    cfg = MapperConfig()
    if args.pointer_mode:
        cfg.gaze_y_offset = 0.0   # Apple cursor has no iris bias
    mapper = GazeToArmMapper(cfg, calibration=cal_data)

    # ---- Optional startup calibration ----
    if args.calibrate:
        log.info("Starting calibration…")
        cal_data = run_calibration(gaze_est, cap)
        mapper._cal = cal_data

    # ---- Fenster fullscreen (auf echte Screenauflösung skaliert) ----
    WIN = "SOLOASSIST Eye Tracking"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, SCREEN_W, SCREEN_H)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    log.info("Screen: %dx%d", SCREEN_W, SCREEN_H)

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
            if apple_gaze is not None:
                # Apple Head Pointer / Eye Tracking: read system cursor directly
                raw_gx, raw_gy = apple_gaze.get()
                face_detected = True
                last_face_time = now
                gaze_x, gaze_y = gaze_smoother.smooth(raw_gx, raw_gy)
                # Still run MediaPipe for head pitch (IO control), but don't need gaze
                gaze_est.estimate(frame)
            else:
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
            display = cv2.resize(frame, (SCREEN_W, SCREEN_H),
                                 interpolation=cv2.INTER_LINEAR)

            # Show calibrated gaze in the overlay so the cursor dot reflects
            # WHERE THE USER IS LOOKING (not the raw iris position).
            # Without calibration: show raw gaze stretched to reach screen edges.
            is_calibrated = (
                mapper._cal is not None
                and not np.allclose(mapper._cal.poly_coeffs, CalibrationData().poly_coeffs)
            )
            if is_calibrated and gaze_x is not None:
                disp_gx, disp_gy = mapper._cal.apply(gaze_x, gaze_y)
                gaze_scale = 1.0
            elif apple_gaze:
                disp_gx, disp_gy = gaze_x, gaze_y
                gaze_scale = 1.0
            else:
                disp_gx, disp_gy = gaze_x, gaze_y
                gaze_scale = 2.2   # stretch raw iris range to fill screen

            display = draw_overlay(
                display, disp_gx, disp_gy, pitch,
                speed_lr, speed_ud, speed_io, direction,
                fps, paused, face_detected,
                deadzone=mapper.config.gaze_deadzone,
                gaze_y_offset=mapper.config.gaze_y_offset,
                gaze_scale=gaze_scale,
            )

            cv2.imshow(WIN, display)

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
