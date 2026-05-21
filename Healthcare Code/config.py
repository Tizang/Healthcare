"""
Zentrale Konfiguration — alle Einstellungen hier anpassen.
"""

# ── Arm-Verbindung ─────────────────────────────────────────────────────────
ARM_IP        = "192.168.1.100"  # IP des Windows-PCs mit RemoteHost
ARM_PORT      = 5000             # TCP-Port des RemoteHost
ARM_SIMULATE  = True             # True = kein echter Arm (Demo-Modus)

# ── Webcam ─────────────────────────────────────────────────────────────────
WEBCAM_INDEX  = 0     # 0 = erste Kamera
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480

# ── Timing ─────────────────────────────────────────────────────────────────
SEND_INTERVAL   = 0.05   # Sekunden zwischen Arm-Befehlen (= 20 Hz)
FACE_TIMEOUT    = 2.0    # Stopp nach X Sekunden ohne Gesicht

# ── Bewegungsgeschwindigkeit (Einheiten aus VB Demo: 0–320) ────────────────
MAX_SPEED_LR  = 150   # links/rechts
MAX_SPEED_UD  = 150   # oben/unten
MAX_SPEED_IO  = 100   # herein/heraus (Tiefe)

# ── Totzone & Glättung ─────────────────────────────────────────────────────
GAZE_DEADZONE  = 0.08   # Gaze-Werte < diese Schwelle werden ignoriert
PITCH_DEADZONE = 4.0    # Kopfneigung < X Grad wird ignoriert
PITCH_MAX_DEG  = 20.0   # Volle IO-Geschwindigkeit ab X Grad Neigung

SMOOTH_GAZE_ALPHA  = 0.22   # EMA-Faktor Gaze (0=träge, 1=direkt)
SMOOTH_PITCH_ALPHA = 0.18   # EMA-Faktor Kopfneigung
