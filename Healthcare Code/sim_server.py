"""
SOLOASSIST II — Arm Simulator Server
=====================================
Simuliert den RemoteHost TCP-Server auf dem lokalen PC.
Empfängt echte MovePolar-Befehle vom Eye-Tracking Controller
und zeigt den Arm live in einem OpenCV-Fenster.

Starten:  python3 sim_server.py
          (dann in einem zweiten Terminal: python3 main.py --ip 127.0.0.1 --port 5000)
"""

import socket
import struct
import threading
import time
import math
import numpy as np
import cv2

HOST = "127.0.0.1"
PORT = 5000

# Client command codes (from RH_CLASSLIB.vb)
CMD_GET_HP      = 0x03
CMD_MOVE_POLAR  = 0x07
CMD_STOP_ALL    = 0xFF
CMD_QUIT        = 0xFE

# Server response codes
RESP_HP_POS     = 0x43

# Simulated arm limits (mm)
LIMIT_X = 150
LIMIT_Y = 150
LIMIT_Z = 150   # z = extension depth

# How much position changes per speed-unit per second
SPEED_SCALE = 0.15


class SimArm:
    """Integrates velocity commands into a simulated arm position."""

    def __init__(self):
        self.x:  float = 0.0   # left/right
        self.y:  float = 0.0   # up/down
        self.z:  float = 0.0   # in/out (depth)
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.vz: float = 0.0
        self._lock = threading.Lock()
        self.last_cmd_time = time.time()
        self.total_commands = 0
        self.last_direction = "IDLE"

    def set_velocity(self, lr: int, ud: int, io: int):
        with self._lock:
            self.vx = lr
            self.vy = ud
            self.vz = io
            self.last_cmd_time = time.time()
            self.total_commands += 1
            parts = []
            if abs(ud) > 10: parts.append("UP" if ud > 0 else "DOWN")
            if abs(lr) > 10: parts.append("LEFT" if lr < 0 else "RIGHT")
            if abs(io) > 10: parts.append("EXTEND" if io > 0 else "RETRACT")
            self.last_direction = " + ".join(parts) if parts else "CENTRE"

    def stop(self):
        with self._lock:
            self.vx = self.vy = self.vz = 0.0
            self.last_direction = "STOP"

    def tick(self, dt: float):
        with self._lock:
            self.x = max(-LIMIT_X, min(LIMIT_X, self.x + self.vx * dt * SPEED_SCALE))
            self.y = max(-LIMIT_Y, min(LIMIT_Y, self.y + self.vy * dt * SPEED_SCALE))
            self.z = max(-LIMIT_Z, min(LIMIT_Z, self.z + self.vz * dt * SPEED_SCALE))

    def get_pos(self):
        with self._lock:
            return self.x, self.y, self.z, self.vx, self.vy, self.vz


arm = SimArm()


# ── TCP Server ─────────────────────────────────────────────────────────────

def handle_client(conn: socket.socket, addr):
    print(f"[Server] Verbunden: {addr}")
    packet_counter = 0
    buf = bytearray()

    def send_response(command: int, data: bytes = b""):
        nonlocal packet_counter
        packet_counter = (packet_counter % 255) + 1
        payload_len = 4 + len(data)
        msg = struct.pack("<IBBh", payload_len, 1, packet_counter, command) + data
        try:
            conn.sendall(msg)
        except Exception:
            pass

    try:
        while True:
            chunk = conn.recv(512)
            if not chunk:
                break
            buf.extend(chunk)

            while len(buf) >= 4:
                payload_len = struct.unpack_from("<I", buf, 0)[0]
                total = 4 + payload_len
                if len(buf) < total:
                    break

                payload = buf[4:total]
                buf = buf[total:]

                if len(payload) < 4:
                    continue

                cmd = struct.unpack_from("<h", payload, 2)[0]
                data = payload[4:]

                if cmd == CMD_MOVE_POLAR and len(data) >= 6:
                    lr, ud, io = struct.unpack_from("<hhh", data)
                    arm.set_velocity(lr, ud, io)

                elif cmd == CMD_STOP_ALL:
                    arm.stop()

                elif cmd == CMD_GET_HP:
                    x, y, z, *_ = arm.get_pos()
                    resp = struct.pack("<hhh", int(x), int(y), int(z))
                    send_response(RESP_HP_POS, resp)

                elif cmd == CMD_QUIT:
                    break

    except Exception as e:
        print(f"[Server] Verbindung getrennt: {e}")
    finally:
        arm.stop()
        conn.close()
        print("[Server] Getrennt")


def tcp_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"[Server] Warte auf Verbindung auf {HOST}:{PORT} ...")
    while True:
        conn, addr = srv.accept()
        t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
        t.start()


# ── Visualisierung ─────────────────────────────────────────────────────────

FONT  = cv2.FONT_HERSHEY_SIMPLEX
W, H  = 800, 600


def draw_scene(x, y, z, vx, vy, vz, direction, n_cmds):
    """Render a top+side view of the simulated arm."""
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 35)

    def txt(msg, px, py, color=(200, 200, 200), scale=0.55, thick=1):
        cv2.putText(canvas, msg, (px, py), FONT, scale, color, thick, cv2.LINE_AA)

    # ── Title bar ──
    cv2.rectangle(canvas, (0, 0), (W, 40), (40, 40, 60), -1)
    txt("SOLOASSIST II  —  Arm Simulator", 10, 27, (100, 200, 255), 0.65, 2)
    txt(f"Befehle empfangen: {n_cmds}", W - 240, 27, (150, 150, 150))

    # ── Top view (left half) ──
    tv_cx, tv_cy, tv_r = 210, 310, 150
    cv2.circle(canvas, (tv_cx, tv_cy), tv_r, (50, 50, 70), 1)
    cv2.circle(canvas, (tv_cx, tv_cy), 5, (80, 80, 100), -1)
    txt("DRAUFSICHT  (L/R  ·  rein/raus)", tv_cx - 130, 60, (120, 120, 140))

    # Arm line: z = depth (into screen in top view = forward)
    # x = left/right,  z = in/out
    arm_px = tv_cx + int(x / LIMIT_X * tv_r)
    arm_pz = tv_cy - int(z / LIMIT_Z * tv_r)   # z+ = extend = forward (up on screen)
    cv2.line(canvas, (tv_cx, tv_cy), (arm_px, arm_pz), (0, 160, 255), 3)
    cv2.circle(canvas, (arm_px, arm_pz), 10, (0, 220, 255), -1)

    # Axis labels
    txt("L", tv_cx - tv_r - 18, tv_cy + 5,  (100, 100, 150))
    txt("R", tv_cx + tv_r + 5,  tv_cy + 5,  (100, 100, 150))
    txt("rein",  tv_cx - 18, tv_cy + tv_r + 18, (100, 100, 150), 0.45)
    txt("raus",  tv_cx - 18, tv_cy - tv_r - 8,  (100, 100, 150), 0.45)

    # ── Side view (right half) ──
    sv_cx, sv_cy, sv_r = 590, 310, 150
    cv2.circle(canvas, (sv_cx, sv_cy), sv_r, (50, 50, 70), 1)
    cv2.circle(canvas, (sv_cx, sv_cy), 5, (80, 80, 100), -1)
    txt("SEITENANSICHT  (oben/unten)", sv_cx - 120, 60, (120, 120, 140))

    # x = left/right, y = up/down
    arm_sx = sv_cx + int(x / LIMIT_X * sv_r)
    arm_sy = sv_cy - int(y / LIMIT_Y * sv_r)   # y+ = up
    cv2.line(canvas, (sv_cx, sv_cy), (arm_sx, arm_sy), (0, 200, 100), 3)
    cv2.circle(canvas, (arm_sx, arm_sy), 10, (0, 255, 120), -1)

    txt("L",     sv_cx - sv_r - 18, sv_cy + 5, (100, 100, 150))
    txt("R",     sv_cx + sv_r + 5,  sv_cy + 5, (100, 100, 150))
    txt("oben",  sv_cx - 20, sv_cy - sv_r - 8, (100, 100, 150), 0.45)
    txt("unten", sv_cx - 22, sv_cy + sv_r + 18,(100, 100, 150), 0.45)

    # ── Position readout ──
    cv2.rectangle(canvas, (0, H - 120), (W, H), (35, 35, 50), -1)
    txt(f"Position   X:{x:+6.1f}mm   Y:{y:+6.1f}mm   Z:{z:+6.1f}mm",
        15, H - 88, (200, 200, 200), 0.6, 1)
    txt(f"Geschwindigkeit  LR:{vx:+5.0f}   UD:{vy:+5.0f}   IO:{vz:+5.0f}",
        15, H - 58, (160, 160, 180), 0.55)

    dir_color = (0, 255, 120) if direction not in ("IDLE", "STOP", "CENTRE") else (100, 100, 130)
    txt(f"Richtung:  {direction}", 15, H - 22, dir_color, 0.72, 2)

    txt("Q / ESC = Beenden", W - 185, H - 10, (80, 80, 100), 0.45)

    return canvas


def run_visualizer():
    last_tick = time.time()
    while True:
        now = time.time()
        dt = now - last_tick
        last_tick = now

        arm.tick(dt)
        x, y, z, vx, vy, vz = arm.get_pos()

        frame = draw_scene(x, y, z, vx, vy, vz, arm.last_direction, arm.total_commands)
        cv2.imshow("SOLOASSIST Arm Simulator", frame)

        key = cv2.waitKey(33) & 0xFF   # ~30 FPS
        if key in (27, ord("q"), ord("Q")):
            break

    cv2.destroyAllWindows()


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    server_thread = threading.Thread(target=tcp_server, daemon=True)
    server_thread.start()

    print("[Server] Visualisierung gestartet.")
    print(f"[Server] Jetzt in einem zweiten Terminal starten:")
    print(f"         /Users/konrad/Desktop/Projekt/venv/bin/python3 main.py --ip 127.0.0.1 --port {PORT}")
    print()

    run_visualizer()
    print("[Server] Beendet.")
