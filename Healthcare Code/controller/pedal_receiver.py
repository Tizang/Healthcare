"""
ESP32 Pedal Receiver — verbindet sich mit dem ESP32 StateBeacon (Port 5566).

Protokoll: Tab-getrennte ASCII-Zeilen mit 100 Hz
  Format:  B0 0\tB1 0\tB2 0\tT 12345\n
  B0 = Zoom In (Arm rein)
  B1 = Zoom Out (Arm raus)
  B2 = Start / Stop (Pause toggle)
"""

import socket
import threading
import time


class PedalReceiver:
    """
    Empfängt Pedal-Zustände vom ESP32 über TCP.

    Verwendung:
        pedals = PedalReceiver("192.168.1.50", 5566)
        pedals.connect()

        # im Hauptloop:
        if pedals.pressed("B2"):   # einmaliger Flanken-Trigger
            paused = not paused
        zoom = pedals.held("B0") - pedals.held("B1")   # +1, 0, -1
    """

    def __init__(self, ip: str, port: int = 5566, timeout: float = 3.0):
        self.ip      = ip
        self.port    = port
        self.timeout = timeout

        self._state:    dict[str, int] = {"B0": 0, "B1": 0, "B2": 0}
        self._prev:     dict[str, int] = {"B0": 0, "B1": 0, "B2": 0}
        self._triggers: dict[str, bool] = {"B0": False, "B1": False, "B2": False}
        self._lock    = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self.connected = False

    def connect(self) -> bool:
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(self.timeout)
            self._sock.connect((self.ip, self.port))
            self._sock.settimeout(1.0)
            self._running = True
            self.connected = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            print(f"[Pedal] Verbunden mit ESP32 {self.ip}:{self.port}")
            return True
        except Exception as e:
            print(f"[Pedal] Verbindung fehlgeschlagen ({e}) — Pedale deaktiviert")
            self.connected = False
            return False

    def disconnect(self):
        self._running = False
        try:
            self._sock.close()
        except Exception:
            pass

    def held(self, button: str) -> bool:
        """True solange Taste gehalten wird."""
        with self._lock:
            return bool(self._state.get(button, 0))

    def pressed(self, button: str) -> bool:
        """True genau einmal bei steigender Flanke (Tastendruck)."""
        with self._lock:
            triggered = self._triggers.get(button, False)
            self._triggers[button] = False   # einmalig konsumieren
            return triggered

    # ── Hintergrund-Thread ────────────────────────────────────────────────────

    def _loop(self):
        buf = ""
        while self._running:
            try:
                chunk = self._sock.recv(256).decode("ascii", errors="ignore")
                if not chunk:
                    break
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    self._parse(line.strip())
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[Pedal] Verbindung unterbrochen ({e})")
                break
        self.connected = False

    def _parse(self, line: str):
        """Parst eine Zeile wie 'B0 0\tB1 1\tB2 0\tT 12345'."""
        new: dict[str, int] = {}
        for field in line.split("\t"):
            parts = field.strip().split()
            if len(parts) == 2 and parts[0].startswith("B"):
                try:
                    new[parts[0]] = int(parts[1])
                except ValueError:
                    pass

        with self._lock:
            for btn, val in new.items():
                prev = self._state.get(btn, 0)
                self._state[btn] = val
                if val == 1 and prev == 0:          # steigende Flanke
                    self._triggers[btn] = True
