"""
Tobii Stream Engine — direkter DLL-Zugriff via ctypes.
Funktioniert mit Tobii 4C + jeder Python-Version.
Voraussetzung: Tobii Experience Software installiert (liefert die DLL).
"""

import ctypes
import threading
import time
import os
import numpy as np

from gaze.filters import KalmanFilter1D

_DLL_NAME = "tobii_stream_engine.dll"

_DLL_CANDIDATES = [
    r"C:\Program Files\Tobii\Tobii Stream Engine\tobii_stream_engine.dll",
    r"C:\Program Files (x86)\Tobii\Tobii Stream Engine\tobii_stream_engine.dll",
    r"C:\Program Files\Tobii\Tobii Streams\tobii_stream_engine.dll",
    r"C:\Program Files\Tobii\tobii_stream_engine.dll",
    r"C:\Program Files (x86)\Tobii\tobii_stream_engine.dll",
    r"C:\Program Files\Mill Mouse\tobii_stream_engine.dll",
    r"C:\Program Files (x86)\Mill Mouse\tobii_stream_engine.dll",
]


def _find_dll() -> ctypes.CDLL:
    # 1. Direkt aus Windows DLL-Suchpfad (PATH, System32, etc.)
    try:
        dll = ctypes.CDLL(_DLL_NAME)
        print(f"[Tobii] DLL aus PATH geladen")
        return dll
    except Exception:
        pass

    # 2. Bekannte Pfade
    for path in _DLL_CANDIDATES:
        if os.path.exists(path):
            print(f"[Tobii] DLL gefunden: {path}")
            return ctypes.CDLL(path)

    # 3. Geladene Prozess-Module durchsuchen (findet Mill Mouse DLL)
    try:
        import subprocess
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-Process | ForEach-Object { try { $_.Modules } catch {} } | "
             "Where-Object { $_.ModuleName -like '*tobii_stream*' } | "
             "Select-Object -ExpandProperty FileName | Sort-Object -Unique"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line and os.path.exists(line):
                print(f"[Tobii] DLL aus Prozess gefunden: {line}")
                return ctypes.CDLL(line)
    except Exception:
        pass

    # 4. Rekursiv in Program Files und AppData suchen
    search_roots = [
        r"C:\Program Files",
        r"C:\Program Files (x86)",
        os.path.expanduser(r"~\AppData\Local"),
    ]
    for base in search_roots:
        for root, _dirs, files in os.walk(base):
            if _DLL_NAME in files and "tobii" in root.lower():
                path = os.path.join(root, _DLL_NAME)
                print(f"[Tobii] DLL gefunden: {path}")
                return ctypes.CDLL(path)

    # 5. Windows Registry
    try:
        import winreg
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            for subkey in (r"SOFTWARE\Tobii", r"SOFTWARE\WOW6432Node\Tobii"):
                try:
                    key = winreg.OpenKey(hive, subkey)
                    install_dir, _ = winreg.QueryValueEx(key, "InstallDir")
                    candidate = os.path.join(install_dir, _DLL_NAME)
                    if os.path.exists(candidate):
                        return ctypes.CDLL(candidate)
                except Exception:
                    pass
    except ImportError:
        pass

    raise FileNotFoundError(
        f"{_DLL_NAME} nicht gefunden.\n"
        "Bitte in PowerShell ausführen:\n"
        "  Get-Process | ForEach-Object { try { $_.Modules } catch {} } | "
        "Where-Object { $_.ModuleName -like '*tobii*' } | "
        "Select-Object FileName | Sort-Object -Unique"
    )


# ── ctypes Strukturen ─────────────────────────────────────────────────────────

class _GazePoint(ctypes.Structure):
    _fields_ = [
        ("timestamp_us", ctypes.c_int64),
        ("validity",     ctypes.c_int),       # 0=ungültig, 1=gültig
        ("position_xy",  ctypes.c_float * 2), # normalisiert [0,1]
    ]


_GazeCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(_GazePoint), ctypes.c_void_p)
_UrlReceiver  = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)


class TobiiStreamEngineEstimator:
    """
    Liest Gaze-Daten direkt vom Tobii 4C über die Stream Engine DLL.
    Liefert (gx, gy) in [-1, +1]:  links=-1  rechts=+1  unten=-1  oben=+1
    """

    def __init__(self):
        self._dll     = _find_dll()
        self._api     = ctypes.c_void_p()
        self._device  = ctypes.c_void_p()
        self._lock    = threading.Lock()
        self._gx      = 0.0
        self._gy      = 0.0
        self._last    = 0.0
        self._running = False
        self._kx = KalmanFilter1D(process_var=1e-4, measure_var=0.02)
        self._ky = KalmanFilter1D(process_var=1e-4, measure_var=0.02)

        self._connect()

    def _connect(self):
        dll = self._dll

        # API erstellen
        err = dll.tobii_api_create(ctypes.byref(self._api), None, None)
        if err != 0:
            raise RuntimeError(f"tobii_api_create fehlgeschlagen (code {err})")

        # Gerät suchen
        urls: list[str] = []

        @_UrlReceiver
        def _recv_url(url, _):
            urls.append(url.decode())

        dll.tobii_enumerate_local_device_urls(self._api, _recv_url, None)

        if not urls:
            dll.tobii_api_destroy(self._api)
            raise RuntimeError("Kein Tobii-Gerät gefunden — USB angeschlossen und Tobii Experience gestartet?")

        print(f"[Tobii] Gerät: {urls[0]}")

        err = dll.tobii_device_create(
            self._api, urls[0].encode(), ctypes.c_int(1), ctypes.byref(self._device)
        )
        if err != 0:
            dll.tobii_api_destroy(self._api)
            raise RuntimeError(f"tobii_device_create fehlgeschlagen (code {err})")

        # Gaze-Daten abonnieren
        self._cb = _GazeCallback(self._on_gaze)   # Referenz halten!
        err = dll.tobii_gaze_point_subscribe(self._device, self._cb, None)
        if err != 0:
            raise RuntimeError(f"tobii_gaze_point_subscribe fehlgeschlagen (code {err})")

        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        print("[Tobii] Stream Engine aktiv")

    def _on_gaze(self, gaze_ptr, _user):
        g = gaze_ptr.contents
        if g.validity != 1:
            return
        x, y = g.position_xy[0], g.position_xy[1]
        # [0,1] → [-1,+1],  Y invertiert: oben=+1
        gx = self._kx.update( (x - 0.5) * 2.0)
        gy = self._ky.update(-(y - 0.5) * 2.0)
        with self._lock:
            self._gx   = float(np.clip(gx, -1.5, 1.5))
            self._gy   = float(np.clip(gy, -1.5, 1.5))
            self._last = time.time()

    def _loop(self):
        while self._running:
            self._dll.tobii_device_process_callbacks(self._device)
            time.sleep(0.004)   # ~250 Hz

    def estimate(self, frame=None):
        with self._lock:
            if time.time() - self._last > 0.4:
                return None, None
            return self._gx, self._gy

    def reset_filter(self):
        self._kx.reset()
        self._ky.reset()

    def disconnect(self):
        self._running = False
        try:
            self._dll.tobii_gaze_point_unsubscribe(self._device)
            self._dll.tobii_device_destroy(self._device)
            self._dll.tobii_api_destroy(self._api)
        except Exception:
            pass
