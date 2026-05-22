"""
SOLOASSIST II TCP/IP Controller
Protocol reverse-engineered from RemoteHostTestMain.vb (AKTORmed demo application).

Packet format (client → server):
  [4-byte int32 LE: payload_length] [1-byte version=1] [1-byte packet_counter]
  [2-byte int16 LE: command] [data bytes...]
  where payload_length = 4 + len(data)

Server response format (same header structure).
"""

import socket
import struct
import threading
import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


# Client command codes from RH_CLASSLIB.vb
class Cmd:
    GET_TP         = 0x01
    SET_TP         = 0x02
    GET_HP         = 0x03
    GET_SENSOR     = 0x04
    GET_AXIS_POS   = 0x05
    MOVE_AXIS      = 0x06
    MOVE_POLAR     = 0x07
    POS_CARTESIAN  = 0x08
    POS_AXIS       = 0x09
    QUIT           = 0xFE
    STOP_ALL       = 0xFF


# Server response codes
class Resp:
    TP_POS          = 0x41
    HP_POS          = 0x43
    SENSOR_VAL      = 0x44
    AXIS_POSITION   = 0x45
    POSITIONING_DONE = 0x46
    NOT_CONNECTED   = 0x47
    JOYSTICK_INFO   = 0x51


# Speed limits — adjust for safety during testing
MAX_SPEED = 125   # protocol range: -125...+125%
MIN_SPEED = -125


@dataclass
class ArmPosition:
    x: int = 0
    y: int = 0
    z: int = 0


class SoloAssistController:
    """
    High-level Python interface to the SOLOASSIST II robotic arm via TCP/IP.

    Usage:
        arm = SoloAssistController("192.168.1.100", 5000)
        arm.connect()
        arm.move_polar(lr=50, ud=0, io=0)   # move right
        arm.stop()
        arm.disconnect()
    """

    def __init__(self, ip: str, port: int, timeout: float = 5.0):
        self.ip = ip
        self.port = port
        self.timeout = timeout

        self._sock: Optional[socket.socket] = None
        self._connected = False
        self._packet_counter = 0
        self._lock = threading.Lock()

        self._reader_thread: Optional[threading.Thread] = None
        self._running = False

        self.current_hp = ArmPosition()

    # ------------------------------------------------------------------ #
    #  Connection                                                          #
    # ------------------------------------------------------------------ #

    def connect(self) -> bool:
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(self.timeout)
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock.connect((self.ip, self.port))
            self._connected = True
            self._running = True
            self._reader_thread = threading.Thread(
                target=self._read_loop, daemon=True
            )
            self._reader_thread.start()
            log.info("Connected to SOLOASSIST at %s:%d", self.ip, self.port)
            return True
        except Exception as exc:
            log.error("Connection failed: %s", exc)
            self._connected = False
            return False

    def disconnect(self):
        self._running = False
        self._connected = False
        try:
            if self._sock:
                self._send(Cmd.QUIT, b"")
                self._sock.close()
        except Exception:
            pass
        log.info("Disconnected from SOLOASSIST")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------ #
    #  Movement commands                                                   #
    # ------------------------------------------------------------------ #

    def move_polar(self, lr: int, ud: int, io: int):
        """
        Continuous velocity command in polar/spherical coordinates.

        lr : int  left/right speed   (+= right, -= left)
        ud : int  up/down speed      (+= up,    -= down)
        io : int  in/out speed       (+= out,   -= in)

        Call repeatedly (e.g. every 50 ms) while movement is desired.
        Call stop() or move_polar(0,0,0) to halt.
        """
        lr = max(MIN_SPEED, min(MAX_SPEED, int(lr)))
        ud = max(MIN_SPEED, min(MAX_SPEED, int(ud)))
        io = max(MIN_SPEED, min(MAX_SPEED, int(io)))
        data = struct.pack("<hhh", lr, ud, io)
        self._send(Cmd.MOVE_POLAR, data)

    def stop(self):
        """Immediate stop of all movement."""
        self._send(Cmd.STOP_ALL, b"")

    def get_position(self) -> ArmPosition:
        """Request current hand-piece position (async — updates self.current_hp)."""
        self._send(Cmd.GET_HP, b"")
        return self.current_hp

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _send(self, command: int, data: bytes):
        if not self._connected:
            return
        with self._lock:
            try:
                self._packet_counter = (self._packet_counter % 255) + 1
                payload_length = 4 + len(data)
                # Header: uint32 payload_len, uint8 version=1, uint8 counter, int16 command
                header = struct.pack(
                    "<IBBh",
                    payload_length,
                    1,
                    self._packet_counter,
                    command,
                )
                self._sock.sendall(header + data)
            except Exception as exc:
                log.warning("Send failed: %s", exc)
                self._connected = False

    def _read_loop(self):
        """Background thread: reads server responses and updates state."""
        buf = bytearray()
        while self._running:
            try:
                chunk = self._sock.recv(256)
                if not chunk:
                    break
                buf.extend(chunk)

                # Parse complete messages
                while len(buf) >= 4:
                    payload_len = struct.unpack_from("<I", buf, 0)[0]
                    total_needed = 4 + payload_len
                    if len(buf) < total_needed:
                        break

                    payload = buf[4:total_needed]
                    buf = buf[total_needed:]

                    if len(payload) < 4:
                        continue

                    # version = payload[0]
                    # packet  = payload[1]
                    command = struct.unpack_from("<h", payload, 2)[0]
                    pdata = payload[4:]

                    self._handle_response(command, pdata)

            except socket.timeout:
                continue
            except Exception as exc:
                log.debug("Read loop error: %s", exc)
                break

        self._connected = False

    def _handle_response(self, command: int, data: bytes):
        if command == Resp.HP_POS and len(data) >= 6:
            self.current_hp.x = struct.unpack_from("<h", data, 0)[0]
            self.current_hp.y = struct.unpack_from("<h", data, 2)[0]
            self.current_hp.z = struct.unpack_from("<h", data, 4)[0]
        elif command == Resp.NOT_CONNECTED:
            log.warning("RemoteHost is not connected to SOLOASSIST hardware")
        elif command == Resp.POSITIONING_DONE:
            log.debug("Positioning done")
