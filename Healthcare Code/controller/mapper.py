"""
Maps normalised gaze & head-pitch values to SOLOASSIST arm speed commands.

Input ranges:
  gaze_x    : [-1, +1]   (negative=left, positive=right)
  gaze_y    : [-1, +1]   (negative=up,   positive=down)
  head_pitch: degrees    (negative=head down → retract, positive=head up → extend)

Output:
  speed_lr  : [-MAX_SPEED, +MAX_SPEED]
  speed_ud  : [-MAX_SPEED, +MAX_SPEED]
  speed_io  : [-MAX_SPEED, +MAX_SPEED]
"""

import math
from dataclasses import dataclass, field


@dataclass
class MapperConfig:
    # Gaze deadzone — gaze values smaller than this are treated as centre
    gaze_deadzone: float = 0.08

    # Head-pitch deadzone (degrees from neutral)
    pitch_deadzone: float = 4.0

    # Gain: how fast to move for a full-deflection gaze/pitch signal
    # lr/ud gain maps gaze ±1 → speed ±max_speed
    max_speed_lr: int = 150
    max_speed_ud: int = 150
    max_speed_io: int = 100

    # Head pitch range at which max io speed is reached (degrees)
    pitch_max_deg: float = 20.0

    # Calibration offsets (set during calibration procedure)
    gaze_x_offset: float = 0.0
    gaze_y_offset: float = 0.0


class GazeToArmMapper:
    """
    Translates gaze + head pose signals into arm velocity commands.

    Usage:
        mapper = GazeToArmMapper()
        speed_lr, speed_ud, speed_io = mapper.map(gaze_x, gaze_y, head_pitch)
    """

    def __init__(self, config: MapperConfig | None = None):
        self.config = config or MapperConfig()

    def map(
        self,
        gaze_x: float | None,
        gaze_y: float | None,
        head_pitch: float | None,
    ) -> tuple[int, int, int]:
        """Returns (speed_lr, speed_ud, speed_io)."""
        cfg = self.config

        # --- Left / Right from gaze X ---
        if gaze_x is not None:
            gx = gaze_x - cfg.gaze_x_offset
            gx = self._apply_deadzone(gx, cfg.gaze_deadzone)
            speed_lr = int(self._scale(gx, -1.0, 1.0, -cfg.max_speed_lr, cfg.max_speed_lr))
        else:
            speed_lr = 0

        # --- Up / Down from gaze Y ---
        if gaze_y is not None:
            gy = gaze_y - cfg.gaze_y_offset
            gy = self._apply_deadzone(gy, cfg.gaze_deadzone)
            # gaze_y positive = looking down → arm down (negative ud)
            speed_ud = int(self._scale(gy, -1.0, 1.0, cfg.max_speed_ud, -cfg.max_speed_ud))
        else:
            speed_ud = 0

        # --- In / Out from head pitch ---
        # Pitch is in degrees, so deadzone uses a range-aware formulation
        if head_pitch is not None and abs(head_pitch) >= cfg.pitch_deadzone:
            sign = math.copysign(1.0, head_pitch)
            effective     = abs(head_pitch) - cfg.pitch_deadzone
            effective_max = cfg.pitch_max_deg - cfg.pitch_deadzone
            ratio = min(1.0, effective / effective_max) if effective_max > 0 else 1.0
            speed_io = int(sign * ratio * cfg.max_speed_io)
        else:
            speed_io = 0

        return speed_lr, speed_ud, speed_io

    @staticmethod
    def _apply_deadzone(value: float, zone: float) -> float:
        """Zero out values within ±zone, rescale rest to start from zero."""
        if abs(value) < zone:
            return 0.0
        sign = math.copysign(1.0, value)
        return sign * (abs(value) - zone) / (1.0 - zone)

    @staticmethod
    def _scale(value: float, in_lo: float, in_hi: float, out_lo: float, out_hi: float) -> float:
        value = max(in_lo, min(in_hi, value))
        ratio = (value - in_lo) / (in_hi - in_lo)
        return out_lo + ratio * (out_hi - out_lo)

    def get_direction_label(self, speed_lr: int, speed_ud: int, speed_io: int) -> str:
        """Human-readable direction string for debug overlay."""
        parts = []
        thresh = 10
        if speed_ud >  thresh: parts.append("UP")
        if speed_ud < -thresh: parts.append("DOWN")
        if speed_lr < -thresh: parts.append("LEFT")
        if speed_lr >  thresh: parts.append("RIGHT")
        if speed_io >  thresh: parts.append("EXTEND")
        if speed_io < -thresh: parts.append("RETRACT")
        return " + ".join(parts) if parts else "CENTRE"
