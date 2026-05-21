"""
Signal smoothing to reduce webcam jitter.
"""

from collections import deque
from typing import Optional
import numpy as np


class ExponentialSmoother:
    """
    Single exponential smoothing (EMA).
    alpha close to 1 = fast / more jitter
    alpha close to 0 = slow / very smooth
    """

    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self._value: Optional[float] = None

    def smooth(self, value: float) -> float:
        if self._value is None:
            self._value = value
        else:
            self._value = self.alpha * value + (1 - self.alpha) * self._value
        return self._value

    def reset(self):
        self._value = None


class MovingAverageSmoother:
    """Simple moving average over a fixed window."""

    def __init__(self, window: int = 8):
        self._buf: deque = deque(maxlen=window)

    def smooth(self, value: float) -> float:
        self._buf.append(value)
        return float(np.mean(self._buf))

    def reset(self):
        self._buf.clear()


class Vec2Smoother:
    """Smooth a 2D (x, y) signal independently."""

    def __init__(self, alpha: float = 0.25):
        self._x = ExponentialSmoother(alpha)
        self._y = ExponentialSmoother(alpha)

    def smooth(self, x: float, y: float):
        return self._x.smooth(x), self._y.smooth(y)

    def reset(self):
        self._x.reset()
        self._y.reset()


class AdaptiveVec2Smoother:
    """
    Velocity-adaptive smoother for gaze.

    During fixation (slow, small movements): heavy smoothing filters out
    micro-tremor so the cursor stays still.

    During a saccade (fast intentional eye movement): light smoothing lets
    the cursor respond immediately with minimal lag.

    Parameters
    ----------
    alpha_still   : EMA weight when eye velocity < threshold (heavy smooth)
    alpha_moving  : EMA weight when eye velocity >= threshold (light smooth)
    velocity_threshold : movement per frame that switches to fast mode
    """

    def __init__(
        self,
        alpha_still: float = 0.15,
        alpha_moving: float = 0.6,
        velocity_threshold: float = 0.04,
    ):
        self.alpha_still = alpha_still
        self.alpha_moving = alpha_moving
        self.velocity_threshold = velocity_threshold
        self._prev: Optional[np.ndarray] = None
        self._smoothed: Optional[np.ndarray] = None

    def smooth(self, x: float, y: float):
        val = np.array([x, y])

        if self._smoothed is None:
            self._prev = val.copy()
            self._smoothed = val.copy()
            return float(val[0]), float(val[1])

        velocity = float(np.linalg.norm(val - self._prev))
        alpha = self.alpha_moving if velocity > self.velocity_threshold else self.alpha_still

        self._smoothed = alpha * val + (1.0 - alpha) * self._smoothed
        self._prev = val.copy()
        return float(self._smoothed[0]), float(self._smoothed[1])

    def reset(self):
        self._prev = None
        self._smoothed = None
