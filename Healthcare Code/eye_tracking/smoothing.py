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
