class KalmanFilter1D:
    """Simple scalar Kalman filter."""

    def __init__(self, process_var: float = 2e-4, measure_var: float = 0.04):
        self.x = 0.0
        self.p = 1.0
        self.q = process_var
        self.r = measure_var

    def update(self, z: float) -> float:
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (z - self.x)
        self.p *= (1.0 - k)
        return self.x

    def reset(self, value: float = 0.0):
        self.x = value
        self.p = 1.0
