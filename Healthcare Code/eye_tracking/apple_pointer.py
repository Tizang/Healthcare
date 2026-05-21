"""
macOS system cursor → normalised gaze coordinates.

Reads the current mouse cursor position via CoreGraphics (Quartz).
When Apple Head Pointer or Eye Tracking is active in Accessibility settings,
the cursor IS the gaze — this gives much better accuracy than MediaPipe iris.

Output:
  gaze_x : float in [-1, +1]   negative = left,  positive = right
  gaze_y : float in [-1, +1]   negative = up,    positive = down

Requires:  pip install pyobjc-framework-Quartz
"""

try:
    from Quartz import CGEventCreate, CGEventGetLocation
    _QUARTZ_OK = True
except ImportError:
    _QUARTZ_OK = False


class ApplePointerGaze:
    """
    Reads macOS cursor position and converts to normalised [-1, +1] gaze coords.

    Usage:
        tracker = ApplePointerGaze(screen_w=2056, screen_h=1329)
        gaze_x, gaze_y = tracker.get()
    """

    def __init__(self, screen_w: int, screen_h: int):
        if not _QUARTZ_OK:
            raise ImportError(
                "pyobjc-framework-Quartz is required for Apple pointer mode.\n"
                "Install with:  pip install pyobjc-framework-Quartz"
            )
        self.screen_w = screen_w
        self.screen_h = screen_h

    def get(self) -> tuple[float, float]:
        """Returns (gaze_x, gaze_y) in [-1, +1], mapped from cursor position."""
        event = CGEventCreate(None)
        pos = CGEventGetLocation(event)
        # macOS origin is bottom-left; Y increases upward → flip for screen coords
        gaze_x = (pos.x / self.screen_w) * 2.0 - 1.0
        gaze_y = (pos.y / self.screen_h) * 2.0 - 1.0   # bottom-left origin → up=negative
        return float(gaze_x), float(gaze_y)

    @staticmethod
    def available() -> bool:
        return _QUARTZ_OK
