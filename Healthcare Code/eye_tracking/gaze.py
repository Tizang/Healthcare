"""
Iris-based gaze estimation using MediaPipe Face Mesh.

Output:
  gaze_x : float in [-1, +1]   negative = looking left, positive = right
  gaze_y : float in [-1, +1]   negative = looking up,   positive = down

Strategy:
  MediaPipe Face Mesh with refine_landmarks=True exposes iris landmarks
  (468–472 = left iris, 473–477 = right iris).
  We calculate the iris centre offset relative to the eye bounding box to
  derive a normalised gaze vector.
"""

import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False

# MediaPipe landmark indices
# Left eye corners / boundary
LEFT_EYE_INNER  = 133
LEFT_EYE_OUTER  = 33
LEFT_EYE_TOP    = 159
LEFT_EYE_BOTTOM = 145

# Right eye corners / boundary
RIGHT_EYE_INNER  = 362
RIGHT_EYE_OUTER  = 263
RIGHT_EYE_TOP    = 386
RIGHT_EYE_BOTTOM = 374

# Iris centres (requires refine_landmarks=True)
LEFT_IRIS_CENTER  = 468
RIGHT_IRIS_CENTER = 473


class GazeEstimator:
    """
    Estimates normalised gaze direction from a single webcam frame.

    Usage:
        estimator = GazeEstimator()
        gaze_x, gaze_y = estimator.estimate(bgr_frame)
        # None, None returned when no face detected
    """

    def __init__(self, min_detection_confidence: float = 0.7):
        if not MEDIAPIPE_OK:
            raise ImportError(
                "mediapipe is not installed. Run: pip install mediapipe"
            )
        mp_face = mp.solutions.face_mesh
        self._mesh = mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,          # enables iris landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )
        self._landmarks = None

    def estimate(self, bgr_frame: np.ndarray):
        """
        Returns (gaze_x, gaze_y) or (None, None) if no face detected.
        gaze_x/y are in range approximately [-1, +1].
        """
        import cv2
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._mesh.process(rgb)

        if not results.multi_face_landmarks:
            self._landmarks = None
            return None, None

        lm = results.multi_face_landmarks[0].landmark
        self._landmarks = lm
        h, w = bgr_frame.shape[:2]

        def pt(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        # Compute per-eye gaze offset, then average
        gaze_x, gaze_y = self._eye_gaze(pt, "left")
        gaze_xr, gaze_yr = self._eye_gaze(pt, "right")

        avg_x = (gaze_x + gaze_xr) / 2.0
        avg_y = (gaze_y + gaze_yr) / 2.0

        return float(avg_x), float(avg_y)

    def _eye_gaze(self, pt, side: str):
        if side == "left":
            inner, outer = LEFT_EYE_INNER, LEFT_EYE_OUTER
            top, bottom  = LEFT_EYE_TOP,   LEFT_EYE_BOTTOM
            iris         = LEFT_IRIS_CENTER
        else:
            inner, outer = RIGHT_EYE_INNER, RIGHT_EYE_OUTER
            top, bottom  = RIGHT_EYE_TOP,   RIGHT_EYE_BOTTOM
            iris         = RIGHT_IRIS_CENTER

        eye_center = (pt(inner) + pt(outer)) / 2.0
        eye_width  = np.linalg.norm(pt(outer) - pt(inner))
        eye_height = np.linalg.norm(pt(bottom) - pt(top))

        if eye_width < 1 or eye_height < 1:
            return 0.0, 0.0

        iris_pt = pt(iris)
        # Normalise to [-1, +1]
        gx = (iris_pt[0] - eye_center[0]) / (eye_width  / 2.0)
        gy = (iris_pt[1] - eye_center[1]) / (eye_height / 2.0)

        return gx, gy

    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        """Draw eye and iris landmarks on frame (in-place, returns frame)."""
        if self._landmarks is None:
            return frame
        import cv2
        h, w = frame.shape[:2]
        lm = self._landmarks
        indices = [
            LEFT_EYE_INNER, LEFT_EYE_OUTER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
            RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
            LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER,
        ]
        for idx in indices:
            x, y = int(lm[idx].x * w), int(lm[idx].y * h)
            color = (0, 255, 0) if idx in (LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER) else (255, 0, 0)
            cv2.circle(frame, (x, y), 3, color, -1)
        return frame

    def close(self):
        self._mesh.close()
