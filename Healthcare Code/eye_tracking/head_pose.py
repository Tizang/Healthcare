"""
Head pose estimation from MediaPipe Face Mesh landmarks.

Uses solvePnP with 6 well-separated face landmarks and a generic 3D face model
to recover pitch (nod up/down), yaw (turn left/right), roll (tilt).

Output:
  pitch : float  degrees, negative = head down, positive = head up
  yaw   : float  degrees, negative = turned left, positive = turned right
  roll  : float  degrees
"""

import numpy as np

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

# 6-point 3D reference model (mm, canonical face)
# Nose tip, Chin, Left eye outer, Right eye outer, Left mouth, Right mouth
_MODEL_POINTS_3D = np.array([
    [  0.0,    0.0,   0.0],   # Nose tip (1)
    [  0.0,  -63.6, -12.5],   # Chin (152)
    [-43.3,   32.7, -26.0],   # Left eye outer corner (33)
    [ 43.3,   32.7, -26.0],   # Right eye outer corner (263)
    [-28.9,  -28.9, -24.1],   # Left mouth corner (61)
    [ 28.9,  -28.9, -24.1],   # Right mouth corner (291)
], dtype=np.float64)

# Corresponding MediaPipe Face Mesh landmark indices
_LM_INDICES = [1, 152, 33, 263, 61, 291]

# Neutral pitch when the user sits normally (calibrated per session)
_PITCH_NEUTRAL = 0.0


class HeadPoseEstimator:
    """
    Estimates head orientation angles from MediaPipe face landmarks.

    Usage:
        estimator = HeadPoseEstimator()
        pitch, yaw, roll = estimator.estimate(landmarks, frame_shape)
        # returns (None, None, None) on failure
    """

    def __init__(self):
        if not CV2_OK:
            raise ImportError("opencv-python is not installed.")
        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs = np.zeros((4, 1))
        self.neutral_pitch: float = 0.0

    def _get_camera_matrix(self, frame_shape):
        h, w = frame_shape[:2]
        focal = w
        return np.array([
            [focal, 0,     w / 2],
            [0,     focal, h / 2],
            [0,     0,     1   ],
        ], dtype=np.float64)

    def estimate(self, landmarks, frame_shape):
        """
        landmarks : list of MediaPipe NormalizedLandmark
        frame_shape: (height, width, ...)
        Returns (pitch, yaw, roll) in degrees, or (None, None, None).
        """
        if landmarks is None:
            return None, None, None

        h, w = frame_shape[:2]
        img_pts = np.array([
            [landmarks[i].x * w, landmarks[i].y * h]
            for i in _LM_INDICES
        ], dtype=np.float64)

        cam = self._get_camera_matrix(frame_shape)

        ok, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS_3D, img_pts, cam, self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None, None, None

        rot_mat, _ = cv2.Rodrigues(rvec)
        # Decompose rotation matrix to Euler angles
        sy = np.sqrt(rot_mat[0, 0]**2 + rot_mat[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            roll  = np.degrees(np.arctan2( rot_mat[2, 1], rot_mat[2, 2]))
            pitch = np.degrees(np.arctan2(-rot_mat[2, 0], sy))
            yaw   = np.degrees(np.arctan2( rot_mat[1, 0], rot_mat[0, 0]))
        else:
            roll  = np.degrees(np.arctan2(-rot_mat[1, 2], rot_mat[1, 1]))
            pitch = np.degrees(np.arctan2(-rot_mat[2, 0], sy))
            yaw   = 0.0

        # Apply neutral offset
        pitch -= self.neutral_pitch

        return float(pitch), float(yaw), float(roll)

    def calibrate_neutral(self, landmarks, frame_shape):
        """
        Call once while the user holds their head level.
        Stores the current pitch as the neutral reference.
        """
        pitch, _, _ = self.estimate(landmarks, frame_shape)
        if pitch is not None:
            self.neutral_pitch = pitch + self.neutral_pitch
