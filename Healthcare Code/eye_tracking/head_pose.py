"""
Head pose estimation from MediaPipe FaceLandmarker transformation matrix.

The FaceLandmarker (Tasks API) provides a 4x4 facial transformation matrix
directly — no manual solvePnP needed.

Output:
  pitch : float  degrees, positive = head up,    negative = head down
  yaw   : float  degrees, positive = turned right, negative = turned left
  roll  : float  degrees
"""

import numpy as np


class HeadPoseEstimator:
    """
    Derives Euler angles from the facial transformation matrix returned by
    GazeEstimator (stored as gaze_est._transform_matrix).

    Usage:
        pitch, yaw, roll = head_est.estimate(gaze_est._transform_matrix)
    """

    def __init__(self):
        self.neutral_pitch: float = 0.0

    def estimate(self, transform_matrix):
        """
        transform_matrix : 4x4 numpy array from FaceLandmarker, or None.
        Returns (pitch, yaw, roll) in degrees, or (None, None, None).
        """
        if transform_matrix is None:
            return None, None, None

        m = np.array(transform_matrix)
        if m.shape != (4, 4):
            return None, None, None

        r = m[:3, :3]
        sy = np.sqrt(r[0, 0]**2 + r[1, 0]**2)
        if sy > 1e-6:
            roll  = np.degrees(np.arctan2( r[2, 1], r[2, 2]))
            pitch = np.degrees(np.arctan2(-r[2, 0], sy))
            yaw   = np.degrees(np.arctan2( r[1, 0], r[0, 0]))
        else:
            roll  = np.degrees(np.arctan2(-r[1, 2], r[1, 1]))
            pitch = np.degrees(np.arctan2(-r[2, 0], sy))
            yaw   = 0.0

        pitch -= self.neutral_pitch
        return float(pitch), float(yaw), float(roll)

    def calibrate_neutral(self, transform_matrix):
        """Call while user holds head level to zero the pitch offset."""
        p, _, _ = self.estimate(transform_matrix)
        if p is not None:
            self.neutral_pitch += p
